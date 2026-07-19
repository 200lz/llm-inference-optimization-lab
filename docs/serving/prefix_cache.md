# Phase S5 deterministic block-level prefix caching

## Scope and policy

S5 adds a metadata-only prefix cache to the continuous-batching simulator. It
models token occupancy, physical block IDs, request block tables, cache keys,
references, and deterministic scheduling decisions. It stores no real K/V
tensors and has no byte-level memory model.

`Request` owns a private authoritative prompt representation. `prompt_length()`
returns the stored count for count-only requests and derives the length from
the immutable vector for exact requests; `has_exact_prompt_tokens()` and the
read-only `prompt_tokens()` accessor distinguish the two forms. The count-only
constructor remains the S1-S4 compatibility path and never reuses or publishes
a prefix. Empty exact
prompts are valid and create no entries. Only complete prompt blocks are
looked up or published. Partial prompt tails and decode-created blocks remain
private.

`PrefixCacheConfig` contains `enabled`, `model_namespace`, `cache_salt`, and
the only S5 eviction policy, `DeterministicLRU`. Namespace and salt are key
material. A namespace is an explicit compatibility boundary for model,
revision, tokenizer, and a future adapter identity; it is not a real model
name. Disabling the cache retains S4 private allocation and accounting.

## Key and collision design

Each `TokenBlockKey` contains the prior block hash, exactly one full block of
signed 32-bit token IDs, model namespace, and cache salt. Block zero uses the
root `0xcbf29ce484222325`. Its result becomes the next block's parent, so a
child depends on every prior complete block.

The project-local stable hash is 64-bit FNV-1a. Starting at the FNV offset
basis, it encodes these tagged fields in order:

1. tag `1`, then the parent as eight little-endian bytes;
2. tag `2`, then the namespace byte length as little-endian `uint64_t` and its
   bytes;
3. tag `3`, then the salt in the same length-prefixed form;
4. tag `4`, then token count as little-endian `uint64_t`, followed by every
   token's two's-complement `uint32_t` bits in four little-endian bytes.

Every byte is XORed and multiplied by FNV prime `1099511628211` modulo 2^64.
No `std::hash`, native object representation, or unchecked reinterpret cast is
used. The known vector `(root, [10,11,12,13], "model", "salt")` is
`0xd5392f858a10326e`.

The hash selects a bucket only. Every candidate is checked against parent,
exact token IDs, namespace, and salt. A mismatch increments collision
verification diagnostics and is not a hit. Tests inject a constant hash to
exercise a mismatch and an exact match in the same bucket. This non-cryptographic
hash is for deterministic simulation, not an adversarial multi-tenant cache.

## Lookup, acquisition, and publication

Read-only lookup walks full blocks from the root and stops on the first miss.
It returns matched blocks/tokens, physical IDs, next parent hash, eligible
tokens, collision checks, and disabled/miss/partial/`AllEligibleBlocksHit`
classification. `AllEligibleBlocksHit` means every complete cache-eligible
prompt block matched; a nonzero private prompt tail may still need prefill. It
does not change references, epochs, or cumulative metrics.

Evicting an ancestor does not recursively delete descendants. A descendant's
entry and physical block may remain cached, but lookup cannot cross the missing
parent. Recomputing the same exact parent recreates the stable hash-chain link
and can make the child reachable again. Unrelated collision entries remain
intact.

Acquisition validates the immutable lookup snapshot on a copied manager,
registers the request table, increments every matched reference, changes
`Cached` blocks to `InUse`, and assigns one new access epoch to the hit blocks.
Only the unmatched suffix is allocated and counted as simulated prefill work.
All checked metrics are prepared before publication.

At acquisition, the manager stores immutable request prompt provenance: the
request ID, exact-token/publication eligibility, original exact tokens when
available, total prompt length, eligible and matched full-block counts, the
newly computed full-block range, optional private-tail position, fixed decode
block start, and publication status. Publication takes only a request ID and
derives keys, parents, and block IDs from this stored record; there is no API
for supplying a second token vector.

After simulated prefill, each newly computed complete prompt block is offered
to the index. If the exact key already has a canonical physical block, that
mapping is retained and the new block stays private to its current request.
Index membership adds no reference. This avoids contradictory exact-key
mappings while preserving a request's already allocated block table.
Publication is idempotent after a successful identical publication. It rejects
count-only requests and exact requests with no newly computed eligible full
blocks. It never revisits matched shared blocks or crosses the recorded prompt
boundary into a partial tail or decode-created block.

## States, references, release, and eviction

- `Free`: in the ordered free set; no table/index membership, key, owner,
  reference, occupancy, or epoch.
- `InUse`: one or more request tables refer to it. Cacheable full shared blocks
  have no exclusive owner. Private blocks have one owner and one reference.
- `Cached`: a complete indexed prompt block with zero request references. Its
  physical occupancy and latest access epoch remain valid.

Every non-free block also has explicit provenance:
`SharedPromptFull`, `ComputedPromptFull`, `PrivatePromptTail`,
`PrivatePromptCountOnly`, or `PrivateDecode`. Only the two full-prompt forms
may carry a key or appear in the prefix index. This prevents a count-only full
block, a partial tail that later fills, or a decode-created full block from
becoming cacheable by accident.

Reference count is exactly live request-table membership. Cache index
membership is not a reference. Releasing a request decrements shared blocks;
the last reference becomes `Cached`. A private tail or decode block returns
directly to `Free`. Repeated release is rejected by the copy-then-swap
transaction.

Allocation evicts only when free IDs are insufficient. Eligible blocks are
unreferenced `Cached` blocks ordered by `(last_access_epoch, block_id)`. The
minimum shortage is evicted. Eviction removes the exact index mapping, clears
key/occupancy/access metadata, returns the ID to the ordered free set, and
increments the checked eviction metric. Referenced blocks are never eligible;
there is no preemption.

A hit assigns a new epoch to all matched blocks. Each successful new cache
publication assigns a new epoch. Miss-only lookup does not advance the global
epoch. Release preserves cacheable epochs; free resets to zero. Copy-then-swap
means failed operations do not advance epochs.

## Cache-aware scheduling and transactions

At an iteration boundary the planner visits candidates sequentially in policy
order. Each candidate queries the current local simulated cache state after all
earlier selected reservations and evictions. A deferred candidate changes no
references, protection, eviction eligibility, access epochs, or lookup/reuse
metrics. Token cost is `original_prompt_tokens - matched_prefix_tokens`; an
`AllEligibleBlocksHit` lookup may
therefore have zero prefill work but still consumes one sequence slot and
advances the lifecycle. Only suffix blocks need new physical IDs. The local
manager copy maintains deterministic free-ID and LRU state across exact
reservations, so an ID cannot be allocated or evicted twice.

Only a selected candidate protects and acquires its exact matched IDs. Its
reservation records exact suffix or decode-growth allocation IDs and exact LRU
victim IDs. Earlier selected evictions are immediately removed from local
lookup state, so later candidates are re-looked-up and their token work and
suffix demand are recomputed.

Preparation replays the recorded work order on a copied manager. Every exact-ID
operation requires equality between planned and deterministic physical IDs; it
does not select different victims. All suffix/decode growth is applied before
new prompt blocks are published, so same-plan misses remain independent and a
new entry becomes visible in the next iteration. Completion release remains
after all planned allocation and publication.

With caching disabled, exact-token requests use the same private count-only KV
path as S4: no lookup, sharing, insertion, cached eviction, or prefix metric
activity occurs. Both prompt forms reject lengths above `max_batched_tokens`
synchronously. With caching enabled, an oversized exact prompt is admitted only
when a committed hit reduces unmatched prefill work to the token budget;
otherwise it is rejected as an oversized unmatched prefill. Deferred-only
plans are `KVCapacity`-only.

Preparation also copies requests, statistics, clock, and trace. Any lookup,
reference, eviction, epoch, timestamp, statistics, or trace-preparation error
marks the engine terminally failed without publishing the copied state. The
live manager, index, references, free pool, request state, clock, trace, and
iteration number remain unchanged.

Fix C verifies this transaction boundary with complete test-only snapshots of
manager configuration, every physical block field, free IDs, request tables,
immutable prompt provenance, collision buckets and canonical mappings,
physical accounting, epochs, and every prefix metric. Engine snapshots add the
authoritative request representations, arrival set, configuration, clock,
iteration number, full physical-ID-bearing trace, all statistics, manager
snapshot, and failure state. Point-specific overflow and preparation seams are
applied to copied state; they are verification hooks and add no serving feature.

An independent manager shadow implements its own tagged FNV-1a encoding, parent
chain, collision buckets, exact canonical map, block/reference/provenance state,
free-ID allocation, LRU ordering, epochs, publication, decode privacy, release,
and accounting. Its forced manager edge cases and documented fixed seed
`0x5eedc0de` are compared after every operation. Disabled exact-token routing,
active cancellation, and KV stall/recovery are covered separately by focused
engine integration and deterministic replay tests. Repeated cache-aware
DecodeFirst and FcfsMixed workloads compare complete request, trace, statistic,
KV, epoch, metric, and `RunResult` state. This is deterministic simulator
validation, not a production readiness claim.

Invariant validation checks enum domains, free/in-use/cached provenance rules,
physical occupancy and references, nonempty index buckets, exact key length,
namespace, salt, configured stable hash and bucket placement, canonical-key
uniqueness, request prompt/decode boundaries, table/provenance agreement, and
exact-token parent chains. Lookup, publication, release, and eviction reject a
corrupt manager before mutation.

## Physical and logical metrics

Physical metrics count each resident block once, including cached blocks:
allocated/free/cached blocks, referenced shared blocks, represented physical
tokens, fragmentation, and utilization. Sharing never multiplies represented
tokens.

Logical metrics count request work: original prompt tokens, matched tokens,
scheduled prefill tokens, cache lookups/hits/misses, reused requests, collision
checks, and saved simulated prefill tokens. The token hit rate is
`matched_token_count / total_cache_eligible_prompt_tokens_looked_up`, or
undefined when no complete exact-token block was looked up. “Saved” means
saved simulated prefill work, not measured latency.

Trace `PrefillWork` snapshots include original, matched, and scheduled tokens;
matched, newly allocated, and evicted IDs; and hit classification. They are
durable values. Decode work records boundary-growth allocations and evictions,
and plan-wide lists make planned, committed, and traced identity comparable.
All output remains labeled SIMULATED.

## Implications for accelerator-oriented serving engines

Prefix reuse can reduce repeated prefill computation, while shared physical KV
blocks can reduce modeled memory demand. These benefits require cache metadata,
request scheduling, residency, and eviction to agree at the same transaction
boundary. Workload locality determines whether LRU retains useful prefixes;
DRAM capacity and bandwidth can influence whether metadata lookup and retained
KV state are worthwhile.

The deterministic simulator is not a direct MN-Core L performance claim and
contains no proprietary MN-Core implementation assumptions. A useful target
cost model would need hardware calibration. Compiler, runtime, memory-system,
and kernel integration could change the preferred block size, cache policy,
and scheduling tradeoffs.

## Limitations

S5 has no real K/V tensors, byte layout, tokenizer, model execution, GPU or
MN-Core measurement, chunked prefill, prefix reuse for partial blocks, decode
sharing, radix tree, distributed cache, preemption, swapping, threads,
HTTP/JSON, llama.cpp adapter, PagedAttention, or RadixAttention. It makes no
feature-parity claim with vLLM or SGLang.
