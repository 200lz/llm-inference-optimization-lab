# Phase S4 deterministic block KV-cache metadata

## Scope

Phase S4 adds a metadata-only `KVCacheManager` and makes
`ContinuousBatchingEngine` respect resident KV capacity. It stores no K/V
tensors and performs no byte-level memory sizing. A configured block represents
only a fixed number of logical token slots, so every value and trace remains
`simulated`, not a GPU, MN-Core, llama.cpp, or other hardware measurement.

The S1/S2 single-active `SimulationEngine` is unchanged. S4 also adds no prefix
sharing, prefix cache, eviction, swapping, preemption, chunked prefill, model
execution, or vLLM PagedAttention feature-parity claim.

## Metadata and invariants

`KVCacheConfig` contains positive `total_num_blocks` and
`block_size_tokens`; capacity multiplication must fit `uint64_t`. The block
size defaults to 16 only at API defaults. Tests and the smoke scenario specify
both values explicitly.

Each stable physical block ID indexes one `PhysicalKVBlock`. S4 states are only
`Free` and `InUse`. A free block has no owner, zero references, zero valid
tokens, and occurs exactly once in the ordered free set. An in-use block occurs
in exactly one request table, names that request as owner, has
`reference_count == 1`, and has between one and `block_size_tokens` valid
tokens. No block is shared. Every block is either free or owned exactly once.

The manager owns `request_id -> ordered vector<block_id>`. Vector order is
logical token order; every block except the final block is full. Zero-token
requests have an empty vector but remain registered until one release.
Diagnostics are read-only.
Diagnostic block vectors, free-ID sets, and request tables are returned as
value snapshots. They remain valid and unchanged across later manager
mutations, assignments, and engine copy/swap commits.

The manager access epoch starts at zero. A successful nonempty prompt
allocation advances it once and assigns the new epoch to every allocated prompt
block. A successful decode append advances it once and assigns it to the
touched tail or new block. Registering a zero-token prompt does not advance the
epoch because no physical block is touched. Release resets released block
epochs to zero. Failed operations never advance the epoch. Consequently every
free block has epoch zero and every in-use block has an epoch in
`[1, manager_epoch]`.

## Allocation, growth, and release

Allocation always removes the lowest available block IDs. A prompt uses
`ceil(prompt_tokens / block_size_tokens)` blocks and records exact occupancy in
the tail. Appending a decode token increments a partial tail or allocates
exactly one lowest-ID block at a boundary. The engine appends KV before
publishing the generated-token increment.

Release resets every owned block and returns it to the ordered free set, making
future reuse deterministic. Normal completion releases after the iteration's
work is prepared and before the next plan. Active cancellation releases
synchronously; waiting cancellation has no KV table. A zero-output request
allocates its prompt and releases it in the same atomic iteration. Completion
and cancellation never release twice.

Allocation and release use copy-then-swap transactions. Failed allocation
leaves blocks, tables, represented tokens, utilization, and peaks unchanged;
the checked allocation-failure counter advances once only for an otherwise
valid prompt that lacks enough free blocks or a valid boundary decode that
needs a block when none is free. Duplicate/unknown requests, invalid metadata,
arithmetic or epoch overflow, C++ container failures, and invariant errors do
not increment it. If a countable failure cannot increment the counter, the
entire manager state is preserved. Copying all metadata is acceptable for this
small educational simulator, but is not a production allocator design.

## Planning and deferral

The planner uses this precedence for every candidate:

1. `TokenBudget` when remaining iteration token work is insufficient;
2. `SequenceBudget` when no iteration sequence slot remains;
3. `KVCapacity` when iteration budgets fit but resident block capacity does
   not.

Planning never mutates the manager. It starts with the current free-block count
and subtracts locally reserved prompt blocks and decode-boundary blocks, so
multiple candidates cannot reserve the same physical capacity. Requests
finishing in the plan do not donate their blocks to another candidate in that
same plan. Completed and cancelled work is released before the next plan.
There is no preemption: capacity-blocked work remains in its lifecycle state
and is reconsidered after capacity becomes available. A KV deferral statistic
counts once per deferred request per iteration; it is expected control flow
and does not increment manager allocation failures.

If every runnable candidate is deferred for KV capacity, the planner emits a
deferred-only plan. `run_next_iteration()` records a deterministic zero-duration
trace, increments stall and per-request KV-deferral statistics, performs no
backend call, and returns `IterationOutcome::Stalled`. Requests, KV metadata,
and the clock do not change, and the engine remains usable for an external
cancellation. `run()` returns `RunResult::Stalled` immediately rather than
spinning. This expected modeled stall is distinct from terminal `Failed`, which
continues to mean an invariant, arithmetic, backend, or transaction error.

`max_num_sequences` and KV capacity are independent:
`max_num_sequences` bounds scheduled work in one iteration, while
`total_num_blocks` bounds all resident request KV across iterations. More
decoding requests may remain resident than can be scheduled in one iteration.

## Transaction model and metrics

Iteration preparation copies the metadata manager, applies every planned prompt
allocation and decode append first, and only then applies all terminal releases
to the copy. Thus blocks released in iteration N first become allocatable while
planning iteration N+1. Preparation then computes
request values, checked statistics, time, and trace data. Commit publishes via
no-throw moves, assignments, and swaps. On failure the live requests, manager,
clock, trace, statistics, and iteration number remain unchanged and results
become unavailable.

The instantaneous metrics are:

```text
represented_tokens = sum(valid_token_count of in-use blocks)
internal_fragmentation_tokens =
    allocated_block_count * block_size_tokens - represented_tokens
block_utilization = allocated_block_count / total_num_blocks
peak_block_utilization = peak_allocated_block_count / total_num_blocks
```

Fragmentation is unused modeled token capacity, not actual KV bytes. Integer
capacity, represented-token, peak-count, failure-count, and deferral-count
accounting is checked. A correctly reserved plan normally has zero attempted
allocation failures.

## Limitations and S5 handoff

S4 has no real tensor storage, byte-level memory sizing, prefix sharing,
reference counts above one, cached-block state, eviction, swapping, preemption,
threads, networking, llama.cpp integration, model execution, or actual
hardware performance measurement. S5 can add immutable shared, block-aligned
prefix entries and reference-counted lifetime without changing S4's private
allocation invariants silently.
