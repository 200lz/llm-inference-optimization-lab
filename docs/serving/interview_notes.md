# Mini serving engine interview notes

## 30-second summary

I built a deterministic C++17 metadata simulator for LLM request serving, then added reproducible Python workload replay and service-level analysis. It compares FCFS with continuous DecodeFirst/FcfsMixed scheduling, models block KV capacity and collision-verified prefix caching, and reports SIMULATED TTFT, tail latency, throughput, and SLO goodput with full provenance. It deliberately has no real tensors or unsupported hardware claims.

## Two-minute explanation

Requests arrive at integer microsecond timestamps and move through waiting, prefill, decode, and finish states. The single-active engine gives a clear FCFS reference. The continuous engine plans full prefills and one decode token per active sequence under sequence, token, and KV budgets. A two-phase transaction reserves exact block IDs on copied state, validates the full plan, and only then commits. Exact-token prompts can share complete KV blocks; hash buckets are collision-verified, references follow request block tables, and unused cached blocks use deterministic LRU.

S6 adds six workload classes, four arrival models, strict JSONL schemas, a TSV-to-native runner boundary, provenance, nearest-rank percentiles, and goodput. The most important conclusion is conditional: prefix caching helps local workloads, a too-small KV pool stalls, and overload damages tails, but several nominal optimizations were neutral under the selected synthetic costs.

## Five-minute technical walkthrough

1. Python validates `serving-workload-v1`, orders equal arrivals by exact lexicographic external ID, maps them bidirectionally to stable integer IDs, and writes strict temporary TSV.
2. The native runner selects single-active FCFS or continuous batching and emits typed JSONL request, iteration, and summary records.
3. Continuous planning orders candidates by policy, checks token/sequence budgets, performs cache lookup, reserves deterministic evictions/allocations, and records deferrals.
4. Preparation copies requests, clock, statistics, and KV state. Exact reservation identity and invariants must match before no-throw publication.
5. Analysis derives nullable TTFT/TPOT/E2E, nearest-rank tails, rates, queue depth, cache/KV metrics, and conjunction-based SLO goodput.
6. Config and workload hashes plus repository/submodule revisions keep comparisons auditable. Every output says SIMULATED.

## Key design decisions

- Deterministic event/iteration simulation makes policy regressions exactly testable.
- FCFS is a simple reference; continuous batching exposes throughput/progress tradeoffs.
- Token and sequence budgets describe one plan; KV capacity describes persistent residency.
- Block ownership, provenance, and exact reservation IDs prevent aliasing and replay drift.
- Prefix keys include parent, tokens, namespace, and salt; exact verification handles collisions.
- Reference counts equal live request-table membership.
- LRU orders `(last_access_epoch, physical_block_id)`.
- Two-phase commit preserves strong exception guarantees across a whole iteration.
- Synthetic coefficients and outputs are an evidence boundary, not proxy hardware results.

## Likely questions

**Why simulate instead of using vLLM directly?** To isolate scheduling/cache invariants in a small dependency-free system and make timelines reproducible. It complements, rather than replaces, production-engine study.

**Why no NVIDIA GPU?** This phase studies policy metadata and reproducibility. Without access and calibration, GPU numbers would add unsupported claims.

**What does `max_num_sequences` mean?** The maximum distinct requests scheduled in one continuous iteration, not total resident requests.

**How does KVCapacity differ from token budget?** Token budget caps work in one plan. KV capacity caps physical blocks retained across plans.

**How does prefix caching reduce work?** A request references already computed full prompt blocks and schedules only its unmatched suffix.

**How are hash collisions handled?** Hash selects a bucket; parent, token IDs, namespace, and salt must match exactly.

**Why only full blocks?** It keeps shareable boundaries immutable and ownership/refcount rules explicit. Partial tails stay private.

**How do reference counts work?** They count live request block-table references; index membership is not a reference.

**How is eviction deterministic?** Only unreferenced cached blocks qualify, ordered by access epoch then physical ID.

**Why can a request stall?** All runnable candidates may need more non-evictable KV blocks than exist. No preemption or swapping is implemented.

**What would change on MN-Core-like hardware?** Cost coefficients, execution programs, batch shapes, block size, and residency policy would need public target-specific calibration; I make no proprietary assumptions.

**How would you validate on real hardware?** Instrument prefill/decode programs, record memory capacity and execution provenance, fit on training shapes, test held-out workloads, and report prediction error/sensitivity before policy claims.

**What are the limitations?** No tensors, tokenizer, real backend, networking, threads, distribution, preemption, swapping, chunked prefill, or production security/performance claim.

## Honest next steps

Calibrate against a declared backend, add held-out error bands, exercise longer steady-state traces that separate DecodeFirst from FcfsMixed, and only then consider an opt-in real backend adapter.
