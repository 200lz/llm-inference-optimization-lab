# Phase S3 deterministic continuous batching

## Scope and architecture

Phase S3 adds `ContinuousBatchingEngine` beside, rather than inside,
`SimulationEngine`. The existing S1/S2 engine remains the single-active FCFS
reference and retains its event queue, scheduler epochs, backend rejection,
terminal failure, and narrow cancelled-stale-event rules. The S3 engine owns
iteration planning, multiple active decodes, request state, simulated time, the
plan trace, and cumulative batch statistics.

Both paths use `Request`, `SimulationClock`, checked signed 64-bit arithmetic,
and `SimulatedBackend`. S3 adds a mixed-batch cost method without changing the
S1/S2 prefill and decode methods. This avoids ambiguously combining a
one-request event machine with an atomic multi-request iteration.

This is a fully **SIMULATED** path. A batch is planned and applied
deterministically by one thread of control; it is not concurrent execution.

## Configuration and admission

`ContinuousBatchingConfig` contains `max_num_sequences` (default 8),
`max_batched_tokens` (default 512), and strongly typed `SchedulingPolicy`
(`DecodeFirst` or `FcfsMixed`). Both budgets must be at least one. Tests and
examples pass explicit values.

`max_num_sequences` limits work scheduled in one iteration. It is not a limit
on all resident `Decoding` requests and is not a KV-cache residency limit.
Active decodes that are deferred remain in `Decoding` and are reconsidered in
later iterations. Phase S4 introduces independent KV-block capacity and
residency constraints.

Each prefill consumes its complete `prompt_token_count`; each decode consumes
one token unit. Chunked prefill is not implemented. Submission therefore
rejects a prompt larger than `max_batched_tokens` synchronously, with an error
naming the exceeded budget and missing chunked-prefill support. Rejection does
not mutate engine state.

## Scheduling iteration lifecycle

One `run_next_iteration()` call performs this deterministic boundary:

1. make arrivals at or before the current timestamp eligible, ordered by
   `(arrival_time_us, request_id)`;
2. exclude requests completed or externally cancelled at the prior boundary;
3. collect active `Decoding` candidates;
4. collect arrived `Waiting` prefill candidates;
5. build one immutable `BatchPlan` under both budgets;
6. estimate one mixed-batch duration;
7. atomically apply each full prefill and one token per selected decode;
8. assign lifecycle timestamps at the iteration start or end;
9. advance the checked integer simulation clock;
10. repeat until no runnable or future work remains.

An arrival strictly inside a batch becomes eligible at the next boundary and
never executes before `arrival_time_us`. When only future arrivals remain, the
engine records an explicit empty/idle plan and jumps to the next timestamp.
There is no wall-clock read, sleep, thread, or equal-time submission-order
dependency.

Iteration execution uses a two-phase commit. Preparation validates every
selected request on a copy, computes the checked end timestamp, resulting
request records, every cumulative statistic, the next iteration number, and a
complete trace entry, then reserves trace capacity. Only after all potentially
throwing preparation succeeds does a no-allocation commit publish request
updates, arrivals, statistics, clock, iteration number, and trace. Preparation
failure marks the engine terminally failed without changing those values; run,
submission, cancellation, and result access are then rejected.

## BatchPlan invariants

`BatchPlan` uses private construction and read-only accessors. Its validated
factories preserve these invariants:

- a request ID occurs only once across prefill, decode, and deferred entries;
- no request is both prefill and decode work;
- scheduled sequences do not exceed `max_num_sequences`;
- scheduled tokens do not exceed `max_batched_tokens`;
- each decode ID contributes exactly one token;
- each prefill contributes the complete prompt count;
- policy input order is preserved;
- all totals are checked component sums.

`BatchPlan::create` rejects zero work. Public `BatchPlan::empty(iteration)` is
the explicit idle form and always contains no scheduled or deferred IDs and
zero totals. Only the engine's planner can construct a nonempty plan containing
deferred work, where candidate costs and remaining budgets are available.
Factories defensively reject invalid enum values.

Deferral reason precedence is deterministic. `TokenBudget` is used whenever a
candidate exceeds the remaining token budget, including when both budgets are
exhausted. `SequenceBudget` is used only when the candidate still fits the
token budget but no per-iteration sequence slot remains.

## Policies

### DecodeFirst

Active decodes are ordered by `(arrival_time_us, request_id)`, followed by
waiting prefills in the same FCFS order. A candidate is selected when both
remaining budgets permit it; otherwise it is recorded as deferred and later
candidates are considered to use available capacity.

This protects ongoing simulated inter-token latency. Long-running decodes can
delay prefills, but a finite workload drains because selected decodes consume
finite output tokens.

### FcfsMixed: budget-aware FCFS traversal

All runnable work is globally ordered by original `arrival_time_us`, then
`request_id`, then work kind. Prefill precedes decode only if invalid internal
state ever gives the same request metadata both kinds; a valid request has one
kind. The same deterministic capacity-filling visit rule then applies. This is
budget-aware FCFS traversal, not strict head-of-line execution: a later fitting
candidate may bypass an earlier candidate that does not fit the remaining
iteration budget.

This provides stronger arrival-order fairness. An older full prefill may
consume the budget and defer a later active decode, worsening simulated
inter-token latency. A large prompt can see head-of-line delay when remaining
capacity is too small, while later fitting work uses that capacity. Neither
policy silently falls back to the other.

## Lifecycle and cancellation

The normal lifecycle is `Waiting -> Prefilling -> Decoding -> Finished`.
Prefill sets admission and first-scheduled time at the iteration start and
completes at its end, at most once. Each selected decode emits exactly one
modeled token. The first decode completion assigns `first_token_time_us` once.
Zero-output work transitions directly from `Prefilling` to `Finished` and
never decodes.

`cancel_request()` is synchronous between iteration boundaries, matching S2's
narrow boundary rather than adding timestamp-priority cancellation events.
Waiting (including pre-arrival) and active decode work can be cancelled.
Prefill is atomic inside a call, so no externally observable in-prefill instant
exists. Cancellation removes the request from future per-iteration scheduling
candidates, produces no later tokens, increments cancellation rather than
completion, and cannot create a stale event. Its validation, counter preflight,
request copy, and arrival-set preparation occur before a no-throw commit, so a
synchronous validation or counter-overflow error leaves the engine unchanged
and usable. Terminal requests never re-enter candidates. `Preempted` is
reserved.

## SIMULATED mixed-batch cost

`SimulatedBackend::estimate_batch_time_us` computes:

```text
batch_time_us = batch_base_us
              + batch_prefill_per_token_us * total_prefill_tokens
              + batch_decode_per_sequence_us * decode_sequence_count
              + batch_active_sequence_overhead_us * total_scheduled_sequences
```

Parameters are non-negative signed 64-bit microseconds. Counts, products, sums,
and the end timestamp are checked. Invalid counts, negative costs, arithmetic
overflow, and timestamp overflow are rejected. This educational formula is not
a GPU, MN-Core, llama.cpp, or kernel timing model.

## Trace and statistics

`plan_trace()` exposes read-only append-only storage. Each **SIMULATED** entry
contains iteration, start/end time, policy, immutable prefill/decode IDs, token
totals, and deferred IDs/reasons.

Statistics are cumulative. Iterations count nonempty and explicit idle plans;
prefill/decode tokens and scheduled sequences sum their plan values. Maximum
batch size and maximum scheduled tokens are plan maxima. Deferred count counts
occurrences, not unique requests. Completion and cancellation are disjoint.
Average batch size is:

`total_scheduled_sequences / nonempty_batch_count`.

It is `std::nullopt` when there are no nonempty batches, never a silent zero.

## SIMULATED educational cost-model comparison

The deterministic comparison test uses aligned S2 and S3 coefficients: a 2 us
base, 1 us per prefill token, and 1 us per decode sequence. For two identical
requests arriving at zero with one prompt token and two output tokens:

- S2 performs two times `(3 us prefill + 2 * 3 us decode) = 18 us`;
- S3 performs one 4 us prefill batch and two 4 us decode batches, totaling
  12 us.

Both paths finish two requests and generate four output tokens. This is a
**SIMULATED educational cost-model comparison**, not hardware speedup and not a
claim that continuous batching universally improves latency or throughput.

## Limitations and Phase S4 handoff

S3 has no threads or true parallel execution, real GPU/MN-Core timing, KV
memory pressure, block allocation, prefix reuse, preemption, HTTP/JSON,
llama.cpp integration, model execution, or vLLM/SGLang feature parity. It uses
whole-prompt prefill; chunking and cancellation inside a batch are deferred.

Phase S4 adds block-based KV management: fixed-size allocation, partial-tail
accounting, deterministic release, capacity pressure, and snapshot information
needed for later recompute preemption. Prefix reuse remains subsequent work.
