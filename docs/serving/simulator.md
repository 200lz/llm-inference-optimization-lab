# Phase S2 serving simulator

## Scope and responsibilities

Phase S2 is a deterministic, single-active-request C++17 discrete-event
simulator. It models arrivals, whole-prompt prefill, one-token decode steps,
completion, and explicit cancellation. Admission is delegated to the formal
`Scheduler` interface; the current policy is FCFS. See [scheduler](scheduler.md)
for its API, ordering, statistics, and responsibility boundary.

Phase S3 does not change this engine. Its separate iteration-level execution
path is documented in [continuous batching](continuous_batching.md).

Public timestamps are signed 64-bit integer microseconds with `_us` suffixes.
`SimulationClock` starts at zero and jumps to event timestamps. It never reads a
wall clock or sleeps. All output is simulated behavior, not measured llama.cpp,
GPU, MN-Core, or other accelerator performance.

The components are:

- `Request`: token counts, lifecycle state, and optional lifecycle timestamps.
- `EventQueue`: stable timestamp/sequence ordering.
- `SimulationClock`: monotonic simulated time.
- `Backend` and `SimulatedBackend`: deterministic cost estimates.
- `Scheduler` and `FcfsScheduler`: metadata-only admission policy.
- `SimulationEngine`: requests, transitions, costs, events, clock, logging,
  cancellation, and terminal failure.

## Request lifecycle

The normal lifecycle is `Waiting -> Prefilling -> Decoding -> Finished`. A zero
output limit legally transitions `Prefilling -> Finished` after prefill and has
no first-token timestamp. Public cancellation adds `Waiting`, `Prefilling`, or
`Decoding -> Cancelled`; cancellation sets `finish_time_us` to the current
simulation clock. `Finished` and `Cancelled` are terminal. `Preempted` remains a
reserved state with no operational support.

Invalid transitions throw `std::logic_error`. Submissions must be pristine:
`Waiting`, zero generated tokens, and no lifecycle timestamps. Backend lifetime
protection is unchanged: the engine borrows a named backend lvalue, and the
rvalue constructor is deleted.

## Events, timestamp boundaries, and stale policy

Event types remain `RequestArrival`, `PrefillComplete`,
`DecodeStepComplete`, and `RequestComplete`. The event queue key remains:

1. `timestamp_us`
2. monotonically assigned `event_sequence`

Equal-timestamp events therefore retain insertion order in the event log. The
engine processes every event at the next timestamp before asking the scheduler
for one admission. This preserves stable event semantics while ensuring all
simultaneous arrivals participate in the scheduler's request-ID tie-break.
`run_next_timestamp()` exposes this deterministic boundary for controlled
stepping; `run()` repeats it to full drain.

`cancel_request()` is called externally between these timestamp boundaries. If
cancellation frees the active slot while an arrival event remains at the
current clock time, admission is deferred until all current-time events have
been processed. Otherwise the next waiter is admitted immediately without
advancing to the cancelled request's obsolete backend event. Because one call
to `run_next_timestamp()` drains the whole timestamp, Phase S2 cannot express a
cancellation event tied exactly against a completion at the same timestamp.
That priority rule belongs to the future event model.

Phase S2 narrows the former rule that every stale event is an error. Each
request has at most one tracked outstanding lifecycle event. Explicit
cancellation authorizes only that exact event sequence and type to be consumed
without a state change. Its one-use authorization is then removed, the
diagnostic `ignored_cancelled_event_count` is incremented, and the ignored event
is excluded from the successful event log. This covers a future arrival after
pre-arrival cancellation and the known backend event after active cancellation.
No other stale event is ignored: a second or mismatched event, duplicate,
unknown ID, corrupt ownership, or invalid transition still fails the engine.

An exception during execution terminally marks the engine failed. The event or
timestamp admission whose processing throws is not appended to the successful
event log. `run()` cannot be retried, and request, event-log, clock, and
scheduler-result accessors are unavailable. `failed()`, processed event count,
queue emptiness, and ignored-cancelled-event count remain diagnostic accessors.

## Simulated cost model

`SimulatedBackendConfig` uses non-negative integer microseconds:

```text
prefill_us = prefill_base_us
           + prefill_per_token_us * prompt_token_count
           + prefill_per_active_sequence_us * active_batch_size

decode_step_us = decode_base_us
               + decode_per_active_sequence_us * active_sequence_count
```

Phase S2 always passes an active count of one. Configuration and event-time
arithmetic is checked before evaluation. Negative durations are rejected and
unrepresentable signed 64-bit results throw `std::overflow_error`.

## Edge cases and limitations

- Arrival timestamps are non-negative; zero is valid.
- Request IDs are unique per engine.
- New submissions cannot precede the current simulated time.
- Zero prompt and output lengths are valid.
- Optional timestamps use no sentinel value.
- The first decode completion alone assigns `first_token_time_us`.
- Successful `run()` consumes meaningful and ignored-cancelled events and fully
  drains the event queue.
- `max_active_requests` is explicit, defaults to one, and must be positive;
  Phase S2 supports exactly one and rejects a larger engine configuration.
- Scheduler decisions carry a checked epoch. Admission commits only the current
  epoch's deterministic FCFS head; stale or non-head commits are rejected before
  mutation.

For this S1/S2 engine there are no threads, wall-clock timing, continuous batching, KV-cache blocks,
prefix caching, preemption, JSON/HTTP, real model execution, llama.cpp adapter,
or hardware measurements. These analytical costs drive a discrete-event
simulation and must not be presented as measured performance.
