# Phase S1 serving simulator

## Scope and responsibilities

Phase S1 is a deterministic, single-active-request discrete-event simulator. It
models request arrival, whole-prompt prefill, one-token decode steps, and request
completion. It accepts several requests and uses an internal FIFO of arrived
requests while allowing only one request to execute at a time. Phase S2 will
replace this internal selection with a `Scheduler`; S1 intentionally has no
scheduler interface or FCFS policy class.

Public simulation timestamps are signed 64-bit integer microseconds and use the
`_us` suffix. `SimulationClock` starts at zero, advances directly to event
timestamps, and rejects backward movement. It never reads a wall clock or
sleeps. These timestamps are modeled time, not measured llama.cpp, GPU, MN-Core,
or other accelerator performance.

The main components are:

- `Request`: token counts, lifecycle state, and optional lifecycle timestamps.
- `EventQueue`: stable timestamp/sequence ordering.
- `SimulationClock`: monotonic simulated time.
- `Backend`: deterministic prefill and decode-step cost estimates.
- `SimulatedBackend`: checked integer linear cost model.
- `SimulationEngine`: event processing, the S1 FIFO, and single-request
  execution.

Phase S1 stores only `prompt_token_count`; it does not store prompt token IDs.

## Request lifecycle

The normal positive-output lifecycle is:

`Waiting -> Prefilling -> Decoding -> Finished`

An output limit of zero completes immediately after prefill through the special
legal transition `Prefilling -> Finished`. Its first-token timestamp remains
unset. The full legal transition table is:

| From | Legal destinations |
| --- | --- |
| `Waiting` | `Prefilling`, `Cancelled` |
| `Prefilling` | `Decoding`, `Finished` for zero output, `Preempted`, `Cancelled` |
| `Decoding` | `Finished`, `Preempted`, `Cancelled` |
| `Preempted` | `Waiting`, `Cancelled` |
| `Finished` | none |
| `Cancelled` | none |

All other transitions are illegal. `Request::transition_to` throws
`std::logic_error` for an illegal transition. `Preempted` and `Cancelled` are
reserved lifecycle states in Phase S1: `SimulationEngine` does not initiate
these transitions, and their operational semantics will be added in later
phases. Their presence in the enum and transition table must not be interpreted
as implemented scheduling, cancellation, or preemption support.

## Events and ordering

The event types are `RequestArrival`, `PrefillComplete`,
`DecodeStepComplete`, and `RequestComplete`. Each event contains
`timestamp_us`, a monotonically assigned `event_sequence`, `request_id`, and
its type.

The queue key is exactly:

1. `timestamp_us`
2. `event_sequence`

Sequence is assigned on insertion, so equal-timestamp events execute in
insertion order. Event type, request ID, addresses, and container iteration
never break ties. This S1 rule intentionally narrows the future priority-based
semantics discussed in `architecture.md`; cancellation events are not present
in S1.

The engine records processed events in an append-only in-memory event log.
Contradictory completion events are treated as duplicate or stale and rejected
with `std::logic_error`. The public engine does not expose event injection, so a
normal run cannot create such an event.

`SimulationEngine` borrows its `Backend` from a named lvalue. The backend must
outlive the engine; construction from an rvalue backend is rejected to prevent
a dangling reference.

An exception during `run()` is terminal and deterministically marks the engine
as failed. The event whose handling throws is not appended to the successful
event log, and `run()` cannot be called again. Request records, the event log,
and the simulation clock are unavailable after failure because they may contain
partial state; their accessors throw `std::logic_error`. `failed()`,
`processed_event_count()`, and `event_queue_empty()` remain available for
failure diagnostics. Phase S1 does not attempt rollback or retry.

## Simulated cost model

`SimulatedBackendConfig` uses non-negative integer microseconds:

```text
prefill_us = prefill_base_us
           + prefill_per_token_us * prompt_token_count
           + prefill_per_active_sequence_us * active_batch_size

decode_step_us = decode_base_us
               + decode_per_active_sequence_us * active_sequence_count
```

The S1 engine supplies an active batch/sequence count of one because it executes
only one request. The backend API still accepts the counts explicitly to keep
the estimate inputs visible. A count of zero is invalid. Negative configuration
values fail during `SimulatedBackend` construction, and the engine calls backend
validation again before accepting a run.

Every addition and multiplication in cost and event-time calculation is checked
before evaluation. Unrepresentable counts or results throw
`std::overflow_error`; signed overflow is never used as behavior. A backend that
returns a negative duration is rejected with `std::logic_error`.

## Edge-case policies

- `arrival_time_us` must be non-negative; timestamp zero is valid.
- Request IDs are explicit `RequestId` values and must be unique per engine.
- A request submitted after time has advanced cannot have an arrival earlier
  than the current simulated time.
- Zero prompt tokens are valid and still incur configured base and active-count
  prefill costs.
- Zero output tokens are valid, finish after prefill, generate no decode event,
  and leave `first_token_time_us` unset.
- Optional lifecycle timestamps represent events that have not occurred; zero
  is never a missing-value sentinel.
- The first decode completion assigns `first_token_time_us`; later steps do not
  change it.
- Arrived requests enter FIFO order. Equal-time arrivals use submission/event
  insertion order.
- A request is never started before arrival. Finished and cancelled requests
  cannot transition back to active states, and completion removes the active
  request before another is selected.
- A successful `run()` drains the event queue. Invalid or overflowing runs throw,
  enter the terminal failed state, and make normal simulation results
  unavailable.

## Current limitations

S1 has no threads, wall-clock timing, real model execution, workload parser,
cancellation API, scheduler abstraction, scheduling budgets, continuous
batching, KV-cache blocks, prefix caching, JSON/HTTP support, or llama.cpp
integration. Costs are transparent analytical inputs used by a discrete-event
run and must not be interpreted as hardware measurements.
