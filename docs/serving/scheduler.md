# Phase S2 scheduler

## Scope

Phase S2 replaces the simulator's temporary FIFO admission container with the
`Scheduler` interface and a deterministic `FcfsScheduler`. This is simulated
scheduling behavior. It uses one active request, no threads, and no wall clock.
It does not implement continuous batching, preemption, a KV or prefix cache,
model execution, or accelerator measurement.

## Responsibility boundary

The scheduler owns only immutable scheduling metadata keyed by request ID. It
does not own `Request` objects, expose mutable request pointers, change request
state, estimate backend costs, advance time, or schedule events. It receives
arrival and terminal notifications and returns an explicit `Admit`, `NoWork`,
or `CapacityFull` decision.

`SchedulerDecision` is an invariant-preserving value type: its fields are
private, and factories construct only `Admit(request_id, decision_epoch)`,
`NoWork(decision_epoch)`, or `CapacityFull(decision_epoch)`. Only `Admit`
contains a request ID. Every scheduler state change advances a checked
monotonic epoch. Admission succeeds only when the decision epoch is current and
the decision still names the first FCFS waiting entry; stale decisions and
attempts to admit a non-head request fail before state or statistics change.

`SimulationEngine` owns requests, lifecycle transitions, the simulation clock,
backend cost estimation, event insertion and processing, the event log, and
terminal failure behavior. The engine validates a decision before applying it.

## FCFS and admission capacity

Waiting order is the ordered key:

```text
(arrival_time_us, request_id)
```

Request ID is an explicit tie-break; submission and container insertion order
do not select among simultaneous arrivals. The engine processes all events at a
timestamp before asking for an admission, making the tie-break observable and
repeatable. An earlier long request cannot be bypassed by a later short request.

`max_active_requests` must be positive and defaults to one. `FcfsScheduler`
tracks the configured bound and returns `CapacityFull`; this keeps its API ready
for a future larger limit. The Phase S2 engine supports exactly one and rejects
other configured values. There is no concurrency or continuous batching yet.
`NoWork` means that no request is waiting, including when all configured slots
are occupied. `CapacityFull` means at least one request is waiting but the
configured running limit prevents admission.

FCFS is fair by arrival order and deterministic, but it creates head-of-line
blocking. A long prompt or output ahead of short work can substantially harm
the short requests' simulated time to first token (TTFT).

## Cancellation and stale events

`SimulationEngine::cancel_request` is an external operation performed between
simulation timestamp boundaries and is synchronous at the current simulated
clock. A waiting request is removed from scheduling and transitions to
`Cancelled`. An active prefill or decode request also transitions to
`Cancelled`, releases the single active slot, and allows the next waiter to be
admitted. In every case `finish_time_us` is the cancellation timestamp. A
cancelled request emits no later tokens. Active cancellation counts as both an
admission and a cancellation.

Cancellation before a future arrival is valid: it is recorded as a cancellation
but not as an arrival, and its later arrival event is stale. `run_next_timestamp`
drains every event at one timestamp, so Phase S2 cannot insert a cancellation
event that competes with a completion at that exact timestamp; event-priority
cancellation is a future design target. Unknown IDs and
requests already `Finished` or `Cancelled` are rejected synchronously.
`Preempted` remains reserved and unimplemented.

The stale-event exception is deliberately narrow. The engine tracks the exact
outstanding lifecycle event sequence and type for every submitted request. On
explicit cancellation it authorizes only that known arrival, prefill, decode,
or completion event to be ignored. The authorization is consumed once. The
ignored event increments `ignored_cancelled_event_count` and is not appended to
the successful event log. A second event, a different sequence or type,
duplicate scheduling, invalid transition, unknown request, or other corruption
still fails the engine terminally. Successful runs consume authorized ignored
events and fully drain the event queue.

## Cumulative statistics

`SchedulerStatistics` exposes:

- `arrived_request_count`: successful arrival notifications.
- `admitted_request_count`: successful admissions.
- `completed_request_count`: admitted requests that finish normally.
- `cancelled_request_count`: all explicit cancellations, including before
  arrival and after admission.
- `maximum_waiting_queue_depth`: maximum waiting count immediately after an
  arrival.
- `total_queue_wait_time_us`: sum over admitted requests of
  `admitted_time_us - arrival_time_us`.

Waiting cancellation contributes no queue-wait duration because it is never
admitted. Counts are cumulative; current waiting and running counts are exposed
separately. Signed 64-bit queue-wait accumulation uses checked arithmetic.

## Phase S3 handoff

Phase S3 can extend admission to several active requests and add iteration-level
continuous batching. That work must define how batch capacity, prefill chunks,
and completion boundaries interact; none of those behaviors are approximated
in Phase S2. The output here remains simulation evidence and produces no real
accelerator performance measurement.
