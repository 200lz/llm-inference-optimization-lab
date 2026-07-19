# Serving simulator development plan

## Delivery principles

Implementation follows the repository's C++17, GCC/Ninja/CMake-preset conventions and keeps `third_party/llama.cpp` pinned and unmodified. Each phase should add the smallest usable behavior, deterministic native tests, and documentation before adding policy complexity. Generated traces and results remain untracked; any published measurements must include the existing benchmark provenance requirements.

## Phases and acceptance gates

### S0 — design (this phase)

- Fix component boundaries, request states, metric definitions, result provenance, directory structure, and namespaces.
- Record ambiguities as open questions rather than selecting hidden assumptions in implementation.
- Gate: documentation is internally linked, whitespace-clean, and existing CTest/pytest suites pass.

### S1 — deterministic simulation core

- Add request/event types, integer simulated time, stable event ordering, workload CSV replay, lifecycle validation, and an append-only event log.
- Add a minimal simulated backend with constant/configured prefill and decode costs.
- Add temporary single-active FIFO admission without preemption or prefix reuse.
- Gate: golden event traces are byte-for-byte repeatable; tests cover simultaneous cancellation/completion/arrival, empty workloads, invalid transitions, and output limits.

### S2 — scheduler abstraction and deterministic FCFS

- Replace temporary FIFO admission with a metadata-only `Scheduler` interface.
- Add deterministic FCFS ordering by `(arrival_time_us, request_id)`, explicit
  admission decisions, positive capacity configuration, cancellation, a narrow
  cancelled-event stale policy, and cumulative scheduler statistics.
- Validate admission against a monotonic scheduler decision epoch and the
  current FCFS head. Authorize cancellation staleness by exact outstanding
  event sequence and type rather than by terminal request state.
- Keep one active request and preserve the S1 event, arithmetic, backend
  lifetime, pristine submission, and terminal failure invariants.
- Gate: tests cover scheduler state validation, FCFS head-of-line behavior,
  simultaneous arrivals, waiting/active/pre-arrival cancellation, checked
  statistics, stale-event auditing, and exact simulated timestamps.

### S3 — deterministic continuous batching

- Preserve the single-active S2 engine and add a separate iteration-level engine
  with multiple active decodes, strict sequence/token budgets, immutable plans,
  `DecodeFirst`, `FcfsMixed`, full-prompt prefill, plan tracing, and batch
  statistics. Iteration publication is a two-phase transaction, and
  synchronous cancellation provides a strong exception guarantee.
- Reject oversized prompts because chunked prefill is not implemented. Keep KV
  allocation, prefix reuse, preemption, and true concurrency out of scope.
- Gate: exact tests cover policy ordering, arrivals during decode, cancellation,
  lifecycle, budgets, arithmetic, trace determinism, and a labeled simulated
  FCFS-versus-continuous behavioral comparison.

### S4 — block KV cache

- Add metadata-only fixed-size block allocation, partial-tail accounting,
  deterministic lowest-ID release/reuse, capacity invariants, cache-aware
  planning snapshots, `KVCapacity` deferral, and KV diagnostics. Keep the S4
  reference non-preemptive; blocked work waits for ordinary completion or
  cancellation to release capacity. A fully KV-deferred boundary returns an
  explicit nonterminal stalled result, and same-plan completions cannot donate
  blocks until the following iteration.
- Gate: tests cover exhaustion, fragmentation, prompt/decode growth,
  finish/cancel cleanup, fixed-seed invariant sequences, transactional failure,
  deterministic IDs, planner reservation, and capacity conservation.

### S5 — prefix cache

- Add exact, longest block-aligned prefix lookup; immutable shared blocks; reference counting; compatibility keys; and deterministic LRU eviction.
- Gate: tests cover full/partial/missed prefixes, incompatible model/tokenizer identities, collision checks, eviction ties, shared-block lifetime, and cache disabled equivalence.

### S6 — metrics and workload experiments

- Emit request-, token-, batch-, cache-, and run-level records using the definitions in [metrics](metrics.md).
- Add seeded synthetic workload generation and versioned replay examples under `benchmarks/serving/workloads/`.
- Compare FCFS and continuous batching across burstiness, prompt/output shape, and KV capacity. Label every result `simulated`.
- Gate: hand-calculated fixtures validate every metric; conservation checks reconcile arrived requests, terminal requests, token counts, and KV block-time.

**Status: implemented.** S6 provides `serving-workload-v1` JSONL replay,
deterministic generators and manifests, a strict Python/TSV/C++ boundary,
request/iteration/run records, service-level metrics, bounded matrices,
provenance, reference SIMULATED summaries, CI smoke coverage, and the
[final serving report](final_report.md). Exact tests cover nullable metric edge
cases and conservation; the simulator still does not claim physical KV
block-time or hardware calibration.

### S7 — calibration and sensitivity analysis

- Fit or select simulated backend parameters from existing or newly collected llama.cpp CPU measurements while preserving source provenance.
- Report sensitivity bands instead of presenting a single calibration as universal. Keep calibration, analytical estimates, and simulation outputs separately labeled.
- Gate: held-out workload shapes quantify model error; configuration fingerprints identify the measurement set and fit method.

### S8 — optional llama.cpp backend adapter

- Add an opt-in project-owned adapter and capability discovery without modifying or updating the submodule.
- Start with supported batch/prefill/decode operations and monotonic timing; do not imitate unsupported preemption or prefix operations.
- Gate: adapter correctness tests, clean submodule verification, recorded model/build/hardware provenance, and side-by-side—but never merged—simulated versus measured results.

## Test strategy

- Unit tests: state transition table, scheduler decisions from immutable snapshots, block/reference invariants, prefix compatibility, event ordering, and metric formulas.
- Scenario tests: small traces with hand-written timelines and expected events; overload, cancellation, preemption, and cache reuse.
- Property/invariant tests without new libraries: seeded loops checking determinism, no time reversal, no double allocation/free, terminal-state uniqueness, and accounting conservation.
- Integration tests: invoke `serving_sim` on checked-in tiny traces and compare normalized output. Avoid wall-clock assertions.
- Future adapter tests: separate from simulator tests and skipped cleanly when llama.cpp/model prerequisites are absent.

Every behavioral phase runs:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
.venv/bin/python -m pytest -q
git diff --check
```

CI should add the new native tests to the existing GCC and Clang jobs. No Python package or third-party C++ dependency is needed for the core simulator.

## Reproducible experiment contract

Each run records simulator version, evidence kind, configuration and workload fingerprints, scheduler and backend names/versions, tick unit and rounding, random algorithm/seed, cache parameters, metric window, host metadata, and—when applicable—llama.cpp commit/model/build details. Output must include raw event/request records plus a derived summary so analyses can be regenerated.

A result is publishable only when its workload is non-fabricated or explicitly labeled synthetic, its methodology is documented, and its evidence kind is visible in tables and charts. Existing benchmark result files are never rewritten by serving experiments.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Cost model appears more accurate than it is | Require evidence labels, calibration provenance, held-out error, and sensitivity analysis |
| Policy depends accidentally on container/thread ordering | Stable IDs, ordered iteration, integer time, explicit event priority, repeatability tests |
| Cache model hides physical costs | Report logical block accounting and model lookup/eviction/recompute costs explicitly |
| Continuous batching starves long prompts | Measure waiting age; add an aging/admission policy only after the reference policy is characterized |
| Adapter constraints leak into core design | Capability interface and separate optional CMake target |
| Scope grows into a production server | Keep networking, distributed serving, and GPU execution as non-goals |

## Questions to resolve before the relevant phase

1. S1: choose the public trace time unit and whether prompt content is inline token IDs or referenced from a companion file.
2. S4: resolved with a 16-token API default and explicit test values;
   preemption is deferred beyond S4.
3. S5: implemented deterministic full-block prefix caching with a documented
   FNV-1a encoding and mandatory exact collision verification. Production
   hash/security tradeoffs remain outside this simulator.
4. S7: choose the cost-model family only after inspecting the available llama.cpp measurements; do not assume linear scaling.
5. S8: decide which pinned llama.cpp APIs are sufficiently stable for an adapter and whether adapter work belongs in the default build.

## Phase S5 completion

S5 extends the S4 metadata pool with exact-token requests, parent-dependent
keys, longest consecutive prefix lookup, shared references, cached
zero-reference blocks, deterministic LRU eviction, cache-aware prefill budgets,
two-phase engine integration, diagnostics, native tests, and a SIMULATED smoke
scenario. Same-plan insertions become visible only at the next iteration.
Real tensors, partial-block/decode sharing, radix trees, chunked prefill,
preemption, swapping, distribution, and hardware execution remain deferred.
