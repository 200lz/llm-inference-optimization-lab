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

### S3 — block KV cache and continuous batching

- Add fixed-size block allocation, partial-tail accounting, deterministic release, capacity invariants, prefill chunking, iteration-level admission, and recompute preemption.
- Add continuous batching with documented stable tie-breaking.
- Gate: tests cover exhaustion, fragmentation, finish/cancel cleanup, deterministic victim selection, starvation scenarios, and FCFS-versus-continuous-batching behavioral differences.

### S4 — prefix cache

- Add exact, longest block-aligned prefix lookup; immutable shared blocks; reference counting; compatibility keys; and deterministic LRU eviction.
- Gate: tests cover full/partial/missed prefixes, incompatible model/tokenizer identities, collision checks, eviction ties, shared-block lifetime, and cache disabled equivalence.

### S5 — metrics and workload experiments

- Emit request-, token-, batch-, cache-, and run-level records using the definitions in [metrics](metrics.md).
- Add seeded synthetic workload generation and versioned replay examples under `benchmarks/serving/workloads/`.
- Compare FCFS and continuous batching across burstiness, prompt/output shape, and KV capacity. Label every result `simulated`.
- Gate: hand-calculated fixtures validate every metric; conservation checks reconcile arrived requests, terminal requests, token counts, and KV block-time.

### S6 — calibration and sensitivity analysis

- Fit or select simulated backend parameters from existing or newly collected llama.cpp CPU measurements while preserving source provenance.
- Report sensitivity bands instead of presenting a single calibration as universal. Keep calibration, analytical estimates, and simulation outputs separately labeled.
- Gate: held-out workload shapes quantify model error; configuration fingerprints identify the measurement set and fit method.

### S7 — optional llama.cpp backend adapter

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
2. S3: choose default block size, token budget, and whether prefill chunks and decode tokens share one budget with equal weight.
3. S4: choose the prefix key/hash after measuring the memory-versus-collision-check tradeoff.
4. S6: choose the cost-model family only after inspecting the available llama.cpp measurements; do not assume linear scaling.
5. S7: decide which pinned llama.cpp APIs are sufficiently stable for an adapter and whether adapter work belongs in the default build.
