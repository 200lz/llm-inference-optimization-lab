# Mini LLM serving engine: final report

## 1. Executive summary

Phase S6 completes a deterministic, metadata-only LLM serving simulator. The project replays versioned workloads through single-active FCFS or continuous batching, models fixed-block KV capacity and exact full-block prefix reuse, and derives TTFT, TPOT, end-to-end latency, throughput, tail percentiles, and SLO goodput. Every value in this report is **SIMULATED** unless a paragraph explicitly refers to the earlier real llama.cpp CPU study.

The representative matrix is intentionally small. Under its educational linear cost model, prefix reuse reduced scheduled prefill work on prefix-local traces, an eight-block KV pool stalled after one completion, and overload grew FCFS tail latency. Several comparisons were neutral: DecodeFirst and FcfsMixed produced identical aggregates on the selected medium trace, medium and large KV pools were indistinguishable, and mixed-traffic prefix caching did not change reported throughput. Those are findings, not gaps to hide.

## 2. Why this project was built

The earlier repository work measured and profiled real CPU execution in a pinned llama.cpp revision. This serving project addresses a different layer: how request arrival, admission, batch policy, KV residency, and prefix locality interact. Simulation makes policy timelines deterministic and testable without presenting synthetic costs as hardware performance.

## 3. Architecture

```text
versioned JSONL workload -> Python validation -> strict temporary TSV
  -> dependency-free C++ runner -> native JSONL records
  -> Python provenance + metrics -> JSON / CSV / Markdown
```

Python owns JSON schemas, seeded workload generation, provenance, experiment orchestration, and analysis. The C++17 runner converts validated rows to existing `Request` objects and selects either `SimulationEngine` or `ContinuousBatchingEngine`. It never parses human smoke output. The core remains independent of Python and external JSON libraries.

Native output uses the closed `serving-simulator-v2` schema. Required and
nullable fields are explicit for provenance, request, iteration, and summary
records; missing or unknown fields are rejected. Before metrics are derived,
Python reconciles request lifecycle/timing, prompt and decode work, continuous
iteration totals and request appearances, the single terminal summary, and the
original workload envelope. FCFS intentionally emits no iteration records and
is conserved directly from request records. The compact schema contract is in
[the workload and result schema](workload_schema.md#closed-result-envelope).

## 4. Request lifecycle

Requests begin `Waiting`, are admitted into `Prefilling`, optionally enter `Decoding`, and end `Finished`. Cancellation remains explicit; preemption is reserved but not implemented. Arrival ties use the exact external request ID in lexicographic UTF-8/string order (`"10"` before `"2"`), independently of JSONL line order. Integer microseconds, checked arithmetic, immutable request prompt representation, and stable event/iteration ordering make replay byte-stable below the provenance row.

## 5. FCFS and continuous batching

Single-active FCFS admits one request at a time in `(arrival_time_us, request_id)` order. Continuous batching plans full prefills and one-token decode steps under independent sequence and token budgets. `DecodeFirst` orders active decode before new prefills; `FcfsMixed` orders both by arrival and ID. Neither policy preempts, swaps, or chunks a prompt.

## 6. KV-cache block management

The KV manager represents fixed-size physical blocks, ordered free IDs, request block tables, valid-token occupancy, and fragmentation. A sequence budget limits one iteration; KV capacity limits all resident sequences across iterations. Reservation and commit use exact physical IDs. A fully KV-deferred plan records a nonterminal stall without advancing simulated time.

## 7. Prefix caching

Exact-token requests can reuse complete prompt blocks. Parent-dependent stable hashes locate buckets; exact token, namespace, and salt material verifies collisions. Reference counts equal request-table membership. Zero-reference cache blocks remain resident and deterministic LRU evicts by `(last_access_epoch, physical_block_id)`. Count-only requests and partial prompt tails never enter the prefix index.

## 8. Workload model

Schema `serving-workload-v1` supports count-only and exact-token prompts; see the [canonical schema](workload_schema.md). Checked-in deterministic classes are chat, shared system prompt, coding agent, mixed, overload, and burst. Arrival modes are fixed interval, equal-time burst, manual trace, and seeded exponential. Exponential gaps use `floor(-mean_interval_us * log1p(-U))` with `U` from `random.Random(seed)`; accumulated integer timestamps are non-decreasing. Sidecar manifests contain the generator config, seed, schema, count, class, and workload SHA-256, with a null timestamp so reproducibility does not depend on wall time.

## 9. Metrics

For arrival `a`, admission `s`, first output `t`, finish `f`, and generated count `n`:

- queue delay = `s - a`;
- TTFT = `t - a`, null when `n == 0`;
- end-to-end latency = `f - a`;
- TPOT = `(f - t) / (n - 1)`, null when `n < 2`.

Native records include decode completion timestamps, so exact modeled inter-token gaps can be reconstructed. TPOT remains the primary reported cadence metric. Rates divide completed work by the first-arrival-to-last-finish window; a nonpositive window is null. Percentiles use nearest rank: sort, take one-based `ceil(p*N)`, clamp to `[1,N]`; an empty population is null. Duplicates remain duplicates and one observation returns itself.

Goodput counts a finished request only when every configured applicable TTFT, TPOT, and E2E threshold passes. Undefined metrics do not fail an inapplicable constraint. Good requests per simulated second and the good/finished ratio are reported with violation counts. Thresholds in checked-in configs are explicitly educational simulated thresholds, not production recommendations.

## 10. Experiment methodology

The default matrix has eight bounded runs; the extended matrix has 24 and is not used by CI. Its chat scheduler group includes single-active FCFS, continuous DecodeFirst, and continuous FcfsMixed. Comparisons normalize all settings except the named dimension. Outputs record the git revision and branch, dirty status, pinned llama.cpp revision, normalized config/hash, workload manifest/hash, seed, Python version, simulator schema, build type, and runner hash. Portable commands use repository-relative paths and symbolic temporary TSV paths. Dirty runs are allowed and marked; the checked-in reference is not a release result.

## 11. Scheduler comparison

| Configuration | Evidence | Completed | Request/s | P99 TTFT (us) | P99 E2E (us) |
| --- | --- | ---: | ---: | ---: | ---: |
| single-active FCFS, chat | SIMULATED | 8 | 68.94 | 946 | 3,745 |
| continuous DecodeFirst, chat | SIMULATED | 8 | 67.97 | 521 | 6,230 |
| single-active FCFS, mixed | SIMULATED | 12 | 104.24 | 1,765 | 5,117 |
| continuous DecodeFirst, mixed | SIMULATED | 12 | 101.08 | 1,181 | 8,721 |
| continuous FcfsMixed, mixed | SIMULATED | 12 | 101.08 | 1,181 | 8,721 |

Continuous batching lowered modeled P99 chat TTFT but slightly lowered completed-request throughput and increased P99 E2E under these coefficients. DecodeFirst and FcfsMixed were aggregate-neutral on this trace; a different arrival overlap or decode length distribution is needed to separate them. The simulator therefore does not assert that continuous batching always wins.

## 12. Prefix-cache comparison

| Workload | Cache | Evidence | Prefix token hit rate | P99 TTFT (us) | Request/s |
| --- | --- | --- | ---: | ---: | ---: |
| shared system prompt | off | SIMULATED | N/A | 556 | 110.21 |
| shared system prompt | on | SIMULATED | 0.88 | 536 | 110.69 |
| coding agent | off | SIMULATED | N/A | 971 | 75.84 |
| coding agent | on | SIMULATED | 0.667 | 941 | 76.31 |
| mixed | off/on | SIMULATED | N/A / 0.571 | 1,181 / 1,181 | 101.08 / 101.08 |

Prefix-local workloads saved simulated prefill tokens and modestly improved modeled latency/rate. Mixed traffic registered reuse but no aggregate throughput change. Count-only/random traffic remains ineligible rather than being mislabeled a cache miss.

## 13. KV-capacity comparison

The 8-block run completed 1 of 12 requests and then recorded one KV-capacity deferral plus one stalled iteration. The 64- and 192-block runs completed all 12 with no deferral and identical aggregate latency. This establishes a capacity threshold for the trace, not a monotonic benefit above it.

## 14. Block-size tradeoff

Eight-token blocks reported a 0.78 token hit rate versus 0.88 for 16 and 32 because full-block eligibility changes with granularity. All three completed at the same modeled rate and latency. Smaller blocks can reduce tail fragmentation while consuming more block metadata; larger blocks need fewer IDs but may waste more token slots. This simulator reports block counts and token-slot fragmentation, not metadata bytes or real memory traffic.

## 15. Load and overload behavior

The controlled mixed sweep changes only arrival timestamps. Low, medium, and overload input produced about 35.43, 101.08, and 458.31 completed requests per simulated second. P99 TTFT grew from 1,181 us at low/medium load to 11,123 us under overload; P99 E2E grew to 20,683 us. The full-drain denominator explains why completed throughput can rise with arrival rate even as tails worsen. A separate single-active overload trace reached P99 TTFT 31,264 us and a 0.667 goodput ratio under the educational SLOs.

## 16. Findings

Scheduling, capacity, and locality are coupled. Batching changes who progresses together; KV capacity can stop admission independently of token budget; prefix reuse helps only when exact full-block locality exists. Tail latency and goodput expose overload that average throughput alone can obscure.

## 17. Negative and inconclusive results

- Continuous batching did not improve modeled request throughput in the small chat/mixed comparisons.
- DecodeFirst and FcfsMixed were indistinguishable in aggregate on the medium mixed trace.
- Medium and large KV pools were neutral once both cleared the working-set threshold.
- Mixed prefix caching produced a hit rate but no aggregate latency/rate gain.
- Block sizes 16 and 32 were indistinguishable in the chosen shared-prefix run.

These outcomes are kept because “optimization” is workload- and model-dependent.

## 18. Simulator versus real serving engines

This project is conceptually inspired by serving-engine scheduling, block KV allocation, and prefix reuse. It is not source-compatible with llama.cpp, vLLM, or SGLang, not performance-equivalent, and not feature parity. See [the engine comparison](engine_comparison.md). The earlier llama.cpp CPU tables are real measurements; S6 tables are synthetic event-model outputs and are never pooled with them.

## 19. Relevance to accelerator-oriented LLM serving

An accelerator serving stack coordinates compiler-generated execution programs, runtime admission, memory residency, and request policy. Prefill exposes different shapes and parallelism from one-step autoregressive decode. KV residency consumes finite DRAM capacity, while repeated exact prefixes can avoid redundant prefill work if lookup, execution, and reference lifecycles agree.

The preferred throughput/TTFT balance, batch budget, and KV block size depend on target kernels, program launch costs, DRAM capacity/bandwidth, and compiler/runtime constraints. An MN-Core-like target would require public, target-specific calibration; no proprietary architecture detail is assumed here. Simulator findings are hypotheses until validated on target hardware with measured program times and memory behavior.

## 20. Limitations

There are no tensors, tokenizer, model kernels, real GPU or MN-Core execution, calibrated costs, byte-level KV layout, HTTP/API layer, networking, threads, distribution, preemption, swapping, chunked prefill, partial-block reuse, model downloads, or llama.cpp adapter. Stalled runs require external cancellation or a larger pool; they do not autonomously recover. The cost model is linear educational input, not a hardware claim.

## 21. Reproduction commands

```bash
cmake --preset debug
cmake --build --preset debug -j
.venv/bin/python benchmarks/run_serving_matrix.py configs/serving/matrix_small.json --force
.venv/bin/python benchmarks/analyze_serving_results.py \
  --command-manifest .artifacts/serving/matrix_small/command_manifest.json \
  --json /tmp/serving-summary.json --markdown /tmp/serving-report.md
.venv/bin/python benchmarks/plot_serving_results.py /tmp/serving-summary.json \
  --output-dir /tmp/serving-plots
scripts/run_serving_demo.sh
scripts/verify_serving_project.sh
```

The default matrix writes only below ignored `.artifacts/serving/` and leaves tracked references untouched. Use `matrix_extended.json --max-runs 24` for the non-CI matrix. Only `run_serving_demo.sh --update-reference` or `run_serving_matrix.py --update-reference --force` may update approved paths below `results/serving/`.

## 22. Future work

Calibrate cost coefficients on a declared target, add sensitivity/error bands, measure queue depth directly in native traces, and compare policy behavior on longer steady-state traces. Real backend adapters, preemption, chunked prefill, and distributed serving remain separate future phases rather than implicit S6 claims.
