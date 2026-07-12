# Characterizing and Optimizing CPU LLM Inference in llama.cpp

## 1. Executive summary

This project investigates CPU inference performance in a pinned revision of `llama.cpp` using F16, Q8_0, and Q4_K_M variants of Qwen2.5-0.5B-Instruct. Its motivating observation was counterintuitive: smaller quantized model files did not always deliver higher throughput. The work progressed from a durable benchmark harness through phase-specific characterization, Linux performance-counter and sampling analysis, assembly inspection, extracted kernel experiments, and controlled end-to-end A/B validation.

The main verified conclusion is specific to this host and workload. In matched whole `llama-bench` invocations, Q8_0 executed approximately 24.3% more instructions than F16 and Q4_K_M executed approximately 56.7% more. Quantized IPC was higher, not lower: +12.6% for Q8_0 and +15.9% for Q4_K_M relative to F16 in the primary comparison. That improvement in completed instructions per cycle was insufficient to offset the much larger number of instructions. Sampling also showed format-specific hot paths. The dominant Q8_0 path, `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>`, accounted for 56.39% of samples. These findings support additional quantized-kernel work—such as unpacking, conversion, and format-specific arithmetic—as the principal observed correlate of the slower combined invocation. They do not prove that F16 is generally superior.

Two optimization candidates illustrate the value of explicit gates. A Q6_K/Q8_K accumulator rearrangement was bit-correct but regressed representative microbenchmarks, so it was stopped before integration. A second candidate packed two F16 activation-scale conversions in the Q8_0 RN=2 tinyBLAS path. The project-owned extracted tile was bit-identical to baseline and improved by 1.32x at depth 28 and 1.34x at depth 152. Assembly and integration checks passed, controlled binaries had verified provenance, and deterministic `llama-cli` output was identical.

The final 20-pair application experiment did not establish a stable gain. Prompt throughput had a median paired change of -1.571% with a bootstrap 95% confidence interval of [-3.538%, +2.252%]; generation had a median paired change of -1.560%; duration changed +1.170%. All confidence intervals included zero, variability was high, and predefined analysis detected material temporal drift. The correct decision is **inconclusive**: this was neither an end-to-end optimization success nor a correctness failure. No application throughput improvement is claimed.

## 2. Motivation and research questions

Quantization reduces the number of bits used to represent model weights. That can reduce file size and, in favorable workloads, memory traffic. It does not remove the need to unpack blocks, reconstruct scales, prepare quantized activations, perform format-specific dot products, or dispatch specialized kernels. Whether a format is faster depends on the model, phase, shapes, CPU ISA, implementation, and system state—not precision alone.

The project asked four connected questions:

1. How do prefill and autoregressive decode respond to prompt length, generation length, and CPU thread count?
2. How do F16, Q8_0, and Q4_K_M compare when the workload and runtime configuration are matched exactly?
3. If smaller quantized formats are slower in a measured case, do counters and sampled hot functions explain why?
4. Can a selected hot-path change pass correctness, microbenchmark, provenance, and paired end-to-end gates?

Each claim is labeled by its evidentiary role in the detailed reports. A **measured observation** is a value recorded by the experiment. A **hypothesis** is a proposed mechanism that the available evidence does not fully isolate. A **verified conclusion** is supported by explicit checks or comparisons. This vocabulary prevents a throughput correlation from silently becoming a broad architectural claim.

## 3. Environment and reproducibility

The recorded system was an AMD Ryzen 7 5800H with 8 physical cores and 16 logical CPUs, running a Linux 6.18.33.2 WSL2 kernel. Builds used `/usr/bin/gcc` and `/usr/bin/g++` (GCC 15.2.0), CMake, Ninja, and CPU-only GGML. The host exposed AVX2, FMA, and F16C but not AVX-512 or AMX. CPU affinity, host load, frequency, temperature, and Windows power behavior were not globally controlled, an important limitation for small effects.

The dependency is a Git submodule pinned at `e3546c7948e3af463d0b401e6421d5a4c2faf565`. The [dependency record](dependencies.md) and [Release build guide](llama_cpp_build.md) document the pin, compiler selection, CMake configuration, disabled accelerator backends, and executable verification. Model manifests record exact filenames, byte sizes, SHA-256 digests, and the upstream model revision. Generated GGUF files, build trees, JSONL, CSV, plots, and `perf.data` remain untracked.

Reproducibility here means more than preserving a command. The harness stores the resolved command, configuration fingerprint, workload identity, timestamps, outputs, return status, and parsed metrics. Warm-ups precede measured repetitions. Results are appended durably so interrupted matrices can resume. A configuration mismatch blocks accidental resume unless explicitly overridden. Failures, timeouts, and parser errors remain records rather than being dropped. Details appear in [benchmark methodology](benchmarking.md).

## 4. Benchmark infrastructure

`benchmarks/run_llama_bench.py` expands validated YAML configurations into exact cases and invokes the pinned `llama-bench`. It preserves raw JSONL as the audit trail and emits normalized CSV for analysis. Deduplication prevents repeated cases from a focused union of sweeps, while keys include model, context, prompt tokens, generated tokens, threads, batch, and microbatch. Cross-format ratios are produced only after exact-key matching.

The analyzers compute mean, median, sample standard deviation, coefficient of variation (CV), extrema, speedup where an appropriate baseline exists, and experiment-specific comparisons. CV is stored consistently as a ratio in machine-readable analysis and rendered as a percentage in reports. Checkpoint/resume, retries, whole-matrix time limits, and explicit partial-analysis behavior make long experiments inspectable even when incomplete.

Phase 7 adds an event capability probe because a requested `perf` counter is not necessarily usable under WSL2. Supported events were collected; unsupported LLC and backend-stall events were reported as unavailable rather than converted to zero. Phase 8 adds an alternating paired runner with deterministic order randomization, binary hashes, exact affinity, warm-ups, timestamps, peak RSS, and bootstrap analysis. The repository also contains extracted C++17 kernel harnesses so correctness and local timing do not require editing the pinned submodule.

## 5. Prefill and decode characterization

Prefill processes prompt tokens and constructs KV state, exposing parallel work across a token batch. Decode generates one token at a time and has a dependency between successive tokens. This difference suggested—but did not by itself prove—that thread scaling would diverge.

The [Phase 4 report](results/prefill_decode_scaling.md) measured a focused Q4_K_M matrix with one warm-up and five measured repetitions per group. In the exact thread-scaling rows (prompt 128 or 1024, generation 64, threads 1/2/4/8), mean prefill ranged from 48.556 to 194.256 tokens/s and mean generation from 14.214 to 47.894 tokens/s. At prompt 1024, prefill speedup at eight threads was 4.001x over one thread, with 50.0% parallel efficiency; generation speedup was 2.983x, with 37.3% efficiency. At prompt 128, corresponding eight-thread speedups were 3.371x and 3.044x.

The verified conclusion is that the two phases scaled differently in these rows and both showed diminishing efficiency. The measurements do not isolate computation, bandwidth, scheduling, or synchronization as the sole cause. They cover one small model, synthetic shapes, fixed batch settings, and uncontrolled host state.

## 6. Quantization comparison

Phase 5 compared exact F16, Q8_0, and Q4_K_M artifacts from the same Qwen repository revision. Their file sizes were 1,266,425,696, 675,710,816, and 491,400,032 bytes. Relative to F16, Q8_0 was 46.64% smaller and Q4_K_M 61.20% smaller. Those are exact storage comparisons—not claims about resident memory or bytes transferred by a kernel.

Results were phase-dependent. Across the reported matched shapes, F16 had higher prompt-processing throughput than both quantized formats. Decode showed the opposite ordering: Q8_0 and Q4_K_M generated more tokens per second than F16 in those Phase 5 rows. For example, at p512/n64/t4, mean prefill was 285.598, 242.224, and 187.702 tokens/s for F16, Q8_0, and Q4_K_M, while mean generation was 29.188, 47.748, and 55.656 tokens/s. A whole-invocation duration mixes those phases and their fixed setup costs, so it cannot be treated as phase-specific attribution.

The [quantization report](results/quantization_comparison.md) provides all workload rows and a small deterministic output comparison. Different quantizations can produce different generated text; that comparison is descriptive, not a quality benchmark. The observed performance does not establish that one format is universally better.

| Metric | F16 | Q8_0 | Q4_K_M | Interpretation |
| --- | ---: | ---: | ---: | --- |
| GGUF file size | 1,266,425,696 B | 675,710,816 B | 491,400,032 B | Quantization reduced storage by 46.64% and 61.20% |
| p512/n64/t4 prefill mean | 285.598 tok/s | 242.224 tok/s | 187.702 tok/s | F16 was faster for this prefill row |
| p512/n64/t4 generation mean | 29.188 tok/s | 47.748 tok/s | 55.656 tok/s | Quantized formats were faster for this decode row |
| Primary whole-invocation instructions vs F16 | reference | +24.3% | +56.7% | Quantized paths executed substantially more instructions |
| Primary IPC vs F16 | reference | +12.6% | +15.9% | Higher IPC compensated only partly |
| Top sampled format-specific path | F16 tinyBLAS | Q8 tinyBLAS, 56.39% | Q5 tinyBLAS, 49.73% | Concrete hot kernels differed by format |

## 7. KV-cache and context scaling

Phase 6 separated three ideas: actual prompt/context growth, allocated context with a fixed prompt, and decode-length growth. It tested F16, Q8_0, and Q4_0 KV types with exact matching across 36 workload/KV groups and five measured repetitions per group. All 216 invocations completed.

The [KV/context report](results/kv_cache_context_scaling.md) shows that larger allocated context increased measured peak RSS and reduced observed throughput under the tested fixed-prompt configuration. Increasing actual prompt/context or decode length changes the amount of useful work as well as allocation, so those axes require separate interpretation. The benchmark did not report exact KV allocation bytes; the report correctly leaves that field unavailable.

Four quantities must not be conflated. **File size** is the on-disk GGUF artifact. **KV-cache allocation** is runtime storage for keys and values and depends on context, layers, dimensions, and KV type. **RSS** is the operating system's view of resident pages and can include anonymous, shared, and file-backed pages. **mmap/page-cache behavior** determines how model-backed pages become resident and may share accounting with the filesystem cache. Consequently, subtracting model file size from RSS does not produce exact KV bytes, and lower GGUF size does not imply a proportional RSS reduction.

## 8. CPU profiling and root-cause analysis

Phase 7 used matched whole invocations with repeated `perf stat` collection and sampled `perf record` profiles. The environment probe established which counters WSL2 exposed. Cache references/misses, branches/misses, faults, migrations, cycles, and instructions were available; LLC-specific and backend-stall events were not.

The [profiling report](results/cpu_profiling.md) found whole-invocation elapsed ordering F16 < Q8_0 < Q4_K_M for every configured profiling shape. In the primary comparison, Q8_0 executed 24.3% more instructions than F16 and Q4_K_M 56.7% more. Their IPC was higher, rejecting low IPC as the explanation. Higher IPC means the CPU retired more instructions in each cycle, but elapsed work depends approximately on instructions divided by IPC (and effective frequency). A 12.6% or 15.9% IPC improvement cannot erase a 24.3% or 56.7% instruction increase. The instruction-count effect remains larger.

Generic cache-miss rates, branch-miss rates, page faults, and migrations did not follow the slowdown ordering and therefore did not explain it. This is not proof that all memory effects were irrelevant: WSL2 lacked LLC-specific events, and the experiment cannot partition phase-level traffic. It is a narrower verified conclusion that the available generic metrics were not sufficient explanations.

Sampling supplied the missing code-level direction. F16, Q8_0, and Q4_K_M were dominated by different concrete GEMM/GEMV paths. Q8_0's `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>` accounted for 56.39% of primary samples; Q4_K_M's Q5_0 tinyBLAS path accounted for 49.73%, with additional Q6_K/Q8_K dot-product and Q4_K GEMM functions visible. The measured hot functions and instruction counts support the hypothesis that quantized block unpacking, scale/conversion preparation, and specialized arithmetic contribute substantially to the extra instruction work. Source-level counters would be required to apportion that cost precisely.

Thus the verified conclusion is not “F16 is faster.” It is: for this pinned implementation, CPU, model, and matched combined workload, the quantized variants executed substantially more instructions; their higher IPC did not compensate; and format-specific quantized kernels dominated samples.

## 9. Hot-path optimization experiments

The first Phase 8 candidate targeted `ggml_vec_dot_q6_K_q8_K`, which represented 9.58% of the primary Q4_K_M profile. The hypothesis was that two independent integer accumulator chains could reduce dependency latency. The extracted implementation preserved unpacking, products, corrections, conversion, and FMA order and was bit-exact in 2,000 trials. Representative microbenchmarks, however, ranged from 0.960x to 1.002x baseline. The performance gate failed, so integration and end-to-end testing were deliberately not run. The [rejected-candidate report](results/q4_hotpath_optimization.md) turns a negative result into a bounded design lesson: rearranging additions without removing unpack/shuffle work was insufficient.

The second candidate followed the dominant Q8_0 tinyBLAS samples. [Target selection](results/phase8b_target_selection.md) narrowed the work to the RN=2 activation-scale preparation in a 4x2 tile. The baseline handled two half-precision activation scales separately. The candidate packed the pair, used one F16C conversion, and reused the two resulting F32 lanes. The project-owned harness reproduced the relevant arithmetic and observed depth families without modifying the external submodule.

Correctness tests compared scale conversion to a scalar reference, optimized output to baseline bit-for-bit, and both against scalar dequantization at depths 28 and 152. Debug and Release CTests passed. Assembly inspection confirmed the intended packed conversion. The local benchmark, with warm-up, alternating variants, 11 repetitions, and input generation outside timing, measured 226.15 to 171.61 ns/tile at depth 28 (1.32x) and 1150.44 to 860.80 ns/tile at depth 152 (1.34x). These are [extracted-kernel results](results/phase8c_q8_rn2_scale.md), not llama.cpp throughput results.

[Integration provenance](results/phase8c_integration_provenance.md) audited the specialization guard, harness match, arithmetic order, fallback behavior, and compiled assembly. The candidate was exported as a patch rather than committed into `third_party/llama.cpp`. [Binary provenance](results/phase8d_binary_provenance.md) then established clean baseline and single-patch optimized sources at the same commit, matched CMake settings, distinct binary hashes, and bit-identical deterministic `llama-cli` output.

| Candidate | Correctness | Microbenchmark | Integration | End-to-end | Decision |
| --- | --- | --- | --- | --- | --- |
| Q6_K/Q8_K accumulator chains | Bit-exact; scalar tolerance passed | 0.960x–1.002x | Not run after failed gate | Not run | Rejected |
| Q8_0 RN=2 packed scale preparation | Bit-identical to baseline | 1.32x–1.34x extracted tile | Guard, fallback, assembly, provenance passed | 20-pair CIs span zero; material drift | Inconclusive application effect; no speedup claim |

## 10. Microbenchmark versus end-to-end performance

A microbenchmark answers whether the selected code fragment is faster under its harness. End-to-end inference includes model loading and setup, graph scheduling, many other kernels, attention, synchronization, memory behavior, and both prefill and decode. A local win can be diluted by limited hot-path coverage, call shapes outside the optimized specialization, or changes elsewhere in execution.

Amdahl's Law makes the upper bound explicit. If a fraction `f` of total time is improved by local factor `s`, total speedup is `1 / ((1 - f) + f / s)`. Even a 1.34x local gain produces a modest application gain if the exact optimized operation occupies only a small fraction of total time. The 56.39% sampled share refers to the entire Q8 tinyBLAS function family, not solely to RN=2 scale conversion, so applying that share directly to the tile speedup would overstate the prediction.

The focused paired A/B therefore remained necessary. Twenty pairs alternated baseline and optimized variants under deterministic randomized order on four reported physical cores. Baseline and optimized binaries were hashed and checked against their sources. Prompt median paired change was -1.571%, generation -1.560%, and duration +1.170% (positive duration is worse). Prompt's bootstrap 95% CI was [-3.538%, +2.252%], and every metric's interval included zero. Prompt CVs were 8.74% and 9.63%; the first-half versus second-half paired median shifted by 2.153 percentage points, crossing the predefined material-drift threshold.

The [paired A/B report](results/phase8d_q8_end_to_end_ab.md) correctly classifies this as inconclusive. It is not a success because the acceptance gates failed and no stable positive effect was measured. It is not a demonstrated regression because the confidence intervals include zero and temporal noise prevents directional attribution. It is not a correctness failure because output was bit-identical and provenance passed. The candidate remains a valid local optimization whose application impact is unresolved in this environment.

## 11. Lessons learned

First, representation size and execution cost are different quantities. Quantization saves storage but introduces format-specific work; profiling must decide which effect dominates a workload.

Second, prefill and decode should be reported separately. Their parallel structure and observed format ordering differ, and a combined duration can conceal those distinctions.

Third, instruction count and IPC must be interpreted together. “Higher IPC” is not automatically “faster” when the algorithm or kernel executes many more instructions.

Fourth, optimization needs staged gates. Correctness alone did not rescue the first candidate, and microbenchmark speed alone did not validate the second end to end. Stopping at the appropriate gate saved work and prevented an unsupported claim.

Fifth, provenance is part of performance correctness. A/B results are meaningful only when source revisions, patches, configurations, binaries, commands, and workload identities are known.

Finally, uncertainty is a result. Preserving negative and inconclusive experiments exposes where the evidence ends and makes the next experiment better targeted.

## 12. Limitations

The measurements cover one AMD Ryzen 7 5800H, WSL2, one small Qwen2.5 model family, CPU-only execution, one pinned `llama.cpp` revision, and limited synthetic shapes. CPU frequency, temperature, Windows power management, page-cache state, and competing host load were uncontrolled. WSL2 virtualized the PMU and did not expose all requested events, including LLC-specific counts and backend stalls. Sampling used a limited number of profiles and some runtime symbols were unresolved.

Peak RSS cannot be interpreted as exact model or KV-cache memory. The deterministic output suite is not a model-quality evaluation. The extracted kernel harness establishes local arithmetic and timing behavior but is not identical to all integrated call paths. The end-to-end A/B showed substantial variability and temporal drift. No CUDA or GPU validation was performed, and the local optimization did not demonstrate a stable end-to-end gain.

## 13. Future work

The next CPU experiment should use native Linux with stronger frequency, thermal, affinity, and background-load control, then repeat the paired A/B with a preregistered stopping rule. Larger models and more workload shapes would test whether instruction overhead and memory savings shift with working-set size. `llama-server` measurements should add time to first token, time per output token, throughput under concurrency, and prefix/KV-cache reuse.

GPU work should add CUDA/NVIDIA profiling and distinguish kernel, launch, transfer, and serving effects. If a stable reproducible application effect emerges, the profiling package, focused patch, correctness evidence, and A/B result can support an upstream issue or pull request.

## 14. Conclusion

This study followed an end-to-end performance-engineering chain: reproducible measurement, phase separation, cross-format comparison, resource scaling, hardware-counter collection, sampled hot-path attribution, assembly-level experimentation, correctness gates, binary provenance, and paired application validation.

On this CPU and workload, smaller Q8_0 and Q4_K_M files did not guarantee faster combined inference. They executed 24.3% and 56.7% more instructions than F16 in the primary matched profile, and higher IPC did not fully compensate. A dominant Q8_0 path motivated an AVX2/F16C scale-preparation experiment that was bit-identical and 1.32x–1.34x faster locally. The application experiment remained noisy and statistically inconclusive, so no end-to-end speedup is claimed.

The outcome demonstrates the core discipline of systems optimization: measure the right boundary, preserve reproducibility and correctness, and let the evidence—not the attractiveness of the optimization—determine the claim.
