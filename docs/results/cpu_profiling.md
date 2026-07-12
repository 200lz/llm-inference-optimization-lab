# CPU profiling and bottleneck attribution

## Status

- **Verified conclusion:** Phase 7 is **completed**: 15 warm-ups and 45/45 measured perf-stat profiles succeeded; 3/3 perf-record and 3/3 perf-report/annotate operations succeeded; none failed or timed out.
- **Verified conclusion:** perf 7.0.6 is installed. Sixteen events were supported; `stalled-cycles-backend`, `LLC-loads`, and `LLC-load-misses` were unsupported. No probe event was permission denied.
- **Measured observation:** Hardware events were multiplexed (typically about 66% running time for some events). Perf scaled counts, but WSL2 PMU virtualization and multiplex scaling limit absolute precision.
- **Verified conclusion:** FlameGraph scripts (`stackcollapse-perf.pl`, `flamegraph.pl`) were unavailable. No scripts were downloaded; symbolized perf reports were preserved instead.

## Environment

The host is an AMD Ryzen 7 5800H (8 physical cores, 16 logical CPUs), 7.5 GiB RAM, running WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`. `perf_event_paranoid=2`, `kptr_restrict=1`, user `chen1`, and affinity CPUs 0-15 were unchanged. CPU governor files were not exposed. Full flags and commands are in [profiling_environment.md](../profiling_environment.md).

The verified CPU Release build is pinned at `e3546c7948e3af463d0b401e6421d5a4c2faf565`: `Release`, `-O3 -DNDEBUG`, `GGML_CPU=ON`, `GGML_NATIVE=ON`, and `GGML_CPU_REPACK=ON`. The ELF is unstripped and has `.symtab`, but no DWARF debug sections; frame pointers were not explicitly enabled. DWARF call graphs at 99 Hz nevertheless resolved GGML shared-library functions, so no separate profiling build was created or substituted. Model source revision is `9217f5db79a29953eb74d5343926648285ec7e67` for all three artifacts.

## Workloads

Every target used `-b 512 -ub 512 -mmp 1 -ngl 0 -dev none -r 1 -o json` and context 4096. The exact model, prompt, generation, thread, and depth arguments are persisted with every record.

| Workload | Prompt | Generation | Threads | Stat repetitions/format |
| --- | ---: | ---: | ---: | ---: |
| primary | 1024 | 64 | 4 | 3 |
| short prefill | 128 | 64 | 4 | 3 |
| long prefill | 2048 | 64 | 4 | 3 |
| thread checkpoint | 1024 | 64 | 1 | 3 |
| thread checkpoint | 1024 | 64 | 8 | 3 |

One runner warm-up preceded each model/workload group. One 99 Hz DWARF-call-graph record was collected per format for the primary workload. Three stat repetitions and one record repetition were chosen because profiling overhead makes the five ordinary benchmark repetitions unnecessarily expensive.

Example primary F16 target command:

```text
third_party/llama.cpp/build-release/bin/llama-bench -m models/qwen2.5-0.5b-instruct-fp16.gguf -t 4 -p 1024 -n 64 -b 512 -ub 512 -d 3008 -r 1 -o json -mmp 1 -ngl 0 -dev none
```

## Throughput reference

- **Measured observation:** Phase 5 primary p1024-n64-t4 prompt throughput was F16 304.158 tokens/s, Q8_0 248.188 tokens/s, and Q4_K_M 195.502 tokens/s. Generation reversed that ordering: 30.560, 52.194, and 62.438 tokens/s respectively.

## Hardware-counter results

These are arithmetic means of three runs. Counter normalization uses all 1088 requested tokens and is therefore explicitly **whole-invocation**, not prompt-only or generation-only attribution.

| Quantization | Elapsed (s) | Cycles | Instructions | IPC | Cache-miss rate | Branch-miss rate | Page faults | Cycles/token | Instructions/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | 21.090 | 274.240B | 723.572B | 2.638 | 20.963% | 0.3219% | 69,924 | 252.058M | 665.048M |
| Q8_0 | 23.949 | 302.911B | 899.497B | 2.970 | 17.571% | 0.3100% | 60,937 | 278.411M | 826.743M |
| Q4_K_M | 29.571 | 370.913B | 1,134.167B | 3.059 | 21.339% | 0.2881% | 67,839 | 340.913M | 1,042.433M |

| Relative to F16 | Elapsed ratio | Cycles ratio | Instructions ratio | IPC ratio | Cache-rate difference | Branch-rate difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1.136× | 1.105× | 1.243× | 1.126× | -3.392 pp | -0.0119 pp |
| Q4_K_M | 1.402× | 1.353× | 1.567× | 1.159× | +0.376 pp | -0.0338 pp |

- **Measured observation:** Q8_0 executed 24.3% more whole-invocation instructions and Q4_K_M 56.7% more than F16. Their higher IPC only partly compensated, leaving 10.5% and 35.3% more cycles.
- **Measured observation:** Q8_0 had a materially lower generic cache-miss rate than F16, while Q4_K_M was only 0.376 percentage points higher. Both quantized formats had slightly lower branch-miss rates. These directions do not support cache or branch misses as the primary explanation of the elapsed ordering.
- **Measured observation:** Mean page faults were lower for both quantized formats than F16 in the primary workload. Perf reported zero CPU migrations for these invocations; context-switch counts are retained in structured results. No page-fault, migration, or scheduling evidence explains quantized slowdown.

## Hot-function results

| Quantization | Function | Family | CPU share |
| --- | --- | --- | ---: |
| F16 | `tinyBLAS<...unsigned short...>::gemm_bloc<4,3>` | GEMM/GEMV | 48.83% |
| F16 | `ggml_compute_forward_flash_attn_ext_tiled` | attention | 21.59% |
| F16 | `ggml_vec_dot_f16` | original symbol retained; uncategorized | 11.08% |
| Q8_0 | `tinyBLAS_Q0_AVX<block_q8_0,...>::gemm4xN<2>` | GEMM/GEMV | 56.39% |
| Q8_0 | `ggml_compute_forward_flash_attn_ext_tiled` | attention | 18.88% |
| Q8_0 | `ggml_vec_dot_q8_0_q8_0` | quantized dot product | 5.02% |
| Q4_K_M | `tinyBLAS_Q0_AVX<block_q5_0,...>::gemm4xN<2>` | GEMM/GEMV | 49.73% |
| Q4_K_M | `ggml_compute_forward_flash_attn_ext_tiled` | attention | 16.39% |
| Q4_K_M | `ggml_vec_dot_q6_K_q8_K` | quantized dot product | 9.58% |
| Q4_K_M | `ggml_gemm_q4_K_8x8_q8_K` | GEMM/GEMV | 4.92% |

- **Verified conclusion:** The top five cumulative sampled shares were F16 89.16%, Q8_0 90.55%, and Q4_K_M 85.22%.
- **Verified conclusion:** All named GGML symbols above resolved in `libggml-cpu.so.0.16.0` or `libggml-base.so.0.16.0`. A small unresolved `libgomp` address accounted for 5.84% in Q8_0 and 4.60% in Q4_K_M, limiting synchronization attribution.
- **Measured observation:** GEMM/GEMV dominates every format, but the exact top kernels are format-specific. Quantized dot products additionally appear among the leading Q8_0/Q4_K_M samples.

## Cross-format comparison

- **Verified conclusion:** Yes, Q4_K_M executed more whole-invocation instructions than F16: 56.7% more in primary.
- **Verified conclusion:** Yes, Q8_0 executed more whole-invocation instructions than F16: 24.3% more in primary.
- **Measured observation:** IPC differed materially in the compensating direction: Q8_0 +12.6%, Q4_K_M +15.9% versus F16. Lower IPC is rejected as the cause.
- **Measured observation:** Generic cache-miss and branch-miss rates do not track the slowdown ordering. LLC-specific evidence is unavailable.
- **Verified conclusion:** The same broad GEMM/GEMV family dominates, but not the same concrete function. F16, Q8_0, and Q4_K_M use distinct tinyBLAS/kernel paths.
- **Measured observation:** Whole-invocation elapsed ordering stayed F16 < Q8_0 < Q4_K_M for every configured shape. For short/primary/long t4 it was respectively 19.206/21.236/25.174 s, 21.090/23.949/29.571 s, and 24.844/28.179/39.575 s. At t1 it was 64.659/71.355/87.335 s and at t8 16.972/20.553/23.263 s. Workload shape changed the size, not the ranking.

## Attribution

- **Measured observation:** Additional quantized instructions, rather than an IPC deficit, accompany the slower combined invocations.
- **Hypothesis:** Block unpacking, conversion, and format-specific quantized GEMM/dot-product work account for much of the additional instructions. Source-level annotation with debug lines would be needed to divide that cost among individual operations.
- **Verified conclusion:** The hottest sampled path changes from F16 tinyBLAS GEMM to quantized tinyBLAS_Q0_AVX GEMM, and Q4_K_M exposes additional quantized dot/GEMM kernels.
- **Measured observation:** Available generic cache, branch, page-fault, migration, and IPC evidence rejects each as a sufficient explanation of the ordering.
- **Hypothesis:** Synchronization may affect scaling, especially at eight threads, but unresolved `libgomp` samples and absent scheduler trace data prevent a stronger conclusion.
- **Verified conclusion:** This evidence does not establish that any format is globally compute-bound or memory-bound.

## Candidate optimization target

1. **Quantized tinyBLAS_Q0_AVX GEMM family.** It is the top path for Q8_0 (56.39%) and Q4_K_M (49.73%), inside GGML/llama.cpp. Ideal removal gives Amdahl upper bounds of 2.29× and 1.99× respectively. Actual gains will be far smaller. Correctness risk is high because arithmetic/reduction changes can alter output; implementation complexity is high and ISA-specific.
2. **Q4_K_M quantized dot/GEMM path** (`ggml_vec_dot_q6_K_q8_K` plus `ggml_gemm_q4_K_8x8_q8_K`). The measured shares are 9.58% and 4.92%; treating the sampled 14.50% together gives an ideal 1.17× upper bound for Q4_K_M. It is inside GGML/llama.cpp, affects Q4_K_M directly, carries high numerical-correctness risk, and medium-to-high implementation complexity.

These are Phase 8 investigation targets only; Phase 7 changes no inference implementation.

## Plots

Twelve primary charts were generated under ignored `profiles/analysis/`: elapsed, instructions, cycles, IPC, cache-miss rate, branch-miss rate, instructions/token, cycles/token, top-function share, and throughput versus instructions/IPC/cache-miss rate. Each is a separate matplotlib figure; unsupported metrics are omitted rather than plotted as zero.

## Limitations

Results cover WSL2 perf with virtualized/multiplexed PMU counters, one CPU, one small model family, three measured stat repetitions, one sample profile, and sampling overhead. CPU frequency, affinity placement within CPUs 0-15, temperature, page cache, and host load were uncontrolled. The Release build lacks source debug lines and explicit frame pointers; unresolved runtime symbols remain. LLC and backend-stall events were unsupported. Most importantly, counters cover a llama-bench invocation containing separate prompt-processing and generation tests; whole-invocation token normalization is not phase-specific attribution, while Phase 5 shows their quantization orderings differ.
