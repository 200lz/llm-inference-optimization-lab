# F16 / Q8_0 / Q4_K_M quantization comparison

## Status

- **Verified conclusion:** Phase 5 is **completed**. 90 invocations succeeded; 0 failed, 0 timed out, and 0 had parse errors.
- **Verified conclusion:** All three exact artifacts are configured from the same Qwen2.5-0.5B-Instruct repository revision.

## Environment and experiment matrix

- **Verified conclusion:** CPU-only llama.cpp at `e3546c7948e3af463d0b401e6421d5a4c2faf565`; `-ngl 0 -dev none`, mmap enabled, batch/ubatch 512, 900-second subprocess timeout, one warm-up and five measured repetitions.
- **Verified conclusion:** Five deduplicated workloads per quantization produce 15 cases, 15 warm-ups, 75 measurements, and 90 total invocations.

## Model artifacts

- **Verified conclusion:** `qwen2.5-0.5b-instruct-fp16.gguf` (F16), 1266425696 bytes, SHA-256 `8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc`, source `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at `9217f5db79a29953eb74d5343926648285ec7e67`.
- **Verified conclusion:** `qwen2.5-0.5b-instruct-q8_0.gguf` (Q8_0), 675710816 bytes, SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`, source `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at `9217f5db79a29953eb74d5343926648285ec7e67`.
- **Verified conclusion:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M), 491400032 bytes, SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, source `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at `9217f5db79a29953eb74d5343926648285ec7e67`.

## Prompt-processing throughput

| Quantization | Workload | N | Mean | Median | Min | Max | SD | CV | Ratio to F16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | p128-n64-t4-b512-ub512 | 5 | 244.512 | 244.250 | 241.850 | 248.230 | 2.402 | 0.98% | 1.000 |
| F16 | p512-n64-t1-b512-ub512 | 5 | 87.712 | 87.280 | 86.620 | 88.980 | 1.033 | 1.18% | 1.000 |
| F16 | p512-n64-t4-b512-ub512 | 5 | 285.598 | 286.590 | 279.930 | 289.880 | 3.775 | 1.32% | 1.000 |
| F16 | p512-n64-t8-b512-ub512 | 5 | 378.816 | 379.070 | 371.290 | 384.540 | 5.122 | 1.35% | 1.000 |
| F16 | p1024-n64-t4-b512-ub512 | 5 | 304.158 | 302.730 | 299.940 | 311.970 | 4.725 | 1.55% | 1.000 |
| Q4_K_M | p128-n64-t4-b512-ub512 | 5 | 168.694 | 169.910 | 154.110 | 182.300 | 10.695 | 6.34% | 0.690 |
| Q4_K_M | p512-n64-t1-b512-ub512 | 5 | 62.580 | 62.630 | 61.810 | 63.100 | 0.482 | 0.77% | 0.713 |
| Q4_K_M | p512-n64-t4-b512-ub512 | 5 | 187.702 | 190.820 | 163.710 | 200.900 | 14.191 | 7.56% | 0.657 |
| Q4_K_M | p512-n64-t8-b512-ub512 | 5 | 239.520 | 240.300 | 230.400 | 246.240 | 5.718 | 2.39% | 0.632 |
| Q4_K_M | p1024-n64-t4-b512-ub512 | 5 | 195.502 | 197.460 | 176.610 | 209.860 | 12.157 | 6.22% | 0.643 |
| Q8_0 | p128-n64-t4-b512-ub512 | 5 | 215.790 | 215.160 | 213.590 | 218.680 | 2.047 | 0.95% | 0.883 |
| Q8_0 | p512-n64-t1-b512-ub512 | 5 | 78.194 | 78.120 | 77.360 | 78.820 | 0.557 | 0.71% | 0.891 |
| Q8_0 | p512-n64-t4-b512-ub512 | 5 | 242.224 | 244.230 | 233.390 | 250.000 | 7.015 | 2.90% | 0.848 |
| Q8_0 | p512-n64-t8-b512-ub512 | 5 | 280.652 | 279.710 | 277.060 | 286.210 | 3.386 | 1.21% | 0.741 |
| Q8_0 | p1024-n64-t4-b512-ub512 | 5 | 248.188 | 248.390 | 247.210 | 248.680 | 0.573 | 0.23% | 0.816 |

## Token-generation throughput

| Quantization | Workload | N | Mean | Median | Min | Max | SD | CV | Ratio to F16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | p128-n64-t4-b512-ub512 | 5 | 27.972 | 27.900 | 27.670 | 28.370 | 0.283 | 1.01% | 1.000 |
| F16 | p512-n64-t1-b512-ub512 | 5 | 15.648 | 15.630 | 15.600 | 15.710 | 0.045 | 0.29% | 1.000 |
| F16 | p512-n64-t4-b512-ub512 | 5 | 29.188 | 29.140 | 28.990 | 29.440 | 0.188 | 0.64% | 1.000 |
| F16 | p512-n64-t8-b512-ub512 | 5 | 24.112 | 23.840 | 23.510 | 25.000 | 0.640 | 2.65% | 1.000 |
| F16 | p1024-n64-t4-b512-ub512 | 5 | 30.560 | 30.460 | 30.200 | 30.970 | 0.299 | 0.98% | 1.000 |
| Q4_K_M | p128-n64-t4-b512-ub512 | 5 | 52.214 | 52.180 | 51.860 | 52.730 | 0.350 | 0.67% | 1.867 |
| Q4_K_M | p512-n64-t1-b512-ub512 | 5 | 21.166 | 21.460 | 20.210 | 21.550 | 0.565 | 2.67% | 1.353 |
| Q4_K_M | p512-n64-t4-b512-ub512 | 5 | 55.656 | 54.870 | 54.250 | 57.820 | 1.507 | 2.71% | 1.907 |
| Q4_K_M | p512-n64-t8-b512-ub512 | 5 | 57.394 | 57.830 | 55.730 | 58.170 | 0.996 | 1.74% | 2.380 |
| Q4_K_M | p1024-n64-t4-b512-ub512 | 5 | 62.438 | 62.420 | 61.390 | 64.060 | 1.119 | 1.79% | 2.043 |
| Q8_0 | p128-n64-t4-b512-ub512 | 5 | 45.060 | 44.860 | 44.570 | 45.580 | 0.441 | 0.98% | 1.611 |
| Q8_0 | p512-n64-t1-b512-ub512 | 5 | 21.844 | 21.830 | 21.760 | 21.920 | 0.066 | 0.30% | 1.396 |
| Q8_0 | p512-n64-t4-b512-ub512 | 5 | 47.748 | 47.610 | 47.270 | 48.750 | 0.605 | 1.27% | 1.636 |
| Q8_0 | p512-n64-t8-b512-ub512 | 5 | 44.860 | 45.190 | 43.960 | 45.540 | 0.689 | 1.54% | 1.860 |
| Q8_0 | p1024-n64-t4-b512-ub512 | 5 | 52.194 | 52.230 | 51.280 | 53.060 | 0.851 | 1.63% | 1.708 |

## Invocation duration

| Quantization | Workload | N | Mean | Median | Min | Max | SD | CV | Ratio to F16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | p128-n64-t4-b512-ub512 | 5 | 9.017 | 9.021 | 8.974 | 9.049 | 0.032 | 0.35% | N/A |
| F16 | p512-n64-t1-b512-ub512 | 5 | 27.762 | 27.841 | 27.292 | 27.968 | 0.274 | 0.99% | N/A |
| F16 | p512-n64-t4-b512-ub512 | 5 | 9.828 | 9.836 | 9.730 | 9.895 | 0.062 | 0.63% | N/A |
| F16 | p512-n64-t8-b512-ub512 | 5 | 8.490 | 8.509 | 8.397 | 8.564 | 0.070 | 0.83% | N/A |
| F16 | p1024-n64-t4-b512-ub512 | 5 | 11.192 | 11.199 | 11.091 | 11.300 | 0.078 | 0.69% | N/A |
| Q4_K_M | p128-n64-t4-b512-ub512 | 5 | 11.802 | 11.530 | 11.321 | 13.070 | 0.719 | 6.09% | N/A |
| Q4_K_M | p512-n64-t1-b512-ub512 | 5 | 38.168 | 38.223 | 37.901 | 38.442 | 0.206 | 0.54% | N/A |
| Q4_K_M | p512-n64-t4-b512-ub512 | 5 | 13.463 | 13.077 | 12.915 | 15.144 | 0.948 | 7.04% | N/A |
| Q4_K_M | p512-n64-t8-b512-ub512 | 5 | 10.756 | 10.696 | 10.662 | 11.019 | 0.148 | 1.38% | N/A |
| Q4_K_M | p1024-n64-t4-b512-ub512 | 5 | 15.563 | 15.118 | 14.908 | 17.212 | 0.949 | 6.09% | N/A |
| Q8_0 | p128-n64-t4-b512-ub512 | 5 | 9.621 | 9.617 | 9.475 | 9.786 | 0.138 | 1.44% | N/A |
| Q8_0 | p512-n64-t1-b512-ub512 | 5 | 30.068 | 30.044 | 29.861 | 30.340 | 0.188 | 0.63% | N/A |
| Q8_0 | p512-n64-t4-b512-ub512 | 5 | 10.612 | 10.530 | 10.376 | 10.926 | 0.238 | 2.25% | N/A |
| Q8_0 | p512-n64-t8-b512-ub512 | 5 | 9.592 | 9.614 | 9.486 | 9.705 | 0.084 | 0.88% | N/A |
| Q8_0 | p1024-n64-t4-b512-ub512 | 5 | 12.452 | 12.472 | 12.335 | 12.517 | 0.070 | 0.56% | N/A |

## Plots

The analyzer generates file-size, phase-throughput, thread-scaling, and variability plots in the ignored local result directory. They are reproducible from the tracked configuration and are not linked here because generated benchmark artifacts are intentionally untracked.

## Deterministic output comparison

- **Verified conclusion:** 18 latest deterministic outputs were analyzed; all completed successfully: `true`.

| Prompt | Quantization | Successful | Exact match to F16 | Normalized match to F16 | Output length | Length difference to F16 |
| --- | --- | --- | --- | --- | ---: | ---: |
| factual | F16 | true | true | true | 30 | 0 |
| arithmetic | F16 | true | true | true | 446 | 0 |
| code | F16 | true | true | true | 153 | 0 |
| multilingual_zh | F16 | true | true | true | 213 | 0 |
| japanese | F16 | true | true | true | 187 | 0 |
| continuation | F16 | true | true | true | 1448 | 0 |
| factual | Q8_0 | true | true | true | 30 | 0 |
| arithmetic | Q8_0 | true | false | false | 342 | -104 |
| code | Q8_0 | true | false | false | 173 | 20 |
| multilingual_zh | Q8_0 | true | false | false | 209 | -4 |
| japanese | Q8_0 | true | false | false | 178 | -9 |
| continuation | Q8_0 | true | false | false | 1449 | 1 |
| factual | Q4_K_M | true | true | true | 30 | 0 |
| arithmetic | Q4_K_M | true | false | false | 343 | -103 |
| code | Q4_K_M | true | false | false | 416 | 263 |
| multilingual_zh | Q4_K_M | true | false | false | 151 | -62 |
| japanese | Q4_K_M | true | false | false | 172 | -15 |
| continuation | Q4_K_M | true | false | false | 1460 | 12 |

llama-cli stdout does not reliably expose generated token IDs separately from chat rendering; token agreement and first-divergence metrics are omitted.

Exact or normalized matches are descriptive and neither prove semantic equivalence nor make differences incorrect.

## Interpretation and limitations

- **Measured observation:** Only measured rows appear above; missing matching F16 references are shown as `N/A`.
- **Hypothesis:** Differences may relate to model-byte traffic or quantized kernel implementations; hardware-counter evidence would be required.
- **Verified conclusion:** File-size reductions are exact artifact-storage comparisons. Throughput alone does not establish compute-bound or memory-bound behavior, and lower precision is not assumed to be faster.
- **Verified conclusion:** This covers one WSL2 host, one small family, synthetic llama-bench workloads, five repetitions, uncontrolled affinity/frequency/temperature, and a limited deterministic prompt suite. Results may not generalize to larger models or GPUs.
- **Verified conclusion:** Peak RSS was omitted because wrapping the process would change the exact executable command and Linux/WSL2 mmap, page-cache, and shared-page accounting is not exact model-memory accounting.
- **Verified conclusion:** Analyzer excluded 0 malformed or duplicate row(s). Deterministic comparison is not a quality benchmark.
