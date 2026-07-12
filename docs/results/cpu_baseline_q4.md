# CPU Q4 baseline

## Environment and method

These results were measured on 2026-07-12 under WSL2; no values in this report are fabricated.

- CPU: AMD Ryzen 7 5800H with Radeon Graphics, 8 cores / 16 logical CPUs
- Host interface: WSL2, Linux kernel `6.18.33.2-microsoft-standard-WSL2`, x86-64
- Model: Qwen2.5-0.5B-Instruct, `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Quantization: Q4_K_M
- Model artifact: 491,400,032 bytes; SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- llama.cpp commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`
- Release executable: `third_party/llama.cpp/build-release/bin/llama-bench`
- Settings: 1, 2, and 4 threads; 128 and 512 prompt tokens; 64 generated tokens; batch 512; total context 1024; mmap enabled; zero GPU layers; device `none`; one warm-up and five measured process invocations per case; 900-second per-process timeout
- Run command: `.venv/bin/python benchmarks/run_llama_bench.py configs/cpu_baseline_q4.yaml`
- Analyzer command: `.venv/bin/python benchmarks/analyze_baseline.py benchmarks/results/cpu_baseline_q4/llama-bench-20260712T025239Z.csv --model-filename qwen2.5-0.5b-instruct-q4_k_m.gguf --output benchmarks/results/cpu_baseline_q4/analysis.json`

The ignored raw audit file contains the exact argument vector, stdout, stderr, return code, elapsed wall time, timeout state, case, repetition, warm-up marker, and environment metadata for each invocation. The ignored normalized CSV contains the 30 successful measured rows.

## Prompt-processing throughput

All rates are measured tokens/second. SD is sample standard deviation and CV is `SD / mean`.

| Prompt | Threads | Count | Mean | Median | Min | Max | SD | CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 5 | 73.866 | 74.180 | 72.540 | 75.160 | 1.161 | 1.572% |
| 128 | 2 | 5 | 130.090 | 128.560 | 125.550 | 135.020 | 3.880 | 2.982% |
| 128 | 4 | 5 | 215.038 | 213.770 | 203.280 | 227.980 | 9.754 | 4.536% |
| 512 | 1 | 5 | 73.808 | 73.980 | 71.510 | 76.240 | 1.698 | 2.301% |
| 512 | 2 | 5 | 133.594 | 133.610 | 132.580 | 135.010 | 0.938 | 0.702% |
| 512 | 4 | 5 | 217.244 | 222.300 | 196.160 | 228.440 | 12.751 | 5.870% |

## Generation throughput

| Prompt | Threads | Count | Mean | Median | Min | Max | SD | CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 5 | 25.790 | 25.980 | 24.980 | 26.350 | 0.518 | 2.008% |
| 128 | 2 | 5 | 42.202 | 42.460 | 40.090 | 43.810 | 1.488 | 3.525% |
| 128 | 4 | 5 | 62.492 | 62.730 | 61.000 | 63.320 | 0.876 | 1.402% |
| 512 | 1 | 5 | 28.438 | 28.770 | 27.230 | 28.890 | 0.691 | 2.429% |
| 512 | 2 | 5 | 47.994 | 47.960 | 46.110 | 49.530 | 1.261 | 2.628% |
| 512 | 4 | 5 | 68.728 | 68.440 | 67.920 | 70.080 | 0.938 | 1.365% |

## Variability and failures

Thirty of thirty measured invocations succeeded and parsed; all six warm-ups also succeeded. There were zero nonzero returns, timeouts, and parse failures. Across the 12 groups, CV ranged from 0.702% to 5.870%. The largest measured CV was prompt processing with 512 prompt tokens and four threads.

## Initial observations

- **Observation:** Within this run, the mean reported throughput was higher at each successive tested thread count for both test types and both prompt lengths.
- **Observation:** Prompt-processing CV was highest in the four-thread groups (4.536% for 128 tokens and 5.870% for 512 tokens). Generation CV ranged from 1.365% to 3.525%.
- **Hypothesis:** Background WSL2 scheduling or CPU frequency/thermal behavior may contribute to the wider four-thread prompt-processing spread. This run did not measure frequency, temperature, or competing host load, so it does not verify that explanation.
- **Verified conclusion:** The pinned CPU-only Release llama-bench completed the defined Q4 baseline matrix and produced 30 usable measurements with no failed runs. This is a procedural conclusion, not an optimization claim.

## Limitations

This is one WSL2 host, one model and quantization, one llama.cpp commit, one batch size, at most four threads, two prompt lengths, one generation length, and five repetitions per group. CPU affinity, host load, power mode, frequency, and temperature were not controlled or recorded. The llama-bench generated-token test is synthetic and does not represent application end-to-end latency. Results should not be generalized to native Linux or other hardware, and this phase makes no optimization claim.
