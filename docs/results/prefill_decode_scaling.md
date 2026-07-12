# Prefill versus decode scaling

## Environment and experiment

- **Verified conclusion:** Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`; quantization: Q4_K_M.
- **Verified conclusion:** CPU: AMD Ryzen 7 5800H with Radeon Graphics; OS/kernel: Linux 6.18.33.2-microsoft-standard-WSL2.
- **Verified conclusion:** llama.cpp commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`; CPU-only, mmap enabled, batch 512, ubatch 512, no GPU offload.
- **Verified conclusion:** The deduplicated experiment has 13 unique cases, 13 warm-ups, and 65 planned measured runs. The focused union isolates one variable at a time; the full 60-case Cartesian product would confound interpretation and require 360 invocations.
- **Verified conclusion:** Prompt scaling: p={32,128,512,1024,2048}, n=64, t=4. Generation scaling: p=512, n={16,64,128}, t=4. Thread scaling: p={128,1024}, n=64, t={1,2,4,8}.

## Concepts

- **Verified conclusion:** Prompt processing (prefill) evaluates the supplied prompt in parallel across tokens and builds the KV cache.
- **Verified conclusion:** Autoregressive decode generates one token at a time, reusing the KV cache; each next token depends on the preceding token.
- **Hypothesis:** Because prefill exposes token-level parallel work while decode has a serial dependency between generated tokens, their response to threads and sequence length may differ. Throughput alone does not establish whether either phase is compute-bound or memory-bound.

## Prompt-processing results

| Prompt | Generated | Threads | N | Mean | Median | Min | Max | SD | CV | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 4 | 5 | 73.246 | 73.160 | 71.230 | 75.860 | 1.905 | 2.601% | N/A | N/A |
| 128 | 64 | 1 | 5 | 48.806 | 49.030 | 47.840 | 49.500 | 0.747 | 1.532% | 1.000 | 1.000 |
| 128 | 64 | 2 | 5 | 86.260 | 87.050 | 83.820 | 87.890 | 1.903 | 2.206% | 1.767 | 0.884 |
| 128 | 64 | 4 | 5 | 133.032 | 135.830 | 119.600 | 141.190 | 8.684 | 6.528% | 2.726 | 0.681 |
| 128 | 64 | 8 | 5 | 164.522 | 164.750 | 159.560 | 171.570 | 5.104 | 3.102% | 3.371 | 0.421 |
| 512 | 16 | 4 | 5 | 152.340 | 153.200 | 148.010 | 154.590 | 2.532 | 1.662% | N/A | N/A |
| 512 | 64 | 4 | 5 | 150.052 | 152.420 | 139.630 | 157.740 | 6.820 | 4.545% | N/A | N/A |
| 512 | 128 | 4 | 5 | 156.890 | 157.000 | 155.290 | 159.530 | 1.740 | 1.109% | N/A | N/A |
| 1024 | 64 | 1 | 5 | 48.556 | 48.350 | 48.200 | 49.580 | 0.576 | 1.187% | 1.000 | 1.000 |
| 1024 | 64 | 2 | 5 | 90.172 | 90.270 | 88.390 | 91.710 | 1.272 | 1.411% | 1.857 | 0.929 |
| 1024 | 64 | 4 | 5 | 152.772 | 155.870 | 137.350 | 160.950 | 9.039 | 5.917% | 3.146 | 0.787 |
| 1024 | 64 | 8 | 5 | 194.256 | 195.950 | 189.820 | 196.410 | 2.918 | 1.502% | 4.001 | 0.500 |
| 2048 | 64 | 4 | 5 | 165.878 | 165.750 | 163.410 | 168.840 | 1.934 | 1.166% | N/A | N/A |

## Token-generation results

| Prompt | Generated | Threads | N | Mean | Median | Min | Max | SD | CV | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 4 | 5 | 38.004 | 37.890 | 35.470 | 39.640 | 1.671 | 4.397% | N/A | N/A |
| 128 | 64 | 1 | 5 | 14.214 | 14.250 | 13.990 | 14.410 | 0.172 | 1.212% | 1.000 | 1.000 |
| 128 | 64 | 2 | 5 | 23.788 | 24.570 | 21.760 | 25.380 | 1.606 | 6.752% | 1.674 | 0.837 |
| 128 | 64 | 4 | 5 | 38.476 | 38.310 | 38.120 | 39.200 | 0.436 | 1.133% | 2.707 | 0.677 |
| 128 | 64 | 8 | 5 | 43.274 | 43.220 | 43.180 | 43.390 | 0.098 | 0.227% | 3.044 | 0.381 |
| 512 | 16 | 4 | 5 | 40.248 | 40.630 | 37.900 | 41.830 | 1.518 | 3.771% | N/A | N/A |
| 512 | 64 | 4 | 5 | 40.962 | 41.300 | 38.920 | 42.620 | 1.580 | 3.857% | N/A | N/A |
| 512 | 128 | 4 | 5 | 41.374 | 41.430 | 40.510 | 41.930 | 0.564 | 1.362% | N/A | N/A |
| 1024 | 64 | 1 | 5 | 16.054 | 16.180 | 15.130 | 16.550 | 0.541 | 3.372% | 1.000 | 1.000 |
| 1024 | 64 | 2 | 5 | 28.266 | 28.470 | 27.190 | 28.740 | 0.627 | 2.218% | 1.761 | 0.880 |
| 1024 | 64 | 4 | 5 | 43.348 | 43.370 | 41.820 | 45.430 | 1.370 | 3.161% | 2.700 | 0.675 |
| 1024 | 64 | 8 | 5 | 47.894 | 47.810 | 47.520 | 48.370 | 0.320 | 0.669% | 2.983 | 0.373 |
| 2048 | 64 | 4 | 5 | 52.316 | 52.530 | 50.650 | 53.460 | 1.024 | 1.956% | N/A | N/A |

## Variability, scaling, and failures

- **Measured observation:** 78 of 78 invocations succeeded; 0 failed (0 timed out, 0 parse failures). Total subprocess wall time was 2870.704 seconds.
- **Measured observation:** Duplicate or malformed normalized rows produced 0 analyzer warning(s); such rows were excluded.
- **Measured observation:** Speedup and efficiency are reported only where a matching one-thread workload exists; `N/A` explicitly denotes a missing baseline.

## Plots

![Prompt length versus prompt processing](prompt_length_prompt_processing.png)

![Prompt length versus generation](prompt_length_generation.png)

![Threads versus prompt processing](threads_prompt_processing.png)

![Threads versus generation](threads_generation.png)

![Generation length versus generation](generation_length_generation.png)

![Coefficient of variation](coefficient_of_variation.png)

## Limitations

- **Verified conclusion:** These measurements cover one WSL2 host, one Q4_K_M model, one llama.cpp commit, fixed batch/ubatch values, five repetitions, and synthetic llama-bench workloads.
- **Verified conclusion:** CPU affinity, frequency, temperature, power mode, and competing host load were not controlled or recorded. Results do not justify compute-bound, memory-bound, or optimization claims and should not be generalized to other systems.
