# KV-cache and context-length scaling

## Status

- **Measured observation:** Completed: 216 unique invocations, comprising 36 warm-ups and 180 measured repetitions. Successes: 216; failures, timeouts, and parse errors: 0, 0, and 0.
- **Measured observation:** Total durable benchmark elapsed time was 6207.82 seconds.

## Environment

- **Verified conclusion:** AMD Ryzen 7 5800H with Radeon Graphics; Linux `6.18.33.2-microsoft-standard-WSL2`.
- **Verified conclusion:** llama.cpp commit `e3546c7948e3af463d0b401e6421d5a4c2faf565`; model `qwen2.5-0.5b-instruct-q4_k_m.gguf`, SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, 491,400,032 bytes.
- **Verified conclusion:** CPU only, threads 4, batch/ubatch 512/512, mmap enabled, and 900-second timeout.

## KV-cache support

- **Verified conclusion:** Supported key/value types: `bf16`, `f16`, `f32`, `iq4_nl`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`.
- **Verified conclusion:** Tested symmetric `f16/f16`, `q8_0/q8_0`, and `q4_0/q4_0`. Other supported and asymmetric combinations were omitted; no unsupported type was substituted.
- **Verified conclusion:** Unified KV was not varied because it is not exposed by the pinned `llama-bench`.

## Experiment matrix

- **Verified conclusion:** Actual prompt/context growth: 512/256, 1024/768, 2048/1536, and 4096/3072, all generating 32 tokens.
- **Verified conclusion:** Fixed-prompt allocated-context scaling: prompt 256, generated 64, contexts 512/1024/2048/4096.
- **Verified conclusion:** Decode-length scaling: context 4096, prompt 512, generated 16/64/128/256.
- **Verified conclusion:** 36 exact workload/KV groups, with 5 measured repetitions per group.

## Performance and variability summary

- **Measured observation:** CV is sample standard deviation divided by mean and is stored as a ratio in `analysis.json`; reader-facing values are percentages. Group-metric CV ranged from 0.006338% to 24.078727%.
- **Verified conclusion:** F16-relative comparisons require exact model, context, prompt, generated tokens, threads, batch, and ubatch matching.
- **Verified conclusion:** These measurements do not establish whether a workload is compute-bound or memory-bound.

## Memory results

- **Measured observation:** Group mean peak RSS ranged from 584,746 to 700,549 KiB.
- **Measured observation:** `reported_kv_cache_bytes` is N/A because the benchmark output did not provide it; it is not inferred from RSS.
- **Verified conclusion:** Peak RSS includes mmap-backed, shared, and page-cache effects and is never an exact KV-cache allocation.

## Relative F16 comparisons

- **Verified conclusion:** Every relative value in `analysis.json` is joined on the full workload key; missing exact F16 references remain N/A.

## Key observations

- **Measured observation:** The three workload families are reported and plotted separately; their points are never connected across slices.
- **Measured observation:** In the actual-growth F16 slice, context/prompt 512/256→4096/3072 changed PP 234.84→171.93 tokens/s, TG 72.25→61.55 tokens/s, duration 3.75→38.67 s, and peak RSS 593404.00→667544.00 KiB.
- **Measured observation:** In the fixed-prompt F16 slice, context 512→4096 changed PP 204.67→130.20 tokens/s, TG 58.36→35.62 tokens/s, duration 4.94→28.16 s, and peak RSS 593237.60→700548.80 KiB.
- **Measured observation:** In the decode-length F16 slice, generated tokens 16→256 changed TG 38.91→33.94 tokens/s and duration 25.51→33.76 s.
- **Hypothesis:** Cache conversion, locality, and memory traffic may contribute to changes; profiling is required to distinguish causes.

## Correctness comparison

- **Verified conclusion:** No descriptive output comparison was executed, and no semantic-equivalence claim is made.

## CV extremes

- **Measured observation:** Minimum CV: metric `peak_rss_kib`, `kv-q8_0-q8_0`, context 4096, prompt 512, generated 16, threads 4; mean 653790.4, sample standard deviation 41.4342853203, CV ratio 6.33754874961e-05, CV 0.006337549%.
- **Measured observation:** Maximum CV: metric `generation_tokens_per_second`, `kv-f16-f16`, context 512, prompt 256, generated 64, threads 4; mean 58.356, sample standard deviation 14.0513817826, CV ratio 0.240787267506, CV 24.078726751%.

## Results

| Context | Prompt | Generated | KV cache | PP mean tok/s | TG mean tok/s | Duration mean s | Peak RSS mean KiB | KV bytes mean |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 256 | 32 | kv-f16-f16 | 234.840 | 72.248 | 3.751 | 593404.000 | N/A |
| 512 | 256 | 32 | kv-q4_0-q4_0 | 189.984 | 71.536 | 4.163 | 585032.800 | N/A |
| 512 | 256 | 32 | kv-q8_0-q8_0 | 191.934 | 71.486 | 4.161 | 586765.600 | N/A |
| 512 | 256 | 64 | kv-f16-f16 | 204.670 | 58.356 | 4.938 | 593237.600 | N/A |
| 512 | 256 | 64 | kv-q4_0-q4_0 | 186.176 | 68.486 | 4.713 | 584745.600 | N/A |
| 512 | 256 | 64 | kv-q8_0-q8_0 | 177.928 | 60.628 | 5.182 | 586496.800 | N/A |
| 1024 | 256 | 64 | kv-f16-f16 | 185.550 | 54.194 | 7.716 | 623791.200 | N/A |
| 1024 | 256 | 64 | kv-q4_0-q4_0 | 134.972 | 55.046 | 8.682 | 605140.800 | N/A |
| 1024 | 256 | 64 | kv-q8_0-q8_0 | 141.162 | 60.328 | 8.207 | 612298.400 | N/A |
| 1024 | 768 | 32 | kv-f16-f16 | 216.852 | 70.372 | 8.631 | 618470.400 | N/A |
| 1024 | 768 | 32 | kv-q4_0-q4_0 | 160.914 | 69.880 | 10.687 | 603896.000 | N/A |
| 1024 | 768 | 32 | kv-q8_0-q8_0 | 164.920 | 70.392 | 10.560 | 609516.000 | N/A |
| 2048 | 256 | 64 | kv-f16-f16 | 162.002 | 50.264 | 13.825 | 649903.200 | N/A |
| 2048 | 256 | 64 | kv-q4_0-q4_0 | 86.254 | 44.192 | 19.481 | 615272.000 | N/A |
| 2048 | 256 | 64 | kv-q8_0-q8_0 | 88.002 | 41.576 | 19.763 | 626672.800 | N/A |
| 2048 | 1536 | 32 | kv-f16-f16 | 199.482 | 67.352 | 17.656 | 634919.200 | N/A |
| 2048 | 1536 | 32 | kv-q4_0-q4_0 | 118.898 | 64.550 | 26.535 | 611258.400 | N/A |
| 2048 | 1536 | 32 | kv-q8_0-q8_0 | 122.950 | 66.184 | 25.714 | 618831.200 | N/A |
| 4096 | 256 | 64 | kv-f16-f16 | 130.196 | 35.620 | 28.163 | 700548.800 | N/A |
| 4096 | 256 | 64 | kv-q4_0-q4_0 | 56.272 | 35.668 | 48.506 | 631488.800 | N/A |
| 4096 | 256 | 64 | kv-q8_0-q8_0 | 58.970 | 36.926 | 47.105 | 654968.800 | N/A |
| 4096 | 512 | 16 | kv-f16-f16 | 148.032 | 38.914 | 25.507 | 698674.400 | N/A |
| 4096 | 512 | 16 | kv-q4_0-q4_0 | 57.280 | 37.496 | 49.127 | 630774.400 | N/A |
| 4096 | 512 | 16 | kv-q8_0-q8_0 | 59.628 | 37.808 | 47.448 | 653790.400 | N/A |
| 4096 | 512 | 64 | kv-f16-f16 | 143.902 | 39.764 | 27.564 | 698109.600 | N/A |
| 4096 | 512 | 64 | kv-q4_0-q4_0 | 58.302 | 37.390 | 49.258 | 630565.600 | N/A |
| 4096 | 512 | 64 | kv-q8_0-q8_0 | 60.798 | 38.186 | 47.773 | 653442.400 | N/A |
| 4096 | 512 | 128 | kv-f16-f16 | 144.380 | 39.218 | 28.288 | 697317.600 | N/A |
| 4096 | 512 | 128 | kv-q4_0-q4_0 | 55.360 | 36.260 | 50.627 | 630341.600 | N/A |
| 4096 | 512 | 128 | kv-q8_0-q8_0 | 61.232 | 38.322 | 48.273 | 653012.000 | N/A |
| 4096 | 512 | 256 | kv-f16-f16 | 128.610 | 33.944 | 33.758 | 692374.400 | N/A |
| 4096 | 512 | 256 | kv-q4_0-q4_0 | 56.624 | 35.576 | 54.654 | 628930.400 | N/A |
| 4096 | 512 | 256 | kv-q8_0-q8_0 | 58.692 | 35.046 | 53.673 | 650301.600 | N/A |
| 4096 | 3072 | 32 | kv-f16-f16 | 171.932 | 61.552 | 38.665 | 667544.000 | N/A |
| 4096 | 3072 | 32 | kv-q4_0-q4_0 | 77.112 | 58.298 | 75.758 | 621878.400 | N/A |
| 4096 | 3072 | 32 | kv-q8_0-q8_0 | 79.700 | 59.878 | 73.373 | 637207.200 | N/A |

## Plots

The analyzer generates throughput, duration, peak-RSS, and CV plots for all three sweeps in the ignored local result directory. They are reproducible from the tracked configuration and are not linked here because generated benchmark artifacts are intentionally untracked.

## Limitations

- **Verified conclusion:** Results cover one WSL2 CPU host, one 0.5B model, synthetic workloads, five repetitions, uncontrolled affinity/frequency/temperature, and a limited context range; they do not generalize to GPUs, larger models, or continuous batching.
- **Verified conclusion:** No semantic-equivalence test was run, so no correctness-equivalence claim is made.
