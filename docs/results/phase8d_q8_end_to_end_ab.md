# Phase 8D.1 Q8_0 paired end-to-end confirmation

## Prior smoke

- **Measured observation:** The initial three-run smoke reported a prompt-processing median ratio of `1.0089x` and a generation median ratio of `0.9947x`.
- **Measured observation:** Initial prompt samples had CVs of 0.97% for baseline and 2.63% for optimized, while indexed prompt differences ranged from -0.53% to +4.52%.
- **Verified conclusion:** Three repetitions with those mixed pair effects could not distinguish the approximately +0.9% prompt trend from execution noise.

## Provenance

- **Verified conclusion:** Baseline `/tmp/phase8c-build-baseline/bin/llama-bench` retained SHA-256 `a6c9df640eccef76e6fcd1ba0c8fcf7be1b6cd04eb58a313614843a171021a9a`.
- **Verified conclusion:** Optimized `/tmp/phase8c-build-optimized/bin/llama-bench` retained SHA-256 `382e58f3b9463e6d616abd7feae7f5055bd60ac8862d941fa65337f3c2faf900`.
- **Verified conclusion:** Both sources remained at `e3546c7948e3af463d0b401e6421d5a4c2faf565`; baseline was clean and optimized differed only by the exported `sgemm.cpp` patch.
- **Verified conclusion:** The patch SHA-256 remained `577832ba582818428dc34ca19e8e5a1cb56902a1411c9612c9fb834666df2a22`, and the optimized diff remained byte-identical to it.
- **Verified conclusion:** Controlled CMake settings compared equal, both binaries passed `--help`, and both loaded the Q8_0 model under the selected affinity without rebuilding.

## Paired design

- **Verified conclusion:** The only workload was `qwen2.5-0.5b-instruct-q8_0.gguf`, prompt 1024, generation 64, threads 4, batch 512, ubatch 512, CPU only, `-ngl 0`, `-dev none`, and mmap enabled, matching the prior smoke context behavior.
- **Verified conclusion:** `lscpu -e` identified logical sibling pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`; affinity `0,2,4,6` therefore selected four distinct reported physical cores.
- **Verified conclusion:** `taskset` confirmed the exact affinity list `0,2,4,6` for target commands.
- **Verified conclusion:** Each variant received three warm-up runs followed by 20 measured pairs, with one baseline and one optimized invocation in every pair.
- **Verified conclusion:** Pair order was deterministically randomized with seed `8081`; all runs were sequential and no pair was incomplete.
- **Verified conclusion:** The resumable runner persisted 46 successful records: 6 warm-ups and 40 measurements, with no command failure or parse failure.
- **Verified conclusion:** Each record contains pair/order/execution IDs, timestamps, exact command, phase throughput, wall and `/usr/bin/time` duration, peak RSS, and binary SHA-256.

## Environment

- **Measured observation:** The WSL2 kernel was `6.18.33.2-microsoft-standard-WSL2` on an AMD Ryzen 7 5800H with 16 logical CPUs reported as 8 cores with 2 threads per core.
- **Measured observation:** `/proc/cpuinfo` reported 3194.030 MHz before and after; WSL2 exposed neither cpufreq controls nor thermal-zone readings.
- **Measured observation:** Before the run, load averages were `0.07, 1.07, 4.46`; after the run they were `3.01, 3.16, 4.12`, with the benchmark itself contributing to the post-run load.
- **Measured observation:** Available memory was approximately 6.5 GiB both before and after, and no unrelated high-CPU process was observed in the captured process lists.
- **Verified conclusion:** No CPU governor, Windows power setting, or other system-wide control was changed.

## Paired results

- **Verified conclusion:** Throughput differences use optimized minus baseline, so positive is better; duration differences use the same arithmetic, so negative is better.
- **Verified conclusion:** Confidence intervals are deterministic 95% percentile bootstrap intervals for the median paired percent difference using seed `20260801` and 50,000 resamples.

| Metric | Baseline mean / median | Optimized mean / median | Baseline / optimized CV | Median paired change | Mean paired change | Bootstrap 95% CI | Signs optimized / baseline / tie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Measured observation:** Prompt throughput (tokens/s) | 232.686 / 232.911 | 230.568 / 228.113 | 8.74% / 9.63% | -1.571% | -0.608% | [-3.538%, +2.252%] | 8 / 12 / 0 |
| **Measured observation:** Generation throughput (tokens/s) | 42.708 / 44.847 | 42.534 / 42.372 | 13.42% / 11.96% | -1.560% | +0.455% | [-2.996%, +5.063%] | 9 / 11 / 0 |
| **Measured observation:** Invocation duration (seconds) | 10.678 / 10.450 | 10.771 / 10.780 | 9.25% / 7.79% | +1.170% | +1.169% | [-1.392%, +3.099%] | 8 / 12 / 0 |

- **Measured observation:** Prompt paired absolute differences ranged from -47.816 to +40.152 tokens/s, with sample standard deviation 20.172 tokens/s.
- **Measured observation:** Generation paired absolute differences ranged from -11.185 to +8.841 tokens/s, with sample standard deviation 4.840 tokens/s.
- **Measured observation:** Duration paired absolute differences ranged from -1.320 to +1.430 seconds, with sample standard deviation 0.679 seconds.
- **Verified conclusion:** Every metric's confidence interval includes zero, and none establishes a directional effect.

## Drift

- **Measured observation:** Prompt median paired change was +0.201% in pairs 1-10 and -1.952% in pairs 11-20, a 2.153 percentage-point shift.
- **Measured observation:** Prompt paired-percent change versus pair index had slope -0.159 percentage points per pair and correlation -0.106.
- **Measured observation:** Baseline and optimized prompt throughput slopes versus execution index were -1.081 and -1.275 tokens/s per execution, respectively.
- **Measured observation:** Generation first-half and second-half paired medians were -2.338% and -0.481%; duration values were +1.707% and +1.019%.
- **Verified conclusion:** The predefined greater-than-1-percentage-point half-to-half prompt threshold marks temporal drift as material, even though the paired correlation alone is weak.

![Paired prompt difference](../../benchmarks/results/phase8d1_q8_paired/paired_prompt_difference.png)

![Paired generation difference](../../benchmarks/results/phase8d1_q8_paired/paired_generation_difference.png)

![Paired duration difference](../../benchmarks/results/phase8d1_q8_paired/paired_duration_difference.png)

![Prompt throughput by execution order](../../benchmarks/results/phase8d1_q8_paired/prompt_throughput_execution_order.png)

## Interpretation and decision

- **Hypothesis:** WSL2 host scheduling or power/thermal behavior not exposed to the guest may contribute to the large phase throughput variation and the declining execution-order trend.
- **Verified conclusion:** The acceptance gates fail because prompt median change is negative, its CI includes zero, only 40% of pairs favor optimized, CV exceeds the 3% controlled threshold, and drift is material.
- **Verified conclusion:** The rejection gates are not established because prompt, generation, and duration confidence intervals all include zero; the observed direction is not consistent enough to prove a material regression.
- **Verified conclusion:** Optional `perf stat` was not run because the paired prompt result was not positive; hardware counters cannot override the noisy end-to-end result.
- **Verified conclusion:** Final decision: **REMAIN INCONCLUSIVE**. Do not proceed to the full Phase 8D matrix from this dataset.

## Artifacts

- **Verified conclusion:** The resumable raw records, manifest, environment snapshots, analysis JSON, and plots are under ignored path `benchmarks/results/phase8d1_q8_paired/`.
- **Verified conclusion:** Reproducible runner and analysis logic are in `benchmarks/run_phase8_paired.py` and `benchmarks/analyze_phase8_optimization.py`, with focused tests in `tests/python/test_phase8_optimization.py`.
