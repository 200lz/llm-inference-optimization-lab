# Q4_K_M hot-path optimization: stopped candidate

## Motivation

- **Measured observation:** Phase 5 found slower Q4_K_M prefill than F16.
- **Measured observation:** Phase 7 attributed 9.58% of primary Q4_K_M samples to
  `ggml_vec_dot_q6_K_q8_K`.
- **Verified conclusion:** The symbol is called from GGML matrix multiplication for
  Q6_K tensors retained in the mixed Q4_K_M model; it is not the Q4_K repack GEMM.

## Baseline implementation

- **Verified conclusion:** The x86 definition is
  `third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:2426`; the generic
  fallback is in `ggml/src/ggml-cpu/quants.c:851` and dispatch is registered in
  `ggml/src/ggml-cpu/ggml-cpu.c:328`.
- **Verified conclusion:** Inputs are 256-value Q6_K weight blocks (210 bytes) and
  Q8_K activation blocks (292 bytes). Accumulation is int32 followed by float FMA.
- **Measured observation:** Ryzen 7 5800H flags include AVX2, FMA, F16C, BMI1 and
  BMI2. The pinned build used `-O3 -march=native`; no AVX-512 is available.

## Optimization

- **Hypothesis:** Two independent `vpaddd` accumulator chains could reduce integer
  dependency latency.
- **Verified conclusion:** The candidate preserves the exact unpacking, integer
  products, correction, float conversion, FMA, and generic/AVX fallback design.
- **Verified conclusion:** It was rejected before llama.cpp integration because it
  failed the required microbenchmark performance gate.

## Correctness

- **Measured observation:** 2,000 deterministic/random trials covered zero data,
  scale extrema, mixed signs, and 1–16 blocks. Baseline and candidate were bit exact.
- **Measured observation:** Against the clarity-first scalar order, maximum absolute
  error was 0.0195312, maximum scaled relative error was 0.000414422, mean absolute
  error was 0.000447266, and mismatches were zero with tolerance
  `1e-3 + 5e-4*abs(reference)`.

## Microbenchmark

Each row used 1,000 warmups, 20,000 timed calls, nine repetitions, randomized input
outside timing, `steady_clock`, and a volatile checksum. Values below are one final
representative run; repeated runs led to the same rejection.

| Shape | Baseline median ns | Candidate median ns | Candidate speedup | Baseline CV | Candidate CV |
|---|---:|---:|---:|---:|---:|
| 1 block / 256 values | 11.98 | 12.13 | 0.988x | 1.44% | 0.34% |
| 4 blocks / 1024 values | 40.39 | 40.31 | 1.002x | 2.32% | 1.95% |
| 8 blocks / 2048 values | 71.47 | 72.44 | 0.987x | 2.32% | 0.97% |
| 16 blocks / 4096 values | 140.65 | 146.51 | 0.960x | 2.39% | 1.19% |

- **Verified conclusion:** The candidate is not a validated optimization.

## Assembly analysis

- **Measured observation:** Both variants use AVX2 integer unpack/dot instructions,
  FMA, and unaligned loads; neither contains AVX-512.
- **Hypothesis:** The added accumulator/final add and register pressure offset any
  dependency-chain benefit. See `q4_kernel_disassembly.md`.

## End-to-end A/B, Amdahl analysis, and re-profiling

- **Verified conclusion:** Not run. The explicit stop condition forbids integration
  when the optimized microbenchmark consistently regresses.
- **Verified conclusion:** The ideal infinite-kernel Amdahl bound from 9.58% share is
  1.106x. A measured-kernel prediction and observed end-to-end result are not valid
  for a rejected candidate.

## Interpretation

- **Measured observation:** Correctness passed but performance did not.
- **Hypothesis:** A useful future candidate likely needs to remove unpack/shuffle work,
  not only rearrange exact integer additions.
- **Verified conclusion:** No llama.cpp source, baseline commit, model semantics, or
  fallback was changed.

## Limitations

- **Verified conclusion:** Results cover one CPU, WSL2, one model/quantization,
  GCC-specific code generation, AVX2, limited block shapes, timing noise, no GPU,
  and no upstream-maintainability evaluation.
