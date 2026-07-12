# Phase 8B quantized GEMM target selection

## Decision

Phase 8B selects exactly one target: the `RN == 2` activation-scale preparation
inside
`tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` in
`third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp:1523-1573`,
specifically the scalar Q8_0 activation-scale conversion at line 1546. No
optimization has been implemented and the llama.cpp submodule has not been
modified.

The proposed second attempt is to add an `RN == 2` specialization that packs the
two Q8_0 activation block `d` fields, converts both binary16 values together with
F16C, and reuses those converted values while forming the two scale vectors. The
existing template remains the fallback for other `RN` values and for builds that
do not satisfy the existing AVX2+F16C guard.

## Evidence and method

This is a source-, model-shape-, dispatch-, and assembly-level audit, not a
performance result. It used the pinned Release binary and its saved Phase 7
`perf report`/`perf annotate` files. Tensor dimensions and types were read from
the profiled Qwen2.5-0.5B GGUF files. The recorded workload was
`p1024-n64-t4`, `b512`, `ub512`, CPU only. Thus prompt processing presents two
512-token microbatches to these prompt-only GEMM paths. No arbitrary shape was
invented, no rejected benchmark was rerun, and no end-to-end benchmark was run.

The CPU is an AMD Ryzen 7 5800H. Runtime flags include AVX, AVX2, F16C, FMA,
SSSE3, BMI1, and BMI2, but no AVX-512. The binary was built with
`-O3 -march=native` and has 16 architectural YMM registers.

## Candidate inventory

### 1. Q8_0 tinyBLAS `gemm4xN<2>` — selected

- **Definition:** `ggml/src/ggml-cpu/llamafile/sgemm.cpp:1353` defines
  `tinyBLAS_Q0_AVX`; `:1523-1573` defines `gemm4xN<RN>`; `:1546` performs the
  scalar activation-scale conversion. The pinned ELF symbol starts at
  `0xbfd10`.
- **Instantiation:** `sgemm.cpp:3935-3945` constructs
  `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>`. On this 16-vector-register
  build, `mnpack` routes full tiles through `gemm4xN<2>` at `:1430-1440`.
- **Types:** Q8_0 weights (`TA = block_q8_0`), quantized Q8_0 activations
  (`TB = block_q8_0`), and F32 output (`TC = float`). Each Q8_0 block represents
  32 values and contains one binary16 scale plus 32 signed bytes.
- **Tile:** four output rows by two activation/token columns, with eight YMM
  accumulators. The inner `l` loop advances one 32-value quantization block.
- **ISA guard:** class availability is AVX, AVX2, or AVX-512F; this specialized
  tile is selected only under AVX2 and F16C (`:1521`). The observed path is
  AVX2+F16C+FMA, not AVX-512.
- **Dispatch:** `ggml_compute_forward_mul_mat` calls `llamafile_sgemm` for
  contiguous input or after conversion to the weight's `vec_dot_type`.
  `llamafile_sgemm` rejects `n < 2`, requires F32 output, matches Q8_0/Q8_0 at
  `:3935`, constructs the template, and recursively tiles through `mnpack`.
- **Observed operation shapes:** GGUF dimensions and the 512-token microbatch
  give `(m,n,k-values)` of `(896,512,4864)` for FFN-down,
  `(4864,512,896)` for FFN-gate/up, `(896,512,896)` for attention Q/output,
  and `(128,512,896)` for attention K/V. In template units, `k` is respectively
  152 or 28 Q8_0 blocks. These seven projections occur in each of 24 layers.
  The 151,936-row output matrix is not attributed to this prompt-only tile
  without a runtime call trace because output selection may reduce its token
  dimension.
- **Call coverage:** the layer structure and two 512-token microbatches derive
  336 logical projection calls (7 x 24 x 2). `llamafile_sgemm` is entered by
  every one of four compute threads, so 1,344 per-thread entries are expected;
  an instrumented call trace would be needed to claim a measured count.
- **Profile share:** 56.39% of sampled Q8_0 CPU cycles, the largest observed
  function in that workload.
- **Isolation:** the private anonymous-namespace member cannot be linked to
  directly. It can be isolated safely by extracting the unchanged block/tile
  into a project-owned harness, as Phase 8A did, or exercised through the public
  `llamafile_sgemm` entry with the observed matrices. The former is preferred
  for a bounded 4x2 kernel benchmark.

### 2. Q4_K_M Q5_0 tinyBLAS `gemm4xN<2>` — runner-up

- **Definition/instantiation:** the same generic definition at
  `sgemm.cpp:1523-1573`; `sgemm.cpp:4009-4019` instantiates
  `tinyBLAS_Q0_AVX<block_q5_0, block_q8_0, float>`. The ELF symbol starts at
  `0xc3690`.
- **Types and tile:** Q5_0 weights, Q8_0 activations, F32 output; 4x2 tile,
  eight YMM accumulators, and 32 values per weight/activation block. A Q5_0
  block contains a binary16 scale, 16 low-nibble bytes, and four high-bit bytes.
- **ISA/dispatch:** identical prompt-only `llamafile_sgemm` dispatch and
  AVX2+F16C tile guard. Q5_0 has no ARM/PPC alternative in this switch at the
  pinned revision; a failed match falls back to GGML's normal matmul path.
- **Observed shapes:** Q5_0 tensors in Q4_K_M yield
  `(4864,512,896)` for FFN gate/up, `(896,512,896)` for attention Q/output, and
  `(128,512,896)` for attention K. In template units `k = 28` blocks. These five
  Q5_0 projections occur in each of 24 layers over two microbatches: 240 logical
  calls, or an expected 960 per-thread entries. This count is derived, not
  measured.
- **Profile share:** 49.73% of sampled Q4_K_M CPU cycles.
- **Isolation:** the same constraints and safe extraction strategy as Q8_0.

### 3. Q4_K repack GEMM — fallback candidate

- **Definition:**
  `ggml/src/ggml-cpu/arch/x86/repack.cpp:2042`; generic fallback at
  `ggml/src/ggml-cpu/repack.cpp:1905`. The ELF symbol starts at `0xde750`.
- **Instantiation:**
  `repack.cpp:4087-4089` specializes
  `gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>`. The traits instance is at
  `:4534-4536`, and AVX2 runtime selection for tensors whose row count is a
  multiple of eight is at `:4600-4604`.
- **Types/tile:** repacked `block_q4_Kx8` weights, repacked `block_q8_Kx4`
  activations, F32 output. Q4_K/Q8_K superblocks represent 256 values. The
  selected trait interleaves eight weight rows in blocks of eight; the AVX2
  body computes eight output columns and processes activation rows in groups of
  four (with larger internal groupings only in its AVX-512 branch).
- **ISA/dispatch:** compile-time AVX2 or AVX-512F body with a generic fallback;
  runtime `ggml_cpu_has_avx2()` selects the 8x8 repack traits. The observed CPU
  therefore executes AVX2. The repack tensor traits quantize F32 activations to
  Q8_K, partition output rows into aligned chunks, and call the specialization
  for `nrows > 3`.
- **Observed shape:** all 12 Q4_K tensors are FFN-down weights of shape
  `[4864,896]`, producing the logical GEMM `(m,n,k-values) =
  (896,512,4864)` for each of two prompt microbatches. This derives 24 logical
  Q4_K matmuls. The kernel entry is additionally chunked across threads, so an
  exact entry count is not claimed without tracing.
- **Profile share:** 4.92% of sampled Q4_K_M CPU cycles.
- **Isolation:** unlike tinyBLAS, the kernel has an externally visible function
  and can be called directly, but a valid benchmark must reproduce the exact
  Q4_Kx8/Q8_Kx4 repacking and output strides. That makes isolation more complex
  and error-prone than extracting a 4x2 tinyBLAS tile.

## Assembly comparison

The saved annotations report local-period percentages within each symbol; they
are diagnostic sampling evidence, not instruction latency measurements.

### Common 4x2 work

Both tinyBLAS variants keep eight vector accumulators conceptually and perform
eight horizontal reductions only after the `k` loop. Each dot sequence uses
`vpsignb`, `vpmaddubsw`, `vpmaddwd`, `vcvtdq2ps`, and `vfmadd*ps`. The loop has
one backedge branch. It loads four weight vectors and two activation vectors per
block after compiler common-subexpression elimination; it does not reload the
activation vector once per output row as a literal reading of the intrinsics
might suggest. There is therefore no assembly basis for another accumulator
splitting attempt or for removing a material inner branch.

### Q8_0

The Q8_0 loop at `0xbfe90-0xc003c` directly loads signed-byte vectors with six
unaligned 256-bit loads, needs no nibble unpack, and converts the four packed
weight scales once using `vcvtph2ps`. However, each of the two source-level
`_mm_set1_ps(unhalf(B[...].d))` operations becomes a scalar half lookup followed
by `vbroadcastss`, `vmulps`, `vperm2f128`, and four `vpermilps` operations. The
two `vbroadcastss` instructions at `0xbfef1` and `0xbff91` carry 2.74% and 7.60%
of the symbol's sampled local period.

Register pressure is visible but bounded: four of the eight accumulators live
at stack offsets (`-0x18`, `0x8`, `0x28`, `0x48`) and are read/updated/stored in
the loop; four stay in YMM registers. There are no scalar spills in the dot
products and no horizontal reductions inside `k`. Address arithmetic is mostly
strength-reduced to pointer increments of 34 bytes, the Q8_0 block size.

### Q5_0

The Q5_0 loop at `0xc3850-0xc3b21` must reconstruct four 5-bit weight vectors.
Its extra sequence includes 128-bit loads, `vpsrlw`, `vinserti128`, masks,
`vpbroadcastd` high-bit metadata loads, `vpshufb`, comparisons, `vpandn`, and
`vpor` before the same signed-byte dot operations. The compiler already reuses
each unpacked weight vector for both activation columns, so no clear duplicate
unpack can be removed locally.

All eight accumulators are stack-resident in this build (`-0x18` through
`0xc8`), and a temporary converted four-weight-scale vector also spills at
`0xf8`. This is stronger spill evidence than Q8_0, but fixing it would require
retuning unpack scheduling or tile structure and risks trading spills for
repeated Q5 unpacking. Its two scalar activation-scale lookups remain visible;
the second `vbroadcastss` at `0xc3a58` has 9.22% local period. Q5_0 therefore
supports the same scale-conversion hypothesis, but it is not selected so that
Phase 8B changes only one path.

### Q4_K 8x8

The AVX2 body is much larger and combines Q4 nibble loads/unpack, six-bit scale
metadata extraction, Q8_K sums, minimum corrections, integer dot products,
conversion, and two families of float accumulators. It has nested block and
subblock branches and extensive load/permutation work. The broad function-level
4.92% sample does not isolate one instruction sequence as convincingly as the
two tinyBLAS annotations. Any useful change would first require a separately
sampled inner-region microbenchmark.

## Ranked assessment

| Rank | Target | Complexity | Correctness risk | Microbenchmark opportunity | Ideal Amdahl bound | Real-workload likelihood |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | Q8_0 4x2 activation-scale conversion | Low-medium | Low-medium: scale bits and FP operation ordering must be preserved | High: bounded change, observed `k=28/152`, direct instruction-count gate | `1/(1-0.5639) = 2.293x` | Very high: 56.39% sampled share and all 24 layers |
| 2 | Q5_0 4x2 scale conversion / spill scheduling | Medium-high | Medium-high due to Q5 unpack and register-pressure interactions | Medium: clear cost, but spill/unpack tradeoff may erase it | `1/(1-0.4973) = 1.989x` | Very high: 49.73% sampled share and five projections/layer |
| 3 | Q4_K 8x8 inner superblock | High | High: scale/min correction and repacked layouts | Medium-low until a narrower region is isolated | `1/(1-0.0492) = 1.052x` | High for 12 FFN-down tensors, but much smaller share |

The bounds assume complete removal of the sampled function and are intentionally
unattainable ideals. The selected edit removes only a few instructions from the
inner block loop, so its realistic whole-workload gain is a small fraction of
2.293x. Its advantage is a clean, testable performance hypothesis rather than a
large promised speedup.

## Proposed Phase 8B experiment (not implemented)

For `TA = block_q8_0`, `TB = block_q8_0`, and `RN == 2`, load the two activation
binary16 scales, pack them into one scalar integer/XMM value, execute one
`_mm_cvtph_ps`, and broadcast/shuffle the resulting two F32 lanes for the two
columns. Keep the four-weight-scale conversion, dot products, eight
accumulators, reduction order, stores, tile traversal, and threading unchanged.
Assembly acceptance requires eliminating both lookup-table `vbroadcastss`
scale loads and reducing scale-preparation instructions without adding spills or
loads to the dot path.

Fallback behavior is unchanged: other `RN` instantiations retain the generic
loop; non-AVX2/F16C builds retain the current generic tinyBLAS tile or normal
GGML fallback; `n < 2` continues to bypass llamafile SGEMM. No Q5_0 or Q4_K code
would be changed in this attempt.

Correctness testing should compare baseline and candidate bitwise over the exact
4x2 tile with observed `k=28` and `k=152`, plus boundary/tail coverage through
`llamafile_sgemm`. Inputs should include zero blocks, positive/negative int8
extrema, binary16 zero/subnormal/normal/max-finite scales, signed zeros, and
deterministic random blocks. A scalar dequantize-and-F32 reference should use a
documented tolerance, while baseline equivalence should be bit exact if the
scale products and accumulation order truly remain unchanged.

The microbenchmark should use the actual operation shapes above, with the
smallest measurement unit being repeated 4x2 tiles at `k=28` and `k=152` and a
second matrix-level check through `llamafile_sgemm` at the four recorded Q8_0
shape families. Randomization stays outside timing; warmups, alternating
baseline/candidate order, at least nine repetitions, median/CV reporting, and a
volatile checksum follow Phase 8A. The gate is both assembly confirmation and a
statistically stable speedup at both observed block depths; failure at either
important depth stops integration.

## Why this replaces the rejected dot-product candidate

The rejected `ggml_vec_dot_q6_K_q8_K` change added an accumulator and a final
addition while removing no unpack, conversion, or reduction work; its 9.58%
share also limited the ideal bound to 1.106x. The new target is the workload's
largest Q8_0 symbol at 56.39%, and its assembly identifies removable repeated
scalar scale-conversion/lookup work. It changes neither the dot-product
dependency structure nor the accumulator topology, so it does not repeat the
unsupported accumulator-splitting hypothesis.

## Repository state at selection

- Project worktree already contained untracked user-owned Phase 8A artifacts
  under `kernels/q4_hotpath/`; this report is the only Phase 8B file added.
- `third_party/llama.cpp` is at
  `e3546c7948e3af463d0b401e6421d5a4c2faf565`, matching
  `docs/dependencies.md`.
- The submodule worktree is clean. No llama.cpp branch was created and no
  submodule file was modified.
