# Phase 8C integration provenance

## Baseline

- llama.cpp submodule commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`
- Recorded dependency commit: `docs/dependencies.md` records the same commit.
- Source path: `third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp`
- Exported patch: `patches/phase8c-q8-rn2-scale-preparation.patch`
- Patch SHA-256: `577832ba582818428dc34ca19e8e5a1cb56902a1411c9612c9fb834666df2a22`
- Saved diff source: `/tmp/phase8c-audit/sgemm-preexisting.diff`
- Export verification: exported patch is byte-identical to the saved diff.

## Specialization guard

The optimized code is gated by:

```cpp
if constexpr (RN == 2 && std::is_same_v<TA, block_q8_0> && std::is_same_v<TB, block_q8_0>)
```

This confines the new path to:

```cpp
tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>
```

inside the existing `#if defined(__AVX2__) && defined(__F16C__)` block.

## Harness match

The dirty llama.cpp diff matches the extracted Phase 8C harness implementation at the operation level:

- packs two `block_q8_0::d` binary16 B scales into one 32-bit value;
- converts both scales with one `_mm_cvtph_ps(_mm_cvtsi32_si128(...))`;
- shuffles lanes `0` and `1` to form the two B-scale products with the already converted four-lane A scale;
- expands each product to a 256-bit vector with `_mm256_permute2f128_ps(..., 0)`;
- uses those prepared scale vectors in the unchanged existing dot-product loop.

The code is not text-identical because the harness uses project-owned names such as `scales[2]` and `half_reference`, while the llama.cpp integration uses the local template types, `dvec2`, and `unhalf`.

## Arithmetic-order audit

Arithmetic order is unchanged for the quantized dot-product and accumulation:

- the same four `Cv[j][i] = madd(scale_lane, updot(...), Cv[j][i])` statements remain in the same order;
- the same `j` loop order remains;
- the same `l` loop order remains;
- the same `hsum(Cv[j][i])` store order remains;
- the same `updot(_mm256_sign_epi8(...), _mm256_sign_epi8(load(B...), ...))` logic remains.

Only activation-scale preparation differs for the guarded RN=2 q8_0/q8_0 path. Baseline prepares each B scale separately through `unhalf` plus scalar broadcast. The patch prepares both B scales with one packed F16C conversion and then selects the converted lanes.

## Fallback audit

- `RN != 2`: falls through the `else` branch and retains the original `unhalf` plus broadcast preparation.
- `TA != block_q8_0`: falls through the `else` branch and retains the original preparation.
- `TB != block_q8_0`: falls through the `else` branch and retains the original preparation.
- Non-AVX2 or non-F16C builds: this whole `gemm4xN` implementation remains under the existing `#if defined(__AVX2__) && defined(__F16C__)`; other paths are not compiled through this patch.

The declaration `__m256 dvec2[RN];` is visible in every instantiation, but it is only assigned and read in the `if constexpr` guarded specialization. In sampled unrelated instantiations, generated code kept the same function size, conversion pattern, and stack-reference count.

## Assembly audit

Controlled assembly was generated from two temporary source snapshots with the same compiler and flags as the existing llama.cpp Release build command:

- baseline source: `/tmp/phase8c-llama-baseline` at clean commit `e3546c7948e3af463d0b401e6421d5a4c2faf565`;
- optimized source: `/tmp/phase8c-llama-optimized` at the same commit with only the exported patch applied;
- compiler: `/usr/bin/g++`;
- flags include `-O3 -DNDEBUG -std=gnu++17 -fPIC -march=native -fopenmp`.

Object checksums from the assembly audit:

- baseline object: `37d7d40fc6107576326ed986106baf9365663d84289bbd131e6d3dadb34be57d`
- optimized object: `7ec318bbffb3d4865c1d5f0907caac5ecfb4af2ff2449a1d6e2540c7783b323c`

Target path:

- baseline `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` size: `0x4c9`;
- optimized target size: `0x4bf`;
- baseline contains one A-scale `vcvtph2ps` and two B-scale `vbroadcastss` preparations;
- optimized contains the existing A-scale `vcvtph2ps` plus one packed B-scale `vcvtph2ps`, and no B-scale `vbroadcastss`;
- no `zmm`, opmask, or AVX-512 instruction use was found;
- no extra function calls were found;
- stack references decreased in the target sample from 37 to 35, so no new spill evidence was found.

Unrelated samples:

- `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<1>` size stayed `0x2f6`;
- `tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float>::gemm4xN<2>` size stayed `0x583`;
- both samples retained the same `vcvtph2ps` and `vbroadcastss` counts;
- both samples retained the same stack-reference counts;
- address differences are from object layout shifts after the target function changed size, not from changed instruction sequences in the sampled fallback paths.

## Existing build provenance

The current dirty source modification timestamp is:

```text
2026-07-12 20:45:36.889927898 +0900 third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp
```

Existing llama.cpp `build-release` artifacts predate the dirty patch:

```text
2026-07-12 11:25:00.744576698 +0900 sgemm.cpp.o
2026-07-12 11:25:16.679603608 +0900 libggml-cpu.so.0.16.0
2026-07-12 11:26:31.091397811 +0900 llama-bench
```

The existing `build-release` q8_0/q8_0 RN=2 object disassembly retains the baseline pattern with two B-scale `vbroadcastss` instructions. It does not contain the packed B-scale conversion patch.

Conclusion: no existing llama.cpp binary in the inspected `build-release` directory is proven to contain the patch. The timestamp and disassembly evidence indicate it does not contain the patch.

## Controlled A/B build readiness

Configured but not built:

- baseline build directory: `third_party/llama.cpp/build-phase8c-baseline`
- optimized build directory: `third_party/llama.cpp/build-phase8c-optimized`

Build source provenance:

- baseline CMake source: `/tmp/phase8c-llama-baseline`
- optimized CMake source: `/tmp/phase8c-llama-optimized`
- both source trees are at commit `e3546c7948e3af463d0b401e6421d5a4c2faf565`;
- optimized source has only `patches/phase8c-q8-rn2-scale-preparation.patch` applied;
- both use `/usr/bin/gcc` and `/usr/bin/g++`;
- both use Release build type and Ninja.

The current dirty llama.cpp worktree is not used as the baseline. Controlled baseline/optimized builds can proceed safely from these configured directories. Binary checksums should be recorded after the actual builds are run.

## Validation

Commands run:

```console
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
.venv/bin/python -m pytest
git diff --check -- . ':(exclude)third_party/llama.cpp'
git -C third_party/llama.cpp diff --check
```

Results:

- CTest: 3/3 tests passed.
- pytest: 94 passed, 3 existing plotting warnings.
- `git diff --check`: passed for main repository and submodule diff.
- Full end-to-end llama.cpp benchmark: not run.
