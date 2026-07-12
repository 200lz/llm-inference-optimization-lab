# Phase 8C Q8_0 RN=2 activation-scale experiment

## Scope

Phase 8C continues from the Phase 8B selected target:
`tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` activation-scale
preparation in `third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp`.

The intended optimization packs two binary16 activation scales, converts both
with one F16C `vcvtph2ps`, and reuses the two F32 lanes for the two activation
columns.

Because `third_party/llama.cpp` is a pinned external submodule, the Phase 8C
correctness, assembly, and microbenchmark gates are run against a project-owned
extracted 4x2 Q8_0 tile harness under `kernels/q8_rn2_scale/`.

## Correctness

Command:

```console
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

The Q8_0 RN=2 harness compares:

- binary16 scale pair conversion against a scalar reference;
- optimized tile output against baseline tile output bit-for-bit;
- both tile outputs against scalar dequantization for observed depths `28` and
  `152`.

Additional validation:

```console
.venv/bin/python -m pytest
```

Result:

```text
94 passed, 3 warnings
```

Release validation:

```console
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

## Microbenchmark

Command:

```console
./build/release/q8_rn2_scale_bench
```

Environment:

- Linux under WSL2
- GCC via `/usr/bin/g++`
- CMake Release preset
- Extracted 4x2 Q8_0 tile
- Depths: `28` and `152` Q8_0 blocks, matching the Phase 8B observed shape
  families
- Warmup: 2000 alternating baseline/optimized calls
- Repetitions: 11 alternating timed repetitions
- Timing excludes random input generation

Result:

| Depth | Baseline median ns/tile | Optimized median ns/tile | Speedup | Baseline CV | Optimized CV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 226.15 | 171.61 | 1.32x | 2.07% | 1.57% |
| 152 | 1150.44 | 860.80 | 1.34x | 1.84% | 0.92% |

Performance gate: passed for both observed depths in the extracted harness.

## Assembly inspection

Command:

```console
objdump -d --start-address=0x2b30 --stop-address=0x2f70 build/release/q8_rn2_scale_bench
objdump -d --start-address=0x2f70 --stop-address=0x3650 build/release/q8_rn2_scale_bench
```

The optimized tile inner loop contains one packed activation-scale conversion:

```text
vmovd       %eax,%xmm0
vcvtph2ps   %xmm0,%xmm0
vpermilps   $0x0,%xmm0,%xmm3
vpermilps   $0x55,%xmm0,%xmm0
```

The baseline tile retains per-column scalar half handling and broadcast:

```text
vmovd        %eax,%xmm1
vbroadcastss %xmm1,%xmm1
vmulps       %xmm2,%xmm1,%xmm1
vperm2f128   $0x0,%ymm1,%ymm1,%ymm1
```

Assembly gate: passed in the extracted harness. The optimized path performs one
F16C conversion for the two activation scales and reuses the converted lanes.

## Submodule status

The submodule commit matches `docs/dependencies.md`:

```text
e3546c7948e3af463d0b401e6421d5a4c2faf565 third_party/llama.cpp
```

However, the submodule worktree is not clean:

```text
 M ggml/src/ggml-cpu/llamafile/sgemm.cpp
```

This local submodule edit was present before this Phase 8C continuation. No
additional Phase 8C edits were made inside `third_party/llama.cpp`.
