# Phase 8D binary provenance and recovery smoke

## Recovery audit

The Linux/WSL restart removed all four controlled temporary source and build directories (baseline and optimized variants).

They were recreated on 2026-07-12. The persistent integration patch was present
at `patches/phase8c-q8-rn2-scale-preparation.patch` and had the required
SHA-256:

```text
577832ba582818428dc34ca19e8e5a1cb56902a1411c9612c9fb834666df2a22
```

The Phase 8B and Phase 8C reports and the project-owned
`kernels/q8_rn2_scale/` harness also survived.

## Controlled sources

Both controlled sources are detached at:

```text
e3546c7948e3af463d0b401e6421d5a4c2faf565
```

The baseline temporary source snapshot is clean and has no Phase 8
patch. The optimized temporary source snapshot has exactly one
modified file:

```text
 M ggml/src/ggml-cpu/llamafile/sgemm.cpp
```

Its diff is byte-identical to the exported patch. No source from the original
dirty `third_party/llama.cpp` worktree was used for either build.

## Build configuration

Both variants used GCC/G++ 15.2.0, CMake 4.2.3, Ninja 1.13.2, Release,
`GGML_NATIVE=ON`, OpenMP, and `-march=native`. BLAS, RPC, curl, and all listed
GPU backends were disabled. The controlled CMake cache settings compare equal
apart from their required source/build paths.

Configure command, with the corresponding source and build variant substituted:

```console
cmake -S <temporary-source-variant> -B <temporary-build-variant> -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF -DGGML_OPENCL=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF
```

Build command:

```console
cmake --build <temporary-build-variant> --target llama-bench llama-cli
```

The baseline was configured and built before the optimized variant. Complete
logs and command/exit-code sidecars were stored in a temporary log directory. Configure
and build exit codes were `0` for both variants. Each tree contains 281 object
files and a `.ninja_log`. The disconnected UI asset step warned and used its
supported no-embedded-UI fallback; it did not fail either build.

## Binary identity

| Variant | Binary | Size (bytes) | SHA-256 | Build timestamp (JST) |
| --- | --- | ---: | --- | --- |
| Baseline | `llama-bench` | 17,904 | `a6c9df640eccef76e6fcd1ba0c8fcf7be1b6cd04eb58a313614843a171021a9a` | 2026-07-12 21:59:34.519810449 +0900 |
| Baseline | `llama-cli` | 1,434,720 | `1e872086193818259086464cb1ee1c470469d1f1679b460628318543120ae15a` | 2026-07-12 22:00:42.293030468 +0900 |
| Baseline | `libggml-cpu.so.0.16.0` | 1,156,160 | `dcfdd19c2c1f8e4777f0a0e83c241eeab7414ed4210c4a5fcba20d1cfd1f1eb9` | 2026-07-12 21:58:12.135162193 +0900 |
| Optimized | `llama-bench` | 17,904 | `382e58f3b9463e6d616abd7feae7f5055bd60ac8862d941fa65337f3c2faf900` | 2026-07-12 22:02:48.522714678 +0900 |
| Optimized | `llama-cli` | 1,434,720 | `e888c15051a6410a933d15e3f42045ac4b5ad5e4ab335f0a7f7ce41d8e9cf3b6` | 2026-07-12 22:03:58.731804271 +0900 |
| Optimized | `libggml-cpu.so.0.16.0` | 1,156,160 | `8a3b97c45f33d92790de3d1a7e34bb3e175d78df37f34f9820e686aab6d1b933` | 2026-07-12 22:01:30.686122688 +0900 |

All corresponding baseline and optimized checksums differ. Both `llama-bench`
and `llama-cli` variants returned exit code 0 for `--help`. Dynamic linkage
resolved each executable to libraries in its own controlled build tree.

Both `llama-cli` variants loaded and executed
`models/qwen2.5-0.5b-instruct-q8_0.gguf` with four threads, batch and ubatch
512, CPU only, mmap enabled, and one generated token. Both commands exited 0.

## Binary assembly

The linked `libggml-cpu.so.0.16.0` in each tree contains
`tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>`.

| Check | Baseline | Optimized |
| --- | ---: | ---: |
| Symbol size | `0x4c9` | `0x4bf` |
| `vcvtph2ps` | 1 | 2 |
| activation-scale `vbroadcastss` | 2 | 0 |
| target stack references | 43 | 41 |
| calls | 0 | 0 |
| AVX-512 ZMM/opmask references | 0 | 0 |

The baseline retains both scalar activation-scale preparation/broadcast
sequences. The optimized target contains the existing A-scale conversion plus
one packed B-scale `vcvtph2ps`. It adds no calls, AVX-512 instructions, or spill
evidence, and the same target specialization remains present.

## Correctness

Fixed-prompt, fixed-seed, greedy 64-token CPU executions used the same Q8_0
model and runtime settings. Both exited 0. Their normalized output was
byte-identical with SHA-256:

```text
a8f34f99772fa2e40c7fec6ba12e4025e191d8a6b3ea84256b1594abcab7afdd
```

## Smoke A/B

The smoke used Q8_0, prompt 1024, generation 64, four threads, batch/ubatch
512, CPU only, mmap enabled, one warm-up per variant, and three measured
repetitions per variant. Runs were sequential in measured order
`baseline, optimized, optimized, baseline, baseline, optimized`. Raw JSON,
stderr, commands, and exit-code files were stored in a temporary smoke directory.

| Metric | Baseline samples (tok/s) | Optimized samples (tok/s) | Baseline median / CV | Optimized median / CV | Median ratio |
| --- | --- | --- | --- | --- | --- |
| Prompt 1024 | 295.876, 300.105, 294.569 | 309.260, 298.523, 293.823 | 295.876 / 0.97% | 298.523 / 2.63% | 1.0089x |
| Generation 64 | 62.017, 58.769, 61.427 | 61.476, 60.985, 61.103 | 61.427 / 2.85% | 61.103 / 0.42% | 0.9947x |

Neither median regressed by more than 1%. However, indexed optimized/baseline
deltas ranged from -0.53% to +4.52% for prompt processing and from -0.87% to
+3.77% for generation. With only three samples, the smoke is inconclusive.
Per the Phase 8D stop condition, the full matrix was not run from this smoke.
A later focused 20-pair confirmation is documented in [Phase 8D.1](phase8d_q8_end_to_end_ab.md).

## Repository state

At this historical checkpoint, the main worktree retained Phase 8 changes and
had no whitespace errors under `git diff --check`; the submodule contained only
the expected integration edit. That edit was subsequently exported and removed.
The final portfolio state keeps the submodule clean at the pinned commit.
