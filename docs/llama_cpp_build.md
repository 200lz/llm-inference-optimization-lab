# Reproducible llama.cpp CPU Release build

This repository pins `llama.cpp` at commit
`e3546c7948e3af463d0b401e6421d5a4c2faf565` (build version 9976). The pin is
also recorded in `docs/dependencies.md`.

## Recorded environment

This build was configured on 2026-07-12 in WSL2 on x86-64:

- WSL2 kernel: `6.18.33.2-microsoft-standard-WSL2`
- Distribution: Ubuntu 26.04 LTS (Resolute)
- CPU: AMD Ryzen 7 5800H with Radeon Graphics, 8 cores / 16 threads
- C and C++ compiler: `/usr/bin/gcc` and `/usr/bin/g++`, GCC 15.2.0
- CMake: 4.2.3
- Ninja: 1.13.2

The build contains only the ggml CPU backend and uses OpenMP. CMake detected
`x86_64` / `x86` and selected `-march=native`. On the recorded host, `lscpu`
reported SSE, SSE2, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, F16C, FMA, BMI1, BMI2,
AES, SHA, VAES, and VPCLMULQDQ among the available instruction sets. It did
not report AVX-512 or AMX. Because `GGML_NATIVE=ON`, the resulting binaries
are optimized for this CPU feature set.

## Configure and build

Run from the repository root. The explicit `OFF` values prevent accidental
activation of GPU or remote backends if the host has additional SDKs installed.

```sh
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DGGML_CPU=ON -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF -DGGML_MUSA=OFF -DGGML_CANN=OFF -DGGML_RPC=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build third_party/llama.cpp/build-release --target llama-bench llama-cli llama-server --parallel
```

The executables are:

- `third_party/llama.cpp/build-release/bin/llama-bench`
- `third_party/llama.cpp/build-release/bin/llama-cli`
- `third_party/llama.cpp/build-release/bin/llama-server`

## Reproduction and verification

1. Initialize the pinned submodule with `git submodule update --init --recursive`.
2. Confirm the pin with `git -C third_party/llama.cpp rev-parse HEAD`.
3. Run the configure and build commands above.
4. Run `.venv/bin/python scripts/verify_llama_cpp_build.py`. The verifier
   checks all three files, executable permissions, and `--help` with a
   per-process timeout while reporting stdout, stderr, and return codes.
5. Optionally confirm the configured backend with
   `grep GGML_AVAILABLE_BACKENDS third_party/llama.cpp/build-release/CMakeCache.txt`;
   it must report only `ggml-cpu`.

The build tree is intentionally untracked. The repository's `.gitignore`
matches `third_party/llama.cpp/build-*`, including this build directory.

## Known limitations

- `-march=native` makes results representative of this host, but the binaries
  may not run on x86-64 CPUs missing any instruction selected by GCC. Rebuild
  on each benchmark host rather than copying these binaries between machines.
- WSL2 scheduling, virtualization, power management, and concurrent Windows
  workloads can affect timing. Pin threads and control host load for measured
  runs, and record the environment with every result.
- The disconnected build could not fetch optional llama-server web UI assets
  from npm or Hugging Face. llama.cpp completed the build without an embedded
  UI; the HTTP server and command-line help remain available.
- `LLAMA_CURL=OFF` disables model downloads through libcurl. This is deliberate
  for an offline, locally supplied GGUF benchmark workflow.
- No model is downloaded by these steps, and `--help` verification does not
  validate model loading or inference correctness.
