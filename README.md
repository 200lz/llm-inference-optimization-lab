# LLM Inference Optimization Lab

A reproducible C++ and Python laboratory for studying LLM inference optimization, created with the Preferred Networks Software Engineer - LLM Inference Optimization role as its primary target.

This initial foundation provides a C++17 smoke-test executable, CTest and pytest smoke tests, GCC/Clang CI, CMake/Ninja presets, and a pinned `llama.cpp` dependency. Benchmark implementations and results are intentionally deferred.

## Prerequisites

- Ubuntu under WSL2 or Linux
- GCC/G++ and optionally Clang
- CMake 3.25 or newer
- Ninja
- Python 3.10 or newer in `.venv`
- Git with submodule support

## Setup

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

## Build and test

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
python -m pytest
```

Use the `release` preset for optimized native builds.

## Layout

- `src/` — native C++ implementation
- `include/` — public C++ headers
- `benchmarks/` — future Python benchmark scripts
- `tests/` — native and Python tests
- `config/` — benchmark and experiment configuration
- `docs/` — project documentation
- `results/` — generated benchmark results (ignored except for guidance)
- `third_party/` — pinned external dependencies
- `.github/workflows/` — continuous integration

## License

MIT. See [LICENSE](LICENSE).
