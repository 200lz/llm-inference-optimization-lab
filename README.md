# LLM Inference Optimization Lab

A reproducible C++ and Python laboratory for studying LLM inference optimization, created with the Preferred Networks Software Engineer - LLM Inference Optimization role as its primary target.

This foundation provides a C++17 smoke-test executable, CTest and pytest tests, CMake/Ninja presets, a pinned `llama.cpp` dependency, and a reproducible `llama-bench` harness.

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

## Benchmark harness

Validate the complete result pipeline without a model or compiled binary:

```bash
python benchmarks/run_llama_bench.py configs/baseline_smoke.yaml --smoke
```

For a real CPU baseline, copy and edit the example. Paths are resolved relative to the YAML file:

```bash
cp configs/cpu_baseline.example.yaml configs/cpu_baseline.yaml
cmake --preset release
cmake --build --preset release
python benchmarks/run_llama_bench.py configs/cpu_baseline.yaml
```

The example binary path may need adjustment to match the separate `llama.cpp` build that produced `llama-bench`. Do not use Debug results for performance conclusions. Generated JSONL and CSV files are written beneath `results/` and remain ignored by Git. See [benchmark methodology](docs/benchmarking.md) for metrics, warm-ups, repetitions, and failure handling.

## Layout

- `src/` — native C++ implementation
- `include/` — public C++ headers
- `benchmarks/` — Python benchmark scripts
- `tests/` — native and Python tests
- `configs/` — benchmark and experiment configuration
- `docs/` — project documentation
- `results/` — generated benchmark results (ignored except for guidance)
- `third_party/` — pinned external dependencies
- `.github/workflows/` — continuous integration

## License

MIT. See [LICENSE](LICENSE).
