#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
python=${PYTHON:-$root_dir/.venv/bin/python}
if [[ ! -x "$python" ]]; then
  echo "missing Python interpreter: $python" >&2
  echo "setup: python3 -m venv .venv && .venv/bin/python -m pip install pytest matplotlib" >&2
  echo "or set PYTHON=/path/to/python" >&2
  exit 2
fi

cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug --output-on-failure
"$python" -m pytest -q
./build/debug/serving-sim-smoke
./build/debug/continuous-batching-smoke
./build/debug/prefix-cache-smoke

verify_dir=${TMPDIR:-/tmp}/llm-serving-s6-verify
mkdir -p "$verify_dir"
"$python" benchmarks/generate_serving_workload.py \
  configs/serving/workloads/burst.json --output "$verify_dir/burst.jsonl" \
  --manifest "$verify_dir/burst.manifest.json" --force
"$python" benchmarks/run_serving_simulation.py \
  configs/serving/continuous_chat_small.json --output "$verify_dir/result.jsonl" --force
"$python" benchmarks/analyze_serving_results.py "$verify_dir/result.jsonl" \
  --json "$verify_dir/summary.json" --markdown "$verify_dir/report.md"
"$python" -m pytest -q tests/python/test_serving_documentation.py
git diff --check
echo "S6 verification passed; temporary outputs: $verify_dir"
