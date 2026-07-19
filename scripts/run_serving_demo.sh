#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"

runner=build/debug/serving-benchmark-runner
python=${PYTHON:-$root_dir/.venv/bin/python}
if [[ ! -x "$runner" ]]; then
  echo "missing $runner; run: cmake --preset debug && cmake --build --preset debug"
  exit 2
fi
if [[ ! -x "$python" ]]; then
  echo "missing Python interpreter: $python" >&2
  echo "setup: python3 -m venv .venv && .venv/bin/python -m pip install pytest matplotlib" >&2
  echo "or set PYTHON=/path/to/python" >&2
  exit 2
fi

update_reference=false
if [[ ${1:-} == "--update-reference" ]]; then
  update_reference=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--update-reference]"
  exit 2
fi

demo_dir=${SERVING_DEMO_DIR:-$root_dir/.artifacts/serving/demo}
mkdir -p "$demo_dir"
"$python" benchmarks/generate_serving_workload.py \
  configs/serving/workloads/chat.json \
  --output "$demo_dir/chat.jsonl" --manifest "$demo_dir/chat.manifest.json" --force

if $update_reference; then
  "$python" benchmarks/run_serving_matrix.py configs/serving/matrix_small.json \
    --update-reference --force
  "$python" benchmarks/analyze_serving_results.py \
    results/serving/raw/fcfs_chat_small.jsonl \
    results/serving/raw/continuous_chat_small.jsonl \
    results/serving/raw/shared_prefix_cache_off.jsonl \
    results/serving/raw/shared_prefix_cache_on.jsonl \
    results/serving/raw/kv_small.jsonl results/serving/raw/kv_large.jsonl \
    results/serving/raw/mixed_low_load.jsonl results/serving/raw/mixed_overload.jsonl \
    --json results/serving/summary.json --csv results/serving/summary.csv \
    --markdown results/serving/report.md \
    --compare fcfs_chat_small:continuous_chat_small \
    --compare shared_prefix_cache_off:shared_prefix_cache_on \
    --compare kv_large:kv_small
  echo "updated tracked serving reference summaries"
else
  "$python" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); out=Path(sys.argv[2]); cfg=json.loads((root/"configs/serving/continuous_chat_small.json").read_text()); cfg["name"]="continuous_chat_demo"; cfg["workload"]={"path":str((out/"chat.jsonl").relative_to(root)),"manifest_path":str((out/"chat.manifest.json").relative_to(root))}; cfg["output_path"]=str((out/"result.jsonl").relative_to(root)); (out/"derived-config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")' "$root_dir" "$demo_dir"
  "$python" benchmarks/run_serving_simulation.py \
    "$demo_dir/derived-config.json" --output "$demo_dir/result.jsonl" --force
  "$python" benchmarks/analyze_serving_results.py "$demo_dir/result.jsonl" \
    --json "$demo_dir/summary.json" --markdown "$demo_dir/report.md"
  "$python" -c 'import hashlib,json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; actual=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert rows[0]["workload_sha256"] == actual' "$demo_dir/result.jsonl" "$demo_dir/chat.jsonl"
fi

echo "SIMULATED demo outputs: $demo_dir"
