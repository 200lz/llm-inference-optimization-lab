# Benchmark Results

Generated benchmark results belong in this directory and are ignored by Git. Do not commit fabricated results. Any future published result must include reproducibility metadata and be explicitly allow-listed.

`results/serving/` is the narrow S6 exception: it allow-lists a normalized
SIMULATED summary/report, command manifest, and eight bounded default-matrix raw
records. Extended-matrix raw output remains ignored. These are deterministic
synthetic-event results with dirty-worktree provenance, not release or hardware
measurements; regenerate them only through `scripts/run_serving_demo.sh
--update-reference` or an explicit `run_serving_matrix.py --update-reference`
command. Default matrix and demo runs leave tracked references unchanged.
