# Benchmarks

`run_llama_bench.py` executes a validated YAML workload matrix and records both a raw JSONL audit trail and normalized CSV measurements. See [benchmark methodology](../docs/benchmarking.md) for metric and failure semantics.

## Deterministic serving simulation (S6)

The serving tools use only the standard library and the project-owned native
runner. Python validates versioned JSON and uses a strict temporary TSV
boundary rather than adding a C++ JSON dependency.

Native `serving-simulator-v2` output is closed: provenance, request, iteration,
and terminal-summary records reject missing and unknown fields. Analysis first
validates lifecycle and derived timestamps, decode/ITL arrays, workload
envelopes, and request/iteration/summary conservation. Continuous runs require
iteration reconciliation; single-active FCFS has no iteration records and is
reconciled directly from requests to its summary.

```bash
python benchmarks/generate_serving_workload.py \
  configs/serving/workloads/chat.json --output /tmp/chat.jsonl
python benchmarks/run_serving_simulation.py \
  configs/serving/continuous_chat_small.json --output /tmp/result.jsonl
python benchmarks/run_serving_matrix.py configs/serving/matrix_small.json --dry-run
python benchmarks/analyze_serving_results.py /tmp/result.jsonl \
  --json /tmp/serving-summary.json
python benchmarks/plot_serving_results.py /tmp/serving-summary.json \
  --output-dir /tmp/serving-plots
```

Every serving result is labeled SIMULATED. Default matrix output is written
below ignored `.artifacts/serving/`; `--update-reference` is required for
approved checked-in reference paths. Use `--force` deliberately within the
selected output root and `--max-runs` to opt into a larger matrix. See the
[serving final report](../docs/serving/final_report.md).
