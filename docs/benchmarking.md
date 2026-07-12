# Benchmark methodology

The Phase 1 harness executes `llama-bench` once for every Cartesian product of the configured thread, prompt-token, generated-token, batch-size, and context-size values. Each command is an argument vector passed directly to `subprocess`; no shell parses paths or values.

The pinned `llama-bench` has no direct context-size option. Its context allocation is prompt tokens plus generated tokens plus `--n-depth`, so the harness translates a configured total context size to `n_depth = context - prompt - generated`. Configurations where the workload cannot fit in the requested context are rejected. The additional depth test emitted by `llama-bench` is retained in raw output but is not one of the normalized prompt-processing or generation metrics. CPU-only configurations pass `--n-gpu-layers 0` and `--device none`; mmap is also passed explicitly.

## Metrics

- **Prompt-processing tokens per second** is the `ppN` throughput reported by `llama-bench`: the rate at which an existing prompt is evaluated. Higher is better.
- **Generation tokens per second** is the `tgN` throughput reported by `llama-bench`: the autoregressive token-generation rate. Higher is better.
- **Elapsed wall-clock time** measures the complete child-process invocation. It includes process startup and is diagnostic metadata, not a substitute for either throughput metric.
- The **test identifier** preserves the source tests (for example, `pp128+tg32`) used to form a normalized row.

The CSV preserves model name, model size when reported, backend, threads, workload sizes, repetition, and throughput. Comparisons are meaningful only when hardware, software, model, backend, and all workload parameters match.

## Warm-up and repetitions

For each parameter combination, configured warm-up invocations run before measured repetitions. Warm-ups allow initialization, page faults, and cold caches to settle. They are retained in raw JSONL with `warmup: true`, but excluded from normalized CSV. A warm-up failure does not erase later attempts.

Repeated measurements are separate process invocations and separate CSV rows. Keep them separate when inspecting variance; report an aggregation and dispersion measure only alongside the repetition count and full methodology.

## Progress, durability, and result formats

Before every invocation, the harness prints and flushes its position in the plan, stable case ID, phase and repetition, workload, thread count, subprocess timeout, and whole-run elapsed time. After the child exits, it immediately prints the return code, duration, parse status, available pp/tg throughput, and cumulative success/failure counts. The configured subprocess timeout is separate from `--max-total-seconds`; when the latter is omitted, the whole matrix is explicitly unlimited.

Every completed invocation is appended immediately to `run.jsonl`, including warm-ups, nonzero exits, timeouts, launch failures, and parser failures. Each append is one complete JSON object plus a newline, followed by `flush()` and, by default, `fsync()`. `--no-fsync` is available for short disposable runs, but should not be used for long benchmarks. Records include stable run, case, and invocation IDs; phase and repetition; UTC start/completion timestamps; stdout, stderr, return code, elapsed time, exact command, parameter case, status, parsed values, and shared environment metadata.

`run-summary.json` is replaced atomically after each completed invocation. It records planned/completed/successful/failed/timed-out counts, the last completed case, elapsed time, the configuration fingerprint, and `running`, `interrupted`, `failed`, or `completed` status.

The JSONL file is the raw audit trail. The CSV is a convenience view containing only successfully executed, successfully parsed, non-warm-up measurements. Retain the JSONL whenever sharing CSV results.

## Resume and retry workflow

Start a new run normally. The output directory must not already contain `run.jsonl` or `run-summary.json`, which prevents accidental overwrite:

```console
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml
```

After interruption, use the same configuration and output directory:

```console
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml --resume
```

Resume validates every JSONL line and the stored configuration fingerprint. It rejects malformed/truncated JSONL and incompatible configurations. It skips invocation IDs already recorded, including failures. `--force-resume` bypasses only the fingerprint check and should be used with care.

To retry failed, timed-out, or parse-error invocations while continuing to skip successes:

```console
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml --resume --retry-failures
```

Retries append a new audit record with the same invocation ID; the latest record determines summary and CSV state. A successful invocation is never rerun by `--retry-failures`.

## Interruption and partial-result semantics

On SIGINT, the harness terminates the active child (escalating to kill after five seconds), preserves every previously fsynced JSONL record, marks the summary `interrupted`, does not create a normal CSV, and exits with status 130. The in-flight invocation has not completed and is intentionally absent, so resume reruns it.

Normalized CSV is generated automatically only after all required measured invocation IDs have records. `--write-csv` or `--allow-partial-analysis` explicitly permits an incomplete CSV; it begins with `# incomplete: true`. Partial CSV is diagnostic and must not be presented as a completed experiment. Raw JSONL and the summary remain authoritative.

## Build type

Debug builds disable or reduce optimization and may add assertions and instrumentation. Their instruction mix and timing do not represent production inference. Debug is appropriate for correctness checks only; use a reproducible Release build for performance conclusions and record its exact commit and build configuration.

Smoke mode does not execute `llama-bench`. It parses a deterministic checked-in fixture and exercises matrix expansion, warm-up filtering, raw JSONL, and normalized CSV creation. It proves pipeline wiring, not performance or binary compatibility.

## Phase 5 multi-model comparison

`configs/quantization_comparison.yaml` uses a focused union of five workload cases for each of F16, Q8_0, and Q4_K_M. Model logical ID prefixes every invocation ID, so resume cannot treat one quantization as another. Raw records and normalized rows include logical ID, filename, quantization, SHA-256, and exact file size. The configuration fingerprint includes the complete ordered model list.

The deduplicated plan is 15 model/workload cases, 15 warm-ups, 75 measured repetitions, and 90 total subprocess invocations. Each subprocess has a 900-second timeout; the matrix has no default whole-run timeout. Results go under the ignored `benchmarks/results/quantization_comparison/` directory.

The deterministic suite uses `llama-cli`, the model chat template, greedy decoding (`--temp 0`), seed 42, a fixed maximum generation length, and identical CPU-only flags. Raw output is append-only and resumable. Token-level agreement is omitted because this pinned `llama-cli` does not provide generated token IDs as a reliable machine-readable stream separate from rendered chat output.

Peak RSS is intentionally omitted. Wrapping only Phase 5 commands with `/usr/bin/time -v` would complicate the exact-command interface, while RSS with file-backed mmap does not equal model size, does not capture shared page-cache behavior exactly, and WSL2 accounting can differ from native Linux. This omission does not block storage and throughput comparisons.
