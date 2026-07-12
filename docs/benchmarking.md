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

## Failures and result formats

Every invocation produces a JSONL record, including warm-ups, nonzero exits, timeouts, launch failures, and parser failures. Records contain stdout, stderr, return code, elapsed time, exact command, parameter case, and shared environment metadata. A timeout or launch failure has a null return code. Parser errors are annotated rather than discarded.

The JSONL file is the raw audit trail. The CSV is a convenience view containing only successfully executed, successfully parsed, non-warm-up measurements. Retain the JSONL whenever sharing CSV results.

## Build type

Debug builds disable or reduce optimization and may add assertions and instrumentation. Their instruction mix and timing do not represent production inference. Debug is appropriate for correctness checks only; use a reproducible Release build for performance conclusions and record its exact commit and build configuration.

Smoke mode does not execute `llama-bench`. It parses a deterministic checked-in fixture and exercises matrix expansion, warm-up filtering, raw JSONL, and normalized CSV creation. It proves pipeline wiring, not performance or binary compatibility.
