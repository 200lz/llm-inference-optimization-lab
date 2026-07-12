#!/usr/bin/env python3
"""Analyze prefill/decode scaling CSV and raw JSONL, then render JSON, Markdown, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = {
    "prompt_processing": "prompt_tokens_per_second",
    "generation": "generation_tokens_per_second",
}
KEY_FIELDS = ("thread_count", "prompt_tokens", "generated_tokens", "batch_size", "context_size", "repetition")


def _number(value: str, field: str, line: int, integer: bool = False) -> float | int:
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: malformed {field}: {value!r}") from exc
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError(f"line {line}: non-finite {field}: {value!r}")
    if parsed < 0:
        raise ValueError(f"line {line}: negative {field}: {value!r}")
    return parsed


def load_normalized(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Strictly parse rows, skip exact duplicate identities, and return warnings."""
    rows, warnings, seen = [], [], set()
    with path.open(encoding="utf-8", newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        reader = csv.DictReader(lines)
        required = set(KEY_FIELDS) | set(METRICS.values()) | {"model_name"}
        missing = required - set(reader.fieldnames or ())
        if not reader.fieldnames or missing:
            raise ValueError(f"missing CSV fields: {', '.join(sorted(missing))}")
        for line, raw in enumerate(reader, 2):
            try:
                row = {field: _number(raw[field], field, line, True) for field in KEY_FIELDS}
                row.update({field: _number(raw[field], field, line) for field in METRICS.values()})
                row["model_name"] = raw["model_name"]
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            identity = tuple(row[field] for field in KEY_FIELDS)
            if identity in seen:
                warnings.append(f"line {line}: duplicate normalized record skipped: {identity}")
                continue
            seen.add(identity)
            rows.append(row)
    return rows, warnings


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        for phase, field in METRICS.items():
            grouped[(phase, row["prompt_tokens"], row["generated_tokens"], row["thread_count"])].append(row[field])
    results = []
    for (phase, prompt, generated, threads), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        results.append({"phase": phase, "prompt_tokens": prompt, "generated_tokens": generated,
                        "thread_count": threads, "count": len(values), "mean": mean,
                        "median": statistics.median(values), "minimum": min(values),
                        "maximum": max(values), "standard_deviation": sd,
                        "coefficient_of_variation": sd / mean if mean else None,
                        "speedup": None, "parallel_efficiency": None})
    baselines = {(r["phase"], r["prompt_tokens"], r["generated_tokens"]): r["mean"]
                 for r in results if r["thread_count"] == 1}
    for row in results:
        baseline = baselines.get((row["phase"], row["prompt_tokens"], row["generated_tokens"]))
        if baseline:
            row["speedup"] = row["mean"] / baseline
            row["parallel_efficiency"] = row["speedup"] / row["thread_count"]
    return results


def failures(path: Path | None) -> dict[str, Any]:
    summary = {"total_runs": 0, "warmup_runs": 0, "measured_runs": 0, "successful_runs": 0,
               "failed_runs": 0, "timed_out_runs": 0, "parse_failures": 0, "total_duration_seconds": 0.0,
               "failed_cases": []}
    if path is None or not path.exists():
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            summary["failed_runs"] += 1
            continue
        summary["total_runs"] += 1
        summary["total_duration_seconds"] += float(record.get("elapsed_seconds", 0))
        warmup = bool(record.get("warmup"))
        summary["warmup_runs" if warmup else "measured_runs"] += 1
        failed = record.get("return_code") != 0 or record.get("timed_out") or "parse_error" in record
        if failed:
            summary["failed_runs"] += 1
            summary["timed_out_runs"] += int(bool(record.get("timed_out")))
            summary["parse_failures"] += int("parse_error" in record)
            summary["failed_cases"].append({"case": record.get("case"), "warmup": warmup,
                                             "repetition": record.get("repetition"),
                                             "return_code": record.get("return_code"),
                                             "timed_out": record.get("timed_out", False),
                                             "parse_error": record.get("parse_error")})
        else:
            summary["successful_runs"] += 1
    return summary


def _table(rows: list[dict[str, Any]], phase: str) -> str:
    selected = [r for r in rows if r["phase"] == phase]
    header = "| Prompt | Generated | Threads | N | Mean | Median | Min | Max | SD | CV | Speedup | Efficiency |\n| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for r in selected:
        fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
        body.append(f"| {r['prompt_tokens']} | {r['generated_tokens']} | {r['thread_count']} | {r['count']} | {r['mean']:.3f} | {r['median']:.3f} | {r['minimum']:.3f} | {r['maximum']:.3f} | {r['standard_deviation']:.3f} | {r['coefficient_of_variation']:.3%} | {fmt(r['speedup'])} | {fmt(r['parallel_efficiency'])} |")
    return header + ("\n" + "\n".join(body) if body else "\n| _No usable groups_ | | | | | | | | | | | |")


def render_report(results: list[dict[str, Any]], run: dict[str, Any], metadata: dict[str, Any], warnings: list[str]) -> str:
    failed = run["failed_runs"]
    return f"""# Prefill versus decode scaling

## Environment and experiment

- **Verified conclusion:** Model: `{metadata.get('model', 'unknown')}`; quantization: Q4_K_M.
- **Verified conclusion:** CPU: {metadata.get('cpu_model', 'unknown')}; OS/kernel: {metadata.get('operating_system', 'unknown')} {metadata.get('kernel', 'unknown')}.
- **Verified conclusion:** llama.cpp commit: `{metadata.get('llama_cpp_git_commit', 'unknown')}`; CPU-only, mmap enabled, batch 512, ubatch 512, no GPU offload.
- **Verified conclusion:** The deduplicated experiment has 13 unique cases, 13 warm-ups, and 65 planned measured runs. The focused union isolates one variable at a time; the full 60-case Cartesian product would confound interpretation and require 360 invocations.
- **Verified conclusion:** Prompt scaling: p={{32,128,512,1024,2048}}, n=64, t=4. Generation scaling: p=512, n={{16,64,128}}, t=4. Thread scaling: p={{128,1024}}, n=64, t={{1,2,4,8}}.

## Concepts

- **Verified conclusion:** Prompt processing (prefill) evaluates the supplied prompt in parallel across tokens and builds the KV cache.
- **Verified conclusion:** Autoregressive decode generates one token at a time, reusing the KV cache; each next token depends on the preceding token.
- **Hypothesis:** Because prefill exposes token-level parallel work while decode has a serial dependency between generated tokens, their response to threads and sequence length may differ. Throughput alone does not establish whether either phase is compute-bound or memory-bound.

## Prompt-processing results

{_table(results, 'prompt_processing')}

## Token-generation results

{_table(results, 'generation')}

## Variability, scaling, and failures

- **Measured observation:** {run['successful_runs']} of {run['total_runs']} invocations succeeded; {failed} failed ({run['timed_out_runs']} timed out, {run['parse_failures']} parse failures). Total subprocess wall time was {run['total_duration_seconds']:.3f} seconds.
- **Measured observation:** Duplicate or malformed normalized rows produced {len(warnings)} analyzer warning(s); such rows were excluded.
- **Measured observation:** Speedup and efficiency are reported only where a matching one-thread workload exists; `N/A` explicitly denotes a missing baseline.

## Plots

![Prompt length versus prompt processing](prompt_length_prompt_processing.png)

![Prompt length versus generation](prompt_length_generation.png)

![Threads versus prompt processing](threads_prompt_processing.png)

![Threads versus generation](threads_generation.png)

![Generation length versus generation](generation_length_generation.png)

![Coefficient of variation](coefficient_of_variation.png)

## Limitations

- **Verified conclusion:** These measurements cover one WSL2 host, one Q4_K_M model, one llama.cpp commit, fixed batch/ubatch values, five repetitions, and synthetic llama-bench workloads.
- **Verified conclusion:** CPU affinity, frequency, temperature, power mode, and competing host load were not controlled or recorded. Results do not justify compute-bound, memory-bound, or optimization claims and should not be generalized to other systems.
"""


def make_plots(results: list[dict[str, Any]], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)

    specs = [
        ("prompt_length_prompt_processing.png", "Prompt processing: prompt length", "Prompt tokens", "Tokens/s", lambda r: r["phase"] == "prompt_processing" and r["thread_count"] == 4 and r["generated_tokens"] == 64, "prompt_tokens"),
        ("prompt_length_generation.png", "Generation: prompt length", "Prompt tokens", "Tokens/s", lambda r: r["phase"] == "generation" and r["thread_count"] == 4 and r["generated_tokens"] == 64, "prompt_tokens"),
        ("generation_length_generation.png", "Generation: generated length", "Generated tokens", "Tokens/s", lambda r: r["phase"] == "generation" and r["thread_count"] == 4 and r["prompt_tokens"] == 512, "generated_tokens"),
    ]
    made = []
    for name, title, xlabel, ylabel, predicate, xkey in specs:
        chosen = sorted(filter(predicate, results), key=lambda r: r[xkey])
        fig, ax = plt.subplots()
        ax.plot([r[xkey] for r in chosen], [r["mean"] for r in chosen], marker="o")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(True); fig.tight_layout()
        path = output / name; fig.savefig(path); plt.close(fig); made.append(path)
    for phase, name, title in (("prompt_processing", "threads_prompt_processing.png", "Prompt processing: thread scaling"), ("generation", "threads_generation.png", "Generation: thread scaling")):
        fig, ax = plt.subplots()
        for prompt in (128, 1024):
            chosen = sorted((r for r in results if r["phase"] == phase and r["prompt_tokens"] == prompt and r["generated_tokens"] == 64), key=lambda r: r["thread_count"])
            ax.plot([r["thread_count"] for r in chosen], [r["mean"] for r in chosen], marker="o", label=f"p={prompt}")
        ax.set(title=title, xlabel="Threads", ylabel="Tokens/s"); ax.grid(True); ax.legend(); fig.tight_layout()
        path = output / name; fig.savefig(path); plt.close(fig); made.append(path)
    fig, ax = plt.subplots()
    ordered = sorted(results, key=lambda r: (r["phase"], r["prompt_tokens"], r["generated_tokens"], r["thread_count"]))
    labels = [f"{'pp' if r['phase']=='prompt_processing' else 'tg'} p{r['prompt_tokens']} n{r['generated_tokens']} t{r['thread_count']}" for r in ordered]
    ax.bar(range(len(ordered)), [100 * (r["coefficient_of_variation"] or 0) for r in ordered])
    ax.set(title="Coefficient of variation by workload", xlabel="Workload", ylabel="CV (%)")
    ax.set_xticks(range(len(labels)), labels, rotation=90); fig.tight_layout()
    path = output / "coefficient_of_variation.png"; fig.savefig(path); plt.close(fig); made.append(path)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path); parser.add_argument("--raw-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-filename", required=True)
    args = parser.parse_args()
    rows, warnings = load_normalized(args.csv); results = aggregate(rows); run = failures(args.raw_jsonl)
    metadata = {"model": args.model_filename}
    if args.raw_jsonl and args.raw_jsonl.exists():
        for line in args.raw_jsonl.read_text(encoding="utf-8").splitlines():
            try: metadata.update(json.loads(line).get("environment", {})); break
            except json.JSONDecodeError: continue
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {"schema_version": 1, "metadata": metadata, "run_summary": run, "warnings": warnings, "groups": results}
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    make_plots(results, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results, run, metadata, warnings), encoding="utf-8")
    print(f"analyzed {len(rows)} records into {len(results)} groups; {len(warnings)} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
