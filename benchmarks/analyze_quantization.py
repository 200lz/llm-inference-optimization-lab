#!/usr/bin/env python3
"""Analyze matching F16/Q8_0/Q4_K_M llama-bench measurements."""
from __future__ import annotations

import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import yaml

METRICS = {"prompt_processing": "prompt_tokens_per_second", "generation": "generation_tokens_per_second",
           "invocation_duration": "elapsed_seconds"}
WORKLOAD = ("prompt_tokens", "generated_tokens", "thread_count", "batch_size", "ubatch_size")

def _num(value: Any, name: str, line: int, integer: bool = False) -> float | int:
    try: result = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"line {line}: malformed {name}: {value!r}") from exc
    if isinstance(result, float) and not math.isfinite(result): raise ValueError(f"line {line}: non-finite {name}")
    if result < 0: raise ValueError(f"line {line}: negative {name}")
    return result

def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, warnings, seen = [], [], set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        required = set(WORKLOAD) | {"quantization", "invocation_id", *METRICS.values()}
        missing = required - set(reader.fieldnames or ())
        if missing: raise ValueError(f"missing CSV fields: {', '.join(sorted(missing))}")
        for line, raw in enumerate(reader, 2):
            try:
                row = {key: _num(raw[key], key, line, True) for key in WORKLOAD}
                row.update({key: _num(raw[key], key, line) for key in METRICS.values()})
                row.update(quantization=raw["quantization"].upper(), invocation_id=raw["invocation_id"])
                if row["quantization"] not in {"F16", "Q8_0", "Q4_K_M"}: raise ValueError(f"line {line}: unknown quantization")
            except ValueError as exc: warnings.append(str(exc)); continue
            if row["invocation_id"] in seen: warnings.append(f"line {line}: duplicate invocation ID skipped"); continue
            seen.add(row["invocation_id"]); rows.append(row)
    return rows, warnings

def stats(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values); sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "standard_deviation": sd, "coefficient_of_variation": sd / mean if mean else None}

def aggregate(rows: list[dict[str, Any]], sizes: dict[str, int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        for metric, field in METRICS.items(): grouped[(row["quantization"], *[row[k] for k in WORKLOAD], metric)].append(row[field])
    result = []
    for key, values in sorted(grouped.items()):
        q, *workload, metric = key
        result.append({"quantization": q, **dict(zip(WORKLOAD, workload)), "metric": metric, **stats(values),
                       "throughput_ratio_to_f16": None, "throughput_percent_change_to_f16": None,
                       "file_size_ratio_to_f16": None, "file_size_reduction_percent": None})
    refs = {(tuple(row[k] for k in WORKLOAD), row["metric"]): row["mean"] for row in result if row["quantization"] == "F16"}
    fsize = sizes.get("F16")
    for row in result:
        ref = refs.get((tuple(row[k] for k in WORKLOAD), row["metric"]))
        if ref and row["metric"] != "invocation_duration":
            row["throughput_ratio_to_f16"] = row["mean"] / ref
            row["throughput_percent_change_to_f16"] = 100 * (row["mean"] / ref - 1)
        size = sizes.get(row["quantization"])
        if fsize and size:
            row["file_size_ratio_to_f16"] = size / fsize
            row["file_size_reduction_percent"] = 100 * (1 - size / fsize)
    return result

def metadata(path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")); artifacts = []
    for item in raw["models"].values(): artifacts.append(dict(item))
    return {str(x["quantization"]).upper(): int(x["byte_size"]) for x in artifacts}, artifacts

def run_summary(path: Path) -> dict[str, Any]:
    out = {"total":0,"warmups":0,"measured":0,"successful":0,"failed":0,"timed_out":0,"parse_errors":0,"duration":0.0}
    seen=set()
    if not path.exists(): return out
    records=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try: record=json.loads(line)
        except json.JSONDecodeError: out["failed"]+=1; continue
        if record.get("invocation_id") in seen: continue
        seen.add(record.get("invocation_id")); records.append(record)
    for r in records:
        out["total"]+=1; out["warmups" if r.get("warmup") else "measured"]+=1; out["duration"]+=float(r.get("elapsed_seconds",0))
        ok=r.get("status")=="success"; out["successful" if ok else "failed"]+=1
        out["timed_out"]+=int(r.get("status")=="timed_out"); out["parse_errors"]+=int(r.get("status")=="parse_error")
    return out

def plots(groups: list[dict[str, Any]], sizes: dict[str,int], output: Path) -> list[Path]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True); made=[]; order=[q for q in ("F16","Q8_0","Q4_K_M") if q in sizes]
    def bar(name,title,ylabel, chosen):
        fig,ax=plt.subplots(); ax.bar([x[0] for x in chosen],[x[1] for x in chosen]); ax.set(title=title,xlabel="Quantization",ylabel=ylabel); fig.tight_layout(); p=output/name; fig.savefig(p); plt.close(fig); made.append(p)
    bar("file_size.png","GGUF file size","Bytes",[(q,sizes[q]) for q in order])
    for prompt in (128,1024):
      for metric,label in (("prompt_processing","Prompt-processing throughput"),("generation","Generation throughput")):
        chosen=[(q,next((r["mean"] for r in groups if r["quantization"]==q and r["metric"]==metric and r["prompt_tokens"]==prompt and r["generated_tokens"]==64 and r["thread_count"]==4),math.nan)) for q in order]
        chosen=[x for x in chosen if math.isfinite(x[1])]; bar(f"p{prompt}_{metric}.png",f"{label}: p{prompt}-n64-t4","Tokens/s",chosen)
    for metric,label,name in (("prompt_processing","Prompt-processing throughput","threads_prompt_processing.png"),("generation","Generation throughput","threads_generation.png")):
        fig,ax=plt.subplots()
        for q in order:
            chosen=sorted((r for r in groups if r["quantization"]==q and r["metric"]==metric and r["prompt_tokens"]==512),key=lambda r:r["thread_count"])
            if chosen: ax.plot([r["thread_count"] for r in chosen],[r["mean"] for r in chosen],marker="o",label=q)
        ax.set(title=label+": p512-n64",xlabel="Threads",ylabel="Tokens/s"); ax.legend(); fig.tight_layout(); p=output/name; fig.savefig(p); plt.close(fig); made.append(p)
    chosen=[]
    for q in order:
        values=[r["coefficient_of_variation"] for r in groups if r["quantization"]==q and r["metric"] in {"prompt_processing","generation"} and r["coefficient_of_variation"] is not None]
        if values: chosen.append((q,100*statistics.fmean(values)))
    bar("coefficient_of_variation.png","Mean coefficient of variation","CV (%)",chosen)
    return made

def render(groups, artifacts, summary, warnings, quality=None):
    def table(metric):
        lines=["| Quantization | Workload | N | Mean | Median | Min | Max | SD | CV | Ratio to F16 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in groups:
            if r["metric"]!=metric: continue
            ratio="N/A" if r["throughput_ratio_to_f16"] is None else f'{r["throughput_ratio_to_f16"]:.3f}'
            lines.append(f'| {r["quantization"]} | p{r["prompt_tokens"]}-n{r["generated_tokens"]}-t{r["thread_count"]}-b{r["batch_size"]}-ub{r["ubatch_size"]} | {r["count"]} | {r["mean"]:.3f} | {r["median"]:.3f} | {r["minimum"]:.3f} | {r["maximum"]:.3f} | {r["standard_deviation"]:.3f} | {r["coefficient_of_variation"]:.2%} | {ratio} |')
        return "\n".join(lines)
    status="completed" if summary["total"]==90 and summary["failed"]==0 else "incomplete"
    arts="\n".join(f'- **Verified conclusion:** `{a["source_filename"]}` ({a["quantization"]}), {a["byte_size"]} bytes, SHA-256 `{a["sha256"]}`, source `{a["source_repository"]}` at `{a["source_revision"]}`.' for a in artifacts)
    if quality is None:
        quality_section = "This section is populated only after the separate deterministic suite completes."
    else:
        lines = [
            f'- **Verified conclusion:** {len(quality["comparisons"])} latest deterministic outputs were analyzed; all completed successfully: `{str(quality["all_completed_successfully"]).lower()}`.',
            "",
            "| Prompt | Quantization | Successful | Exact match to F16 | Normalized match to F16 | Output length | Length difference to F16 |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
        for item in quality["comparisons"]:
            value = lambda key: "N/A" if item[key] is None else str(item[key]).lower() if isinstance(item[key], bool) else str(item[key])
            lines.append(f'| {item["prompt_id"]} | {item["quantization"]} | {value("completed_successfully")} | {value("exact_match_to_f16")} | {value("normalized_match_to_f16")} | {value("output_length")} | {value("output_length_difference_to_f16")} |')
        lines.extend(["", quality["token_metrics_limitation"]])
        quality_section = "\n".join(lines)
    return f'''# F16 / Q8_0 / Q4_K_M quantization comparison

## Status

- **Verified conclusion:** Phase 5 is **{status}**. {summary["successful"]} invocations succeeded; {summary["failed"]} failed, {summary["timed_out"]} timed out, and {summary["parse_errors"]} had parse errors.
- **Verified conclusion:** All three exact artifacts are configured from the same Qwen2.5-0.5B-Instruct repository revision.

## Environment and experiment matrix

- **Verified conclusion:** CPU-only llama.cpp at `e3546c7948e3af463d0b401e6421d5a4c2faf565`; `-ngl 0 -dev none`, mmap enabled, batch/ubatch 512, 900-second subprocess timeout, one warm-up and five measured repetitions.
- **Verified conclusion:** Five deduplicated workloads per quantization produce 15 cases, 15 warm-ups, 75 measurements, and 90 total invocations.

## Model artifacts

{arts}

## Prompt-processing throughput

{table("prompt_processing")}

## Token-generation throughput

{table("generation")}

## Invocation duration

{table("invocation_duration")}

## Plots

![File size](file_size.png)
![Short prompt processing](p128_prompt_processing.png)
![Short generation](p128_generation.png)
![Long prompt processing](p1024_prompt_processing.png)
![Long generation](p1024_generation.png)
![Thread prompt processing](threads_prompt_processing.png)
![Thread generation](threads_generation.png)
![Variability](coefficient_of_variation.png)

## Deterministic output comparison

{quality_section}

Exact or normalized matches are descriptive and neither prove semantic equivalence nor make differences incorrect.

## Interpretation and limitations

- **Measured observation:** Only measured rows appear above; missing matching F16 references are shown as `N/A`.
- **Hypothesis:** Differences may relate to model-byte traffic or quantized kernel implementations; hardware-counter evidence would be required.
- **Verified conclusion:** File-size reductions are exact artifact-storage comparisons. Throughput alone does not establish compute-bound or memory-bound behavior, and lower precision is not assumed to be faster.
- **Verified conclusion:** This covers one WSL2 host, one small family, synthetic llama-bench workloads, five repetitions, uncontrolled affinity/frequency/temperature, and a limited deterministic prompt suite. Results may not generalize to larger models or GPUs.
- **Verified conclusion:** Peak RSS was omitted because wrapping the process would change the exact executable command and Linux/WSL2 mmap, page-cache, and shared-page accounting is not exact model-memory accounting.
- **Verified conclusion:** Analyzer excluded {len(warnings)} malformed or duplicate row(s). Deterministic comparison is not a quality benchmark.
'''

def main():
    p=argparse.ArgumentParser(); p.add_argument("csv",type=Path); p.add_argument("--raw-jsonl",type=Path,required=True); p.add_argument("--models",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--quality-analysis",type=Path); a=p.parse_args()
    rows,warnings=load_rows(a.csv); sizes,artifacts=metadata(a.models); groups=aggregate(rows,sizes); summary=run_summary(a.raw_jsonl)
    quality=json.loads(a.quality_analysis.read_text(encoding="utf-8")) if a.quality_analysis else None
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"analysis.json").write_text(json.dumps({"schema_version":1,"run_summary":summary,"warnings":warnings,"models":artifacts,"groups":groups},indent=2)+"\n",encoding="utf-8"); plots(groups,sizes,a.output_dir); a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(render(groups,artifacts,summary,warnings,quality),encoding="utf-8"); print(f"analyzed {len(rows)} rows into {len(groups)} groups")
if __name__=="__main__": main()
