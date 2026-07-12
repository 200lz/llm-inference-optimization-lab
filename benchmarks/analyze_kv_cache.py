#!/usr/bin/env python3
"""Analyze Phase 6 context/KV-cache benchmark records without mismatched references."""
from __future__ import annotations

import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import yaml

KNOWN_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
KEYS = ("model_logical_id", "context_size", "prompt_tokens", "generated_tokens", "thread_count", "batch_size", "ubatch_size")
METRICS = ("prompt_tokens_per_second", "generation_tokens_per_second", "elapsed_seconds", "peak_rss_kib", "reported_kv_cache_bytes")
SLICE_A = {(512, 256, 32), (1024, 768, 32), (2048, 1536, 32), (4096, 3072, 32)}
SLICE_B = {(context, 256, 64) for context in (512, 1024, 2048, 4096)}
SLICE_C = {(4096, 512, generated) for generated in (16, 64, 128, 256)}

def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError): return None

def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings, rows, seen = [], [], set()
    with path.open(encoding="utf-8", newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        for index, row in enumerate(csv.DictReader(lines), 2):
            iid = row.get("invocation_id")
            if not iid or iid in seen:
                warnings.append(f"duplicate or missing invocation ID at CSV row {index}"); continue
            seen.add(iid)
            if row.get("kv_key_type") not in KNOWN_TYPES or row.get("kv_value_type") not in KNOWN_TYPES:
                warnings.append(f"unknown KV-cache type at CSV row {index}"); continue
            try:
                for key in ("context_size", "prompt_tokens", "generated_tokens", "thread_count", "batch_size", "ubatch_size"): row[key] = int(row[key])
            except (KeyError, ValueError): warnings.append(f"malformed workload at CSV row {index}"); continue
            for metric in METRICS: row[metric] = number(row.get(metric))
            rows.append(row)
    return rows, warnings

def stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None]
    if not clean: return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None, "sample_stdev": None, "coefficient_of_variation": None}
    mean = statistics.mean(clean); sd = statistics.stdev(clean) if len(clean) > 1 else None
    return {"count": len(clean), "mean": mean, "median": statistics.median(clean), "minimum": min(clean), "maximum": max(clean), "sample_stdev": sd, "coefficient_of_variation": sd / mean if sd is not None and mean else None}

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in KEYS) + (row["kv_key_type"], row["kv_value_type"])].append(row)
    result = []
    for key, items in grouped.items():
        entry = dict(zip(KEYS + ("kv_key_type", "kv_value_type"), key)); entry["kv_cache_id"] = f"kv-{key[-2]}-{key[-1]}"
        entry["metrics"] = {metric: stats(item[metric] for item in items) for metric in METRICS}; result.append(entry)
    references = {tuple(item[k] for k in KEYS): item for item in result if item["kv_key_type"] == item["kv_value_type"] == "f16"}
    for item in result:
        reference = references.get(tuple(item[k] for k in KEYS)); relative = {}
        for metric in METRICS:
            value = item["metrics"][metric]["mean"]; base = reference["metrics"][metric]["mean"] if reference else None
            ratio = value / base if value is not None and base not in (None, 0) else None
            relative[metric] = {"ratio_to_f16": ratio, "percent_change": (ratio - 1) * 100 if ratio is not None else None,
                                "reduction_percent": (1 - ratio) * 100 if ratio is not None else None}
        item["relative_to_f16"] = relative
    return sorted(result, key=lambda x: tuple(x[k] for k in KEYS) + (x["kv_cache_id"],))

def memory_growth(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for g in groups:
        if g["prompt_tokens"] == 256 and g["generated_tokens"] == 64:
            buckets[(g["model_logical_id"], g["kv_cache_id"])].append(g)
    for key, values in buckets.items():
        values.sort(key=lambda x: x["context_size"])
        for low, high in zip(values, values[1:]):
            for metric in ("peak_rss_kib", "reported_kv_cache_bytes"):
                a, b = low["metrics"][metric]["mean"], high["metrics"][metric]["mean"]
                output.append({"model": key[0], "kv_cache_id": key[1], "metric": metric, "from_context": low["context_size"], "to_context": high["context_size"],
                               "growth_per_context_token": (b-a)/(high["context_size"]-low["context_size"]) if a is not None and b is not None else None})
    return output

def select_slice(groups: list[dict[str, Any]], workload_slice: set[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """Select an explicit (context, prompt, generated) experiment slice."""
    return [g for g in groups if (g["context_size"], g["prompt_tokens"], g["generated_tokens"]) in workload_slice]

def plots(groups: list[dict[str, Any]], out: Path) -> list[str]:
    created: list[str] = []
    slices = (("actual_growth", SLICE_A, "actual prompt/context growth (generated=32)"),
              ("allocated_context", SLICE_B, "allocated-context scaling (prompt=256, generated=64)"),
              ("decode_length", SLICE_C, "decode-length scaling (context=4096, prompt=512)"))
    metrics = (("prompt_tokens_per_second", "Prompt processing (tokens/s)"),
               ("generation_tokens_per_second", "Generation (tokens/s)"),
               ("elapsed_seconds", "Invocation duration (s)"),
               ("peak_rss_kib", "Peak RSS (KiB)"),
               ("reported_kv_cache_bytes", "Reported KV allocation (bytes)"))
    for slug, workload_slice, title in slices:
        selected = select_slice(groups, workload_slice)
        x = "generated_tokens" if slug == "decode_length" else "context_size"
        xlabel = "Generated tokens" if slug == "decode_length" else "Context size"
        for metric, ylabel in metrics:
            series = {cache: sorted((g for g in selected if g["kv_cache_id"] == cache and g["metrics"][metric]["mean"] is not None), key=lambda g: g[x])
                      for cache in sorted({g["kv_cache_id"] for g in selected})}
            series = {cache: values for cache, values in series.items() if values}
            if not series:
                continue
            name = f"{slug}_{metric}.png"
            plt.figure()
            for cache, values in series.items():
                plt.plot([g[x] for g in values], [g["metrics"][metric]["mean"] for g in values], marker="o", label=cache)
            plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(out/name); plt.close(); created.append(name)
        # CV values remain ratios in analysis.json and are converted exactly once here.
        cv_metric = "generation_tokens_per_second"
        series = {cache: sorted((g for g in selected if g["kv_cache_id"] == cache and g["metrics"][cv_metric]["coefficient_of_variation"] is not None), key=lambda g: g[x])
                  for cache in sorted({g["kv_cache_id"] for g in selected})}
        series = {cache: values for cache, values in series.items() if values}
        if series:
            name = f"{slug}_generation_cv_percent.png"; plt.figure()
            for cache, values in series.items():
                plt.plot([g[x] for g in values], [100 * g["metrics"][cv_metric]["coefficient_of_variation"] for g in values], marker="o", label=cache)
            plt.title(title); plt.xlabel(xlabel); plt.ylabel("Generation throughput CV (%)"); plt.legend(); plt.tight_layout(); plt.savefig(out/name); plt.close(); created.append(name)
    return created

def raw_summary(path: Path) -> dict[str, Any]:
    records=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: continue
    latest={r.get("invocation_id"):r for r in records if r.get("invocation_id")}
    measured=sum(r.get("phase")=="measured" for r in latest.values()); warmups=sum(r.get("phase")=="warmup" for r in latest.values())
    return {"records":len(records),"unique_invocations":len(latest),"measured":measured,"warmups":warmups,
            "total_elapsed_seconds":sum(number(r.get("elapsed_seconds")) or 0 for r in latest.values()),
            "environment":records[0].get("environment",{}) if records else {},
            **{s:sum(r.get("status")==s for r in latest.values()) for s in ("success","failed","timed_out","parse_error")}}

def cv_cases(groups: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cases=[]
    for g in groups:
        for metric in METRICS:
            cv=g["metrics"][metric]["coefficient_of_variation"]
            if cv is not None: cases.append({"metric":metric,"group":g,"stats":g["metrics"][metric]})
    return min(cases,key=lambda c:c["stats"]["coefficient_of_variation"]), max(cases,key=lambda c:c["stats"]["coefficient_of_variation"])

def render(groups: list[dict[str, Any]], summary: dict[str,Any], config: dict[str,Any], plots_made: list[str]) -> str:
    env=summary.get("environment",{}); model=config["models"][0]; low_cv,high_cv=cv_cases(groups)
    all_values=lambda metric,field: [g["metrics"][metric][field] for g in groups if g["metrics"][metric][field] is not None]
    cv_values=[g["metrics"][m]["coefficient_of_variation"] for g in groups for m in METRICS if g["metrics"][m]["coefficient_of_variation"] is not None]
    def endpoints(workload_slice: set[tuple[int,int,int]], x: str) -> tuple[dict[str,Any],dict[str,Any]]:
        values=sorted((g for g in select_slice(groups,workload_slice) if g["kv_cache_id"]=="kv-f16-f16"),key=lambda g:g[x])
        return values[0],values[-1]
    a0,a1=endpoints(SLICE_A,"context_size"); b0,b1=endpoints(SLICE_B,"context_size"); c0,c1=endpoints(SLICE_C,"generated_tokens")
    change=lambda lo,hi,metric: f"{lo['metrics'][metric]['mean']:.2f}→{hi['metrics'][metric]['mean']:.2f}"
    lines=["# KV-cache and context-length scaling", "", "## Status", "", f"- **Measured observation:** Completed: {summary['unique_invocations']} unique invocations, comprising {summary['warmups']} warm-ups and {summary['measured']} measured repetitions. Successes: {summary['success']}; failures, timeouts, and parse errors: {summary['failed']}, {summary['timed_out']}, and {summary['parse_error']}.",
           f"- **Measured observation:** Total durable benchmark elapsed time was {summary['total_elapsed_seconds']:.2f} seconds.", "", "## Environment", "",
           f"- **Verified conclusion:** {env.get('cpu_model','N/A')}; Linux `{env.get('kernel','N/A')}`.",
           f"- **Verified conclusion:** llama.cpp commit `{env.get('llama_cpp_git_commit','N/A')}`; model `{Path(model['path']).name}`, SHA-256 `{model['sha256']}`, {model['byte_size']:,} bytes.",
           f"- **Verified conclusion:** CPU only, threads {config['threads'][0]}, batch/ubatch {config['batch_sizes'][0]}/{config['ubatch_sizes'][0]}, mmap enabled, and {config['timeout_seconds']}-second timeout.",
           "", "## KV-cache support", "", f"- **Verified conclusion:** Supported key/value types: {', '.join(f'`{x}`' for x in sorted(KNOWN_TYPES))}.",
           "- **Verified conclusion:** Tested symmetric `f16/f16`, `q8_0/q8_0`, and `q4_0/q4_0`. Other supported and asymmetric combinations were omitted; no unsupported type was substituted.",
           "- **Verified conclusion:** Unified KV was not varied because it is not exposed by the pinned `llama-bench`.",
           "", "## Experiment matrix", "", "- **Verified conclusion:** Actual prompt/context growth: 512/256, 1024/768, 2048/1536, and 4096/3072, all generating 32 tokens.",
           "- **Verified conclusion:** Fixed-prompt allocated-context scaling: prompt 256, generated 64, contexts 512/1024/2048/4096.",
           "- **Verified conclusion:** Decode-length scaling: context 4096, prompt 512, generated 16/64/128/256.",
           f"- **Verified conclusion:** {len(groups)} exact workload/KV groups, with {config['repetitions']} measured repetitions per group.",
           "", "## Performance and variability summary", "",
           f"- **Measured observation:** CV is sample standard deviation divided by mean and is stored as a ratio in `analysis.json`; reader-facing values are percentages. Group-metric CV ranged from {100*min(cv_values):.6f}% to {100*max(cv_values):.6f}%.",
           "- **Verified conclusion:** F16-relative comparisons require exact model, context, prompt, generated tokens, threads, batch, and ubatch matching.",
           "- **Verified conclusion:** These measurements do not establish whether a workload is compute-bound or memory-bound.",
           "", "## Memory results", "", f"- **Measured observation:** Group mean peak RSS ranged from {min(all_values('peak_rss_kib','mean')):,.0f} to {max(all_values('peak_rss_kib','mean')):,.0f} KiB.",
           "- **Measured observation:** `reported_kv_cache_bytes` is N/A because the benchmark output did not provide it; it is not inferred from RSS.",
           "- **Verified conclusion:** Peak RSS includes mmap-backed, shared, and page-cache effects and is never an exact KV-cache allocation.",
           "", "## Relative F16 comparisons", "", "- **Verified conclusion:** Every relative value in `analysis.json` is joined on the full workload key; missing exact F16 references remain N/A.",
           "", "## Key observations", "", "- **Measured observation:** The three workload families are reported and plotted separately; their points are never connected across slices.",
           f"- **Measured observation:** In the actual-growth F16 slice, context/prompt 512/256→4096/3072 changed PP {change(a0,a1,'prompt_tokens_per_second')} tokens/s, TG {change(a0,a1,'generation_tokens_per_second')} tokens/s, duration {change(a0,a1,'elapsed_seconds')} s, and peak RSS {change(a0,a1,'peak_rss_kib')} KiB.",
           f"- **Measured observation:** In the fixed-prompt F16 slice, context 512→4096 changed PP {change(b0,b1,'prompt_tokens_per_second')} tokens/s, TG {change(b0,b1,'generation_tokens_per_second')} tokens/s, duration {change(b0,b1,'elapsed_seconds')} s, and peak RSS {change(b0,b1,'peak_rss_kib')} KiB.",
           f"- **Measured observation:** In the decode-length F16 slice, generated tokens 16→256 changed TG {change(c0,c1,'generation_tokens_per_second')} tokens/s and duration {change(c0,c1,'elapsed_seconds')} s.",
           "- **Hypothesis:** Cache conversion, locality, and memory traffic may contribute to changes; profiling is required to distinguish causes.",
           "", "## Correctness comparison", "", "- **Verified conclusion:** No descriptive output comparison was executed, and no semantic-equivalence claim is made.",
           "", "## CV extremes", ""]
    for label,case in (("Minimum",low_cv),("Maximum",high_cv)):
        g,s=case["group"],case["stats"]
        lines.append(f"- **Measured observation:** {label} CV: metric `{case['metric']}`, `{g['kv_cache_id']}`, context {g['context_size']}, prompt {g['prompt_tokens']}, generated {g['generated_tokens']}, threads {g['thread_count']}; mean {s['mean']:.12g}, sample standard deviation {s['sample_stdev']:.12g}, CV ratio {s['coefficient_of_variation']:.12g}, CV {100*s['coefficient_of_variation']:.9f}%.")
    lines += ["", "## Results", "", "| Context | Prompt | Generated | KV cache | PP mean tok/s | TG mean tok/s | Duration mean s | Peak RSS mean KiB | KV bytes mean |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    fmt=lambda v:"N/A" if v is None else f"{v:.3f}"
    for g in groups: lines.append(f"| {g['context_size']} | {g['prompt_tokens']} | {g['generated_tokens']} | {g['kv_cache_id']} | "+" | ".join(fmt(g["metrics"][m]["mean"]) for m in METRICS)+" |")
    lines += ["", "## Plots", ""] + [f"- ![{name}]({Path('../../benchmarks/results/kv_cache_context_scaling') / name})" for name in plots_made]
    lines += ["", "## Limitations", "", "- **Verified conclusion:** Results cover one WSL2 CPU host, one 0.5B model, synthetic workloads, five repetitions, uncontrolled affinity/frequency/temperature, and a limited context range; they do not generalize to GPUs, larger models, or continuous batching.", "- **Verified conclusion:** No semantic-equivalence test was run, so no correctness-equivalence claim is made.", ""]
    return "\n".join(lines)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("csv",type=Path); p.add_argument("--raw-jsonl",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--model-metadata",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    rows,warnings=load_rows(a.csv); groups=aggregate(rows); summary=raw_summary(a.raw_jsonl)
    config=yaml.safe_load(a.config.read_text()); yaml.safe_load(a.model_metadata.read_text())
    a.output_dir.mkdir(parents=True,exist_ok=True); made=plots(groups,a.output_dir)
    (a.output_dir/"analysis.json").write_text(json.dumps({"schema_version":1,"run_summary":summary,"warnings":warnings,"groups":groups,"memory_growth":memory_growth(groups),"plots":made},indent=2)+"\n")
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(render(groups,summary,config,made),encoding="utf-8"); return 0
if __name__ == "__main__": raise SystemExit(main())
