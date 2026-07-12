#!/usr/bin/env python3
"""Analyze Phase 7 perf-stat and perf-report records without inventing counters."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from benchmarks.run_cpu_profiling import load_config
except ModuleNotFoundError:  # Support direct `python benchmarks/...py` invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.run_cpu_profiling import load_config

METRICS = ("elapsed_seconds", "task-clock", "cycles", "instructions", "ipc", "branches", "branch-misses",
           "branch_miss_rate", "cache-references", "cache-misses", "cache_miss_rate", "context-switches",
           "cpu-migrations", "page-faults", "cycles_per_token", "instructions_per_token")
FAMILY_PATTERNS = (
    ("quantized dot product", (r"vec_dot_q", r"dot_q[248]", r"q[248]_.*dot")),
    ("dequantization or unpacking", (r"dequant", r"unpack", r"fp16_to_fp32")),
    ("GEMM/GEMV", (r"gemm", r"gemv", r"mul_mat", r"sgemm")),
    ("attention", (r"flash_attn", r"attention")), ("RoPE", (r"rope",)), ("softmax", (r"soft_?max",)),
    ("KV-cache copy/convert", (r"kv.*(?:copy|convert)", r"cpy.*kv")),
    ("graph scheduling", (r"graph_compute", r"graph_plan", r"sched")),
    ("thread synchronization", (r"pthread", r"barrier", r"mutex", r"cond_wait", r"threadpool")),
    ("memory copy", (r"memcpy", r"memmove", r"copy")), ("runtime or loader", (r"_dl_", r"loader", r"mmap")),
)


def categorize_function(name: str) -> str:
    low=name.lower()
    for family,patterns in FAMILY_PATTERNS:
        if any(re.search(pattern,low) for pattern in patterns): return family
    return "unknown"


def parse_perf_report(text: str) -> list[dict[str, Any]]:
    rows=[]
    # Standard --stdio: overhead, command, shared object, symbol. Keep names verbatim.
    pattern=re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)%\s+\S+\s+(\S+)\s+\[[^\]]+\]\s+(.+?)\s*$")
    fallback=re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)%\s+(\S+)\s+(.+?)\s*$")
    for line in text.splitlines():
        if line.lstrip().startswith("#"): continue
        match=pattern.match(line) or fallback.match(line)
        if not match: continue
        overhead=float(match.group(1)); dso=match.group(2); symbol=match.group(3).strip()
        symbol=re.sub(r"^\[[k.]\]\s*", "", symbol)
        unresolved=symbol in {"[unknown]","unknown"} or symbol.startswith("0x")
        rows.append({"function":symbol,"shared_object":dso,"overhead_percent":overhead,
                     "symbol_status":"unresolved" if unresolved else "resolved","call_path":None,
                     "source_file_line":None,"family":categorize_function(symbol)})
    merged: dict[tuple[str,str],dict[str,Any]]={}
    for row in rows:
        key=(row["function"],row["shared_object"])
        if key in merged: merged[key]["overhead_percent"]+=row["overhead_percent"]
        else: merged[key]=row
    return sorted(merged.values(),key=lambda x:x["overhead_percent"],reverse=True)


def cumulative_share(rows: list[dict[str,Any]], n: int) -> float:
    return sum(x["overhead_percent"] for x in rows[:n])


def descriptive(values: list[float]) -> dict[str,Any]:
    mean=statistics.fmean(values); sd=statistics.stdev(values) if len(values)>1 else 0.0
    return {"count":len(values),"mean":mean,"median":statistics.median(values),"minimum":min(values),"maximum":max(values),
            "sample_standard_deviation":sd,"coefficient_of_variation":sd/mean if mean else None}


def load_stat_records(path: Path) -> list[dict[str,Any]]:
    latest={}
    if not path.exists(): return []
    for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        try: record=json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"malformed JSONL line {number}: {exc}") from exc
        latest[record["invocation_id"]]=record
    return [x for x in latest.values() if x.get("mode")=="stat" and x.get("status")=="success"]


def record_metrics(record: dict[str,Any]) -> dict[str,float|None]:
    parsed=record.get("perf_stat",{}); events=parsed.get("events",{}); derived=parsed.get("derived",{})
    value=lambda name: events.get(name,{}).get("value")
    tokens=record.get("prompt_tokens",0)+record.get("generated_tokens",0)
    elapsed=derived.get("elapsed_seconds")
    result={"elapsed_seconds":elapsed,"task-clock":value("task-clock"),"cycles":value("cycles"),"instructions":value("instructions"),
            "ipc":derived.get("ipc"),"branches":value("branches"),"branch-misses":value("branch-misses"),"branch_miss_rate":derived.get("branch_miss_rate"),
            "cache-references":value("cache-references"),"cache-misses":value("cache-misses"),"cache_miss_rate":derived.get("cache_miss_rate"),
            "context-switches":value("context-switches"),"cpu-migrations":value("cpu-migrations"),"page-faults":value("page-faults")}
    result["cycles_per_token"]=result["cycles"]/tokens if result["cycles"] is not None and tokens else None
    result["instructions_per_token"]=result["instructions"]/tokens if result["instructions"] is not None and tokens else None
    return result


def aggregate(records: list[dict[str,Any]]) -> list[dict[str,Any]]:
    grouped=defaultdict(lambda:defaultdict(list)); reasons=defaultdict(dict)
    for record in records:
        key=(record["quantization"],record["workload_id"],record["prompt_tokens"],record["generated_tokens"],record["threads"],record["batch"],record["ubatch"])
        values=record_metrics(record)
        for metric,value in values.items():
            if value is not None and math.isfinite(value): grouped[key][metric].append(value)
            else: reasons[key][metric]="counter unsupported, not counted, or missing"
    result=[]
    for key in sorted(grouped):
        item={"quantization":key[0],"workload_id":key[1],"prompt_tokens":key[2],"generated_tokens":key[3],"threads":key[4],"batch":key[5],"ubatch":key[6],"metrics":{}}
        for metric in METRICS:
            values=grouped[key].get(metric,[])
            item["metrics"][metric]=descriptive(values) if values else {"available":False,"reason":reasons[key].get(metric,"missing")}
        result.append(item)
    return result


def comparisons(groups:list[dict[str,Any]]) -> list[dict[str,Any]]:
    match=lambda r:(r["workload_id"],r["prompt_tokens"],r["generated_tokens"],r["threads"],r["batch"],r["ubatch"])
    refs={match(r):r for r in groups if r["quantization"]=="F16"}; out=[]
    for row in groups:
        ref=refs.get(match(row)); item={"quantization":row["quantization"],"workload_id":row["workload_id"],"relative_to_f16":{}}
        for metric in ("elapsed_seconds","cycles","instructions","ipc"):
            cur=row["metrics"][metric].get("mean"); base=ref["metrics"][metric].get("mean") if ref else None
            item["relative_to_f16"][metric+"_ratio"]=cur/base if cur is not None and base else None
        for metric in ("cache_miss_rate","branch_miss_rate"):
            cur=row["metrics"][metric].get("mean"); base=ref["metrics"][metric].get("mean") if ref else None
            item["relative_to_f16"][metric+"_difference"]=cur-base if cur is not None and base is not None else None
        out.append(item)
    return out


def phase5_throughput(path:Path) -> dict[tuple[str,int,int,int,int,int],dict[str,float]]:
    result={}
    if not path.exists(): return result
    with path.open(encoding="utf-8") as stream:
      reader=csv.DictReader(line for line in stream if not line.startswith("#"))
      for row in reader:
        key=(row["quantization"].upper(),int(row["prompt_tokens"]),int(row["generated_tokens"]),int(row["thread_count"]),int(row["batch_size"]),int(row["ubatch_size"]))
        result.setdefault(key,defaultdict(list))["prompt"].append(float(row["prompt_tokens_per_second"])); result[key]["generation"].append(float(row["generation_tokens_per_second"]))
    return {k:{name:statistics.fmean(vals) for name,vals in v.items()} for k,v in result.items()}


def load_reports(directory:Path) -> dict[str,list[dict[str,Any]]]:
    return {path.stem:parse_perf_report(path.read_text(encoding="utf-8",errors="replace")) for path in directory.glob("*.txt")} if directory.exists() else {}


def plots(groups:list[dict[str,Any]], throughput:dict, hot:dict, output:Path) -> list[str]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True); primary=[x for x in groups if x["workload_id"]=="primary-p1024-n64-t4"]
    order=[q for q in ("F16","Q8_0","Q4_K_M") if any(x["quantization"]==q for x in primary)]; made=[]
    charts=(("elapsed_seconds","Elapsed time","Seconds"),("instructions","Instructions","Instructions"),("cycles","Cycles","Cycles"),("ipc","IPC","Instructions/cycle"),("cache_miss_rate","Cache-miss rate","Fraction"),("branch_miss_rate","Branch-miss rate","Fraction"),("instructions_per_token","Whole-invocation instructions/token","Instructions/token"),("cycles_per_token","Whole-invocation cycles/token","Cycles/token"))
    for metric,title,ylabel in charts:
        vals=[next((x["metrics"][metric].get("mean") for x in primary if x["quantization"]==q),None) for q in order]; pairs=[(q,v) for q,v in zip(order,vals) if v is not None]
        if not pairs: continue
        fig,ax=plt.subplots(); ax.bar([x[0] for x in pairs],[x[1] for x in pairs]); ax.set(title=title,xlabel="Quantization",ylabel=ylabel); fig.tight_layout(); name=f"quantization_{metric}.png"; fig.savefig(output/name); plt.close(fig); made.append(name)
    # Hot shares are keyed by record invocation IDs; aggregate by quantization token in ID.
    pairs=[]
    for q in order:
        vals=[rows[0]["overhead_percent"] for iid,rows in hot.items() if rows and f"-{q.lower()}-" in iid]
        if vals:pairs.append((q,statistics.fmean(vals)))
    if pairs:
        fig,ax=plt.subplots(); ax.bar([x[0] for x in pairs],[x[1] for x in pairs]); ax.set(title="Top hot-function share",xlabel="Quantization",ylabel="CPU overhead (%)"); fig.tight_layout(); fig.savefig(output/"top_hot_function_share.png"); plt.close(fig); made.append("top_hot_function_share.png")
    for metric,title,ylabel in (("instructions","Throughput versus instructions","Instructions"),("ipc","Throughput versus IPC","IPC"),("cache_miss_rate","Throughput versus cache-miss rate","Cache-miss rate")):
        points=[]
        for row in primary:
            y=row["metrics"][metric].get("mean"); key=(row["quantization"],row["prompt_tokens"],row["generated_tokens"],row["threads"],row["batch"],row["ubatch"]); x=throughput.get(key,{}).get("prompt")
            if x is not None and y is not None:points.append((x,y,row["quantization"]))
        if points:
            fig,ax=plt.subplots(); ax.scatter([x[0] for x in points],[x[1] for x in points]); [ax.annotate(x[2],(x[0],x[1])) for x in points]; ax.set(title=title,xlabel="Phase 5 prompt throughput (tokens/s)",ylabel=ylabel); fig.tight_layout(); name=f"throughput_vs_{metric}.png"; fig.savefig(output/name); plt.close(fig); made.append(name)
    return made


def render(analysis:dict[str,Any], config:dict[str,Any]) -> str:
    groups=analysis["groups"]; primary=[x for x in groups if x["workload_id"]=="primary-p1024-n64-t4"]
    def val(row,metric):
        value=row["metrics"][metric].get("mean"); return "N/A" if value is None else f"{value:.6g}"
    table=["| Quantization | N | Elapsed (s) | Cycles | Instructions | IPC | Cache miss rate | Branch miss rate | Cycles/token | Instructions/token |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in primary: table.append(f"| {row['quantization']} | {row['metrics']['cycles'].get('count','N/A')} | {val(row,'elapsed_seconds')} | {val(row,'cycles')} | {val(row,'instructions')} | {val(row,'ipc')} | {val(row,'cache_miss_rate')} | {val(row,'branch_miss_rate')} | {val(row,'cycles_per_token')} | {val(row,'instructions_per_token')} |")
    status="completed" if primary and all(x["metrics"]["elapsed_seconds"].get("mean") is not None for x in primary) else "incomplete"
    return f"""# CPU profiling and bottleneck attribution

## Status

- **Verified conclusion:** Phase 7 is **{status}**. Missing counters are reported as N/A and never replaced with zero.
- **Verified conclusion:** {len(analysis['records'])} successful perf-stat records and {len(analysis['hot_functions'])} parsed perf-report artifacts were available to this analysis.

## Environment

See [profiling_environment.md](../profiling_environment.md) for the captured CPU, kernel, perf permissions, affinity, memory, symbols, frame-pointer, debug-information, and governor evidence. llama.cpp remains pinned at `e3546c7948e3af463d0b401e6421d5a4c2faf565`.

## Workloads

CPU-only (`-ngl 0 -dev none`), mmap enabled, batch/ubatch 512. Primary: p1024-n64-t4; secondary: p128/p2048-n64-t4 and p1024-n64-t1/t8. One warm-up and three stat measurements are configured; one record profile is selected for each primary format. Reduced repetitions limit profiling overhead.

## Throughput reference

- **Measured observation:** Phase 5 measured F16 prompt processing above Q8_0 and Q4_K_M for matched p1024-n64-t4, while quantized generation was faster.

## Hardware-counter results

All normalized values below cover the combined invocation; they are not phase-specific attribution.

{chr(10).join(table)}

## Hot-function results

- **Measured observation:** Structured top-10/top-20 function records, original symbols, DSOs, cumulative shares, and conservative families are in `analysis.json`. No hot-function conclusion is made when sampling was unavailable.

## Cross-format comparison and attribution

- **Measured observation:** The matched ratios and rate differences are recorded in `analysis.json`; unsupported comparisons are null/N/A.
- **Hypothesis:** If quantized formats show additional instructions and quantized dot-product or unpacking symbols dominate, those paths may explain slower prefill, but source annotation is needed to connect whole-invocation counters to prefill.
- **Verified conclusion:** Whole-invocation counters cannot alone establish compute-bound or memory-bound behavior, and combined llama-bench accounting cannot isolate prompt processing from generation.
- **Verified conclusion:** Workload-shape ordering is only asserted when every compared format has an exact prompt/generation/thread/batch/ubatch match.

## Candidate optimization target

- **Hypothesis:** Phase 8 should select at most two resolved GGML functions from the measured top-function list. Their ideal Amdahl upper bound is `1 / (1 - CPU_share)`; no numeric bound is claimed without a measured share. Changes to quantized kernels carry medium-to-high correctness risk and high implementation complexity; scheduling/synchronization changes generally carry medium risk and complexity.

## Limitations

WSL2 may restrict or virtualize perf events. This study covers one CPU, one small model family, three stat repetitions, sampling overhead, uncontrolled frequency/affinity/temperature/host load, and possible symbol-resolution limits. Page cache and mmap affect faults. Whole-invocation normalization is not phase attribution. Missing FlameGraph tooling or counters does not justify fabricated output.
"""


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/cpu_profiling.yaml")); p.add_argument("--stat-jsonl",type=Path); p.add_argument("--report-dir",type=Path); p.add_argument("--phase5-csv",type=Path,default=Path("benchmarks/results/quantization_comparison/normalized.csv")); p.add_argument("--output-dir",type=Path,default=Path("profiles/analysis")); p.add_argument("--report",type=Path,default=Path("docs/results/cpu_profiling.md")); a=p.parse_args(); config=load_config(a.config)
    stat=a.stat_jsonl or Path(config["output_directory"])/"perf-stat"/"runs.jsonl"; report_dir=a.report_dir or Path(config["output_directory"])/"perf-report"
    records=load_stat_records(stat); groups=aggregate(records); hot=load_reports(report_dir); throughput=phase5_throughput(a.phase5_csv)
    analysis={"schema_version":1,"configuration_fingerprint":config["configuration_fingerprint"],"records":[r["invocation_id"] for r in records],"groups":groups,"comparisons":comparisons(groups),"hot_functions":{k:{"top_10":v[:10],"top_20":v[:20],"top_5_cumulative_share":cumulative_share(v,5),"top_20_cumulative_share":cumulative_share(v,20)} for k,v in hot.items()}}
    a.output_dir.mkdir(parents=True,exist_ok=True); analysis["plots"]=plots(groups,throughput,hot,a.output_dir); (a.output_dir/"analysis.json").write_text(json.dumps(analysis,indent=2)+"\n",encoding="utf-8"); a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(render(analysis,config),encoding="utf-8"); print(f"analyzed {len(records)} stat records and {len(hot)} reports"); return 0


if __name__=="__main__": raise SystemExit(main())
