#!/usr/bin/env python3
"""Durable CPU profiling runner for matched llama-bench workloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

UNAVAILABLE = {"<not supported>", "<not counted>", "not supported", "not counted"}


@dataclass(frozen=True)
class Model:
    logical_id: str
    path: Path
    quantization: str


@dataclass(frozen=True)
class Workload:
    workload_id: str
    prompt_tokens: int
    generated_tokens: int
    threads: int


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("profiling configuration must be a mapping")
    base = path.parent
    for key in ("executable", "perf_executable", "output_directory"):
        p = Path(raw[key]).expanduser()
        raw[key] = p if p.is_absolute() else (base / p).resolve()
    raw["models"] = [Model(str(x["id"]), ((base / x["path"]).resolve() if not Path(x["path"]).is_absolute() else Path(x["path"])), str(x["quantization"]).upper()) for x in raw["models"]]
    raw["workloads"] = [Workload(str(x["id"]), int(x["prompt_tokens"]), int(x["generated_tokens"]), int(x["threads"])) for x in raw["workloads"]]
    canonical = {k: v for k, v in yaml.safe_load(path.read_text(encoding="utf-8")).items() if k != "configuration_fingerprint"}
    raw["configuration_fingerprint"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return raw


def build_target_command(config: Mapping[str, Any], model: Model, workload: Workload) -> list[str]:
    depth = int(config["context_size"]) - workload.prompt_tokens - workload.generated_tokens
    if depth < 0:
        raise ValueError(f"context too small for {workload.workload_id}")
    return [str(config["executable"]), "-m", str(model.path), "-t", str(workload.threads),
            "-p", str(workload.prompt_tokens), "-n", str(workload.generated_tokens),
            "-b", str(config["batch"]), "-ub", str(config["ubatch"]), "-d", str(depth),
            "-r", "1", "-o", "json", "-mmp", "1" if config["mmap"] else "0",
            *[str(x) for x in config["cpu_only_flags"]]]


def invocation_id(mode: str, model: Model, workload: Workload, repetition: int, warmup: bool = False) -> str:
    text = f"{mode}|{model.logical_id}|{model.quantization}|{workload.workload_id}|{repetition}|{int(warmup)}"
    return f"{mode}-{model.quantization.lower()}-{workload.workload_id}-{'warmup' if warmup else repetition}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def _number(value: str) -> float | None:
    stripped = value.strip().replace(" ", "")
    if stripped.lower() in UNAVAILABLE or not stripped:
        return None
    stripped = stripped.replace(",", "")
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_perf_stat(text: str) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            malformed.append(line); continue
        if len(fields) < 3:
            malformed.append(line); continue
        value_text, unit, event = fields[0].strip(), fields[1].strip(), re.sub(r":[ukhp]+$", "", fields[2].strip())
        if not event:
            malformed.append(line); continue
        reason = None
        lower = value_text.lower()
        if "not supported" in lower: reason = "not supported"
        elif "not counted" in lower: reason = "not counted"
        value = _number(value_text)
        events[event] = {"value": value, "unit": unit, "supported": value is not None, "reason": reason}
    values = {key: item["value"] for key, item in events.items() if item["value"] is not None}
    def ratio(a: str, b: str) -> float | None:
        return values[a] / values[b] if a in values and b in values and values[b] else None
    derived = {"ipc": ratio("instructions", "cycles"), "branch_miss_rate": ratio("branch-misses", "branches"),
               "cache_miss_rate": ratio("cache-misses", "cache-references")}
    elapsed = next((v for k, v in values.items() if k in {"seconds time elapsed", "duration_time"}), None)
    if elapsed is not None: derived["elapsed_seconds"] = elapsed
    return {"events": events, "derived": derived, "malformed_lines": malformed}


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try: record = json.loads(line)
            except json.JSONDecodeError: continue
            if record.get("invocation_id"): latest[record["invocation_id"]] = record
    return latest


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())


def run_process(argv: Sequence[str], timeout: float) -> tuple[int | None, str, str, bool]:
    try:
        result = subprocess.run(list(argv), text=True, capture_output=True, timeout=timeout, check=False)
        return result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or "", exc.stderr or "", True
    except OSError as exc:
        return None, "", str(exc), False


def probe_events(perf: Path, events: Sequence[str]) -> tuple[list[str], list[dict[str,Any]]]:
    """Select individually working events so one WSL2 gap cannot poison a stat run."""
    supported=[]; diagnostics=[]
    for event in events:
        code,out,err,timed=run_process([str(perf),"stat","-x",",","-e",event,"--","true"],15)
        lower=(out+err).lower()
        classification="supported" if code==0 and not timed else "permission denied" if "permission" in lower or "perf_event_paranoid" in lower else "unsupported"
        diagnostics.append({"event":event,"classification":classification,"exit_code":code,"diagnostic":err.strip()})
        if classification=="supported": supported.append(event)
    return supported,diagnostics


def _target_failure(perf_code: int | None, stderr: str) -> bool:
    match = re.search(r"child process exited with code (\d+)", stderr, re.I)
    return bool(match and int(match.group(1)) != 0) or perf_code == 255 and "failed to load model" in stderr.lower()


def run_stat(config: Mapping[str, Any], resume: bool, retry_failures: bool, smoke: bool = False) -> dict[str, int]:
    output = Path(config["output_directory"]) / "perf-stat" / ("smoke.jsonl" if smoke else "runs.jsonl")
    prior = read_latest(output); counts = {"planned": 0, "completed": 0, "successful": 0, "failed": 0}
    supported_events,event_diagnostics=probe_events(config["perf_executable"],config["perf_events"])
    models: Iterable[Model] = config["models"][:1] if smoke else config["models"]
    workloads: Iterable[Workload] = (Workload("smoke-p128-n16-t4",128,16,4),) if smoke else config["workloads"]
    repetitions = 1 if smoke else int(config["stat_profile_count"])
    for model in models:
      for workload in workloads:
        target = build_target_command(config, model, workload)
        for warmup in range(int(config["warmup_count"])):
            iid = invocation_id("warmup", model, workload, warmup, True); counts["planned"] += 1
            if resume and iid in prior and (prior[iid].get("status") == "success" or not retry_failures): continue
            start=utcnow(); code,out,err,timed=run_process(target,float(config["timeout_seconds"])); end=utcnow()
            status="timed_out" if timed else "success" if code==0 else "target_failure"
            record={"schema_version":1,"invocation_id":iid,"mode":"warmup","warmup":True,"model_logical_id":model.logical_id,"quantization":model.quantization,"workload_id":workload.workload_id,"target_command":target,"start_timestamp":start,"end_timestamp":end,"target_exit_code":code,"perf_exit_code":None,"status":status,"stdout":out,"stderr":err,"configuration_fingerprint":config["configuration_fingerprint"]}
            append_jsonl(output,record); counts["completed"]+=1; counts["successful" if status=="success" else "failed"]+=1
            if status != "success": return counts
        for rep in range(repetitions):
            iid=invocation_id("stat",model,workload,rep); counts["planned"]+=1
            if resume and iid in prior and (prior[iid].get("status")=="success" or not retry_failures): continue
            stat_file=Path(config["output_directory"])/"perf-stat"/f"{iid}.csv"
            argv=[str(config["perf_executable"]),"stat","-x",",","-o",str(stat_file)]
            if supported_events: argv.extend(["-e",",".join(supported_events)])
            argv.extend(["--",*target])
            start=utcnow(); monotonic_start=time.monotonic(); code,out,err,timed=run_process(argv,float(config["timeout_seconds"])); wall_elapsed=time.monotonic()-monotonic_start; end=utcnow()
            stat_text=stat_file.read_text(encoding="utf-8",errors="replace") if stat_file.exists() else ""
            parsed=parse_perf_stat(stat_text)
            parsed["derived"].setdefault("elapsed_seconds",wall_elapsed)
            target_failed=_target_failure(code,err)
            status="timed_out" if timed else "target_failure" if target_failed else "success" if code==0 else "perf_failure"
            record={"schema_version":1,"invocation_id":iid,"mode":"stat","warmup":False,"repetition":rep,"model_logical_id":model.logical_id,"model_path":str(model.path),"quantization":model.quantization,"workload_id":workload.workload_id,"prompt_tokens":workload.prompt_tokens,"generated_tokens":workload.generated_tokens,"threads":workload.threads,"batch":config["batch"],"ubatch":config["ubatch"],"mmap":config["mmap"],"target_command":target,"profiler_command":argv,"start_timestamp":start,"end_timestamp":end,"target_exit_code":None if not target_failed else code,"perf_exit_code":code,"status":status,"stdout":out,"stderr":err,"perf_stat":parsed,"event_probe":event_diagnostics,"configuration_fingerprint":config["configuration_fingerprint"]}
            append_jsonl(output,record); counts["completed"]+=1; counts["successful" if status=="success" else "failed"]+=1
    return counts


def run_record(config: Mapping[str, Any], resume: bool, retry_failures: bool, smoke: bool = False) -> dict[str,int]:
    log=Path(config["output_directory"])/"perf-record"/("smoke.jsonl" if smoke else "runs.jsonl"); prior=read_latest(log)
    counts={"planned":0,"completed":0,"successful":0,"failed":0}
    models=config["models"][:1] if smoke else config["models"]
    workloads=[Workload("smoke-p128-n16-t4",128,16,4)] if smoke else [w for w in config["workloads"] if w.workload_id in config["record_workloads"]]
    reps=1 if smoke else int(config["record_profile_count"])
    for model in models:
      for workload in workloads:
       for rep in range(reps):
        iid=invocation_id("record",model,workload,rep); counts["planned"]+=1
        if resume and iid in prior and (prior[iid].get("status")=="success" or not retry_failures): continue
        data=Path(config["output_directory"])/"perf-record"/f"{iid}.data"; data.parent.mkdir(parents=True,exist_ok=True)
        target=build_target_command(config,model,workload)
        argv=[str(config["perf_executable"]),"record","-F","99","--call-graph","dwarf","-o",str(data),"--",*target]
        start=utcnow(); code,out,err,timed=run_process(argv,float(config["timeout_seconds"])); end=utcnow(); target_failed=_target_failure(code,err)
        status="timed_out" if timed else "target_failure" if target_failed else "success" if code==0 and data.exists() else "perf_failure"
        rec={"schema_version":1,"invocation_id":iid,"mode":"record","model_logical_id":model.logical_id,"quantization":model.quantization,"workload_id":workload.workload_id,"prompt_tokens":workload.prompt_tokens,"generated_tokens":workload.generated_tokens,"threads":workload.threads,"target_command":target,"profiler_command":argv,"perf_data":str(data),"start_timestamp":start,"end_timestamp":end,"target_exit_code":None if not target_failed else code,"perf_exit_code":code,"status":status,"stdout":out,"stderr":err,"configuration_fingerprint":config["configuration_fingerprint"]}
        append_jsonl(log,rec); counts["completed"]+=1; counts["successful" if status=="success" else "failed"]+=1
    return counts


def report_records(config: Mapping[str,Any], annotate: bool=False) -> dict[str,int]:
    source=read_latest(Path(config["output_directory"])/"perf-record"/"runs.jsonl")
    counts={"planned":0,"completed":0,"successful":0,"failed":0}
    for rec in source.values():
        if rec.get("status")!="success": continue
        counts["planned"]+=1; iid=rec["invocation_id"]; data=rec["perf_data"]
        argv=[str(config["perf_executable"]),"annotate" if annotate else "report","--stdio","-i",data]
        if not annotate: argv[2:2]=["--no-children","--percent-limit","0.1"]
        code,out,err,timed=run_process(argv,float(config["timeout_seconds"])); directory="perf-annotate" if annotate else "perf-report"
        dest=Path(config["output_directory"])/directory/f"{iid}.txt"; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(out+err,encoding="utf-8")
        counts["completed"]+=1; counts["successful" if code==0 and not timed else "failed"]+=1
    return counts


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("stat","record","report","annotate")); p.add_argument("--config",type=Path,default=Path("configs/cpu_profiling.yaml")); p.add_argument("--resume",action="store_true"); p.add_argument("--retry-failures",action="store_true"); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); config=load_config(a.config)
    result=run_stat(config,a.resume,a.retry_failures,a.smoke) if a.mode=="stat" else run_record(config,a.resume,a.retry_failures,a.smoke) if a.mode=="record" else report_records(config,a.mode=="annotate")
    print(json.dumps(result,sort_keys=True)); return 0 if result["failed"]==0 else 1


if __name__=="__main__": raise SystemExit(main())
