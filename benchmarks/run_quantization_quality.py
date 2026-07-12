#!/usr/bin/env python3
"""Run a durable deterministic llama-cli prompt suite across quantizations."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

class QualityError(ValueError): pass

def load_config(path: Path) -> dict[str, Any]:
    try: raw=yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError,yaml.YAMLError) as exc: raise QualityError(str(exc)) from exc
    required={"executable","models","output_directory","timeout_seconds","seed","temperature","max_generated_tokens","threads","prompts"}
    if not isinstance(raw,dict) or required-set(raw): raise QualityError("missing required quality configuration fields")
    if len({m.get("id") for m in raw["models"]}) != len(raw["models"]): raise QualityError("duplicate model ID")
    if {str(m.get("quantization","")).upper() for m in raw["models"]}!={"F16","Q8_0","Q4_K_M"}: raise QualityError("exactly F16, Q8_0, and Q4_K_M are required")
    if len({p.get("id") for p in raw["prompts"]}) != len(raw["prompts"]): raise QualityError("duplicate prompt ID")
    for key in ("executable","output_directory"):
        value=Path(raw[key]).expanduser(); raw[key]=value if value.is_absolute() else (path.parent/value).resolve()
    for model in raw["models"]:
        value=Path(model["path"]).expanduser(); model["path"]=value if value.is_absolute() else (path.parent/value).resolve()
    return raw

def build_command(config:dict[str,Any],model:dict[str,Any],prompt:str)->list[str]:
    rendered=("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
              "<|im_end|>\n<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n")
    command=[str(config["executable"]),"-m",str(model["path"]),"-p",rendered,"-n",str(config["max_generated_tokens"]),
             "-t",str(config["threads"]),"-s",str(config["seed"]),"--temp",str(config["temperature"]),
             "--no-conversation","--single-turn","--no-display-prompt","--simple-io","-ngl","0","-dev","none"]
    return command

def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""

def execute(command:list[str],timeout:float)->dict[str,Any]:
    start=time.perf_counter()
    try:
        p=subprocess.run(command,text=True,capture_output=True,timeout=timeout,check=False)
        return {"stdout":p.stdout,"stderr":p.stderr,"return_code":p.returncode,"timed_out":False,"duration_seconds":time.perf_counter()-start}
    except subprocess.TimeoutExpired as exc:
        return {"stdout":_text(exc.stdout),"stderr":_text(exc.stderr),"return_code":None,"timed_out":True,"duration_seconds":time.perf_counter()-start}
    except OSError as exc: return {"stdout":"","stderr":str(exc),"return_code":None,"timed_out":False,"duration_seconds":time.perf_counter()-start}

def generated_token_count(stderr:str)->int|None:
    matches=re.findall(r"eval time\s*=.*?/\s*(\d+) runs",stderr)
    return int(matches[-1]) if matches else None

def fingerprint(config:dict[str,Any])->str:
    payload={k:v for k,v in config.items() if k!="output_directory"}; payload["executable"]=str(payload["executable"])
    payload["models"]=[dict(m,path=str(m["path"])) for m in payload["models"]]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def run(config:dict[str,Any],resume:bool=False,retry_failures:bool=False,force_resume:bool=False)->Path:
    out=config["output_directory"]; out.mkdir(parents=True,exist_ok=True); path=out/"run.jsonl"; summary_path=out/"run-summary.json"; fp=fingerprint(config)
    existing=[]
    if resume and path.exists(): existing=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    if not resume and path.exists(): raise QualityError("output exists; use --resume")
    if resume and summary_path.exists() and json.loads(summary_path.read_text())["configuration_fingerprint"]!=fp and not force_resume: raise QualityError("configuration fingerprint mismatch")
    latest={r["invocation_id"]:r for r in existing}; run_id=str(uuid.uuid4()); total=len(config["models"])*len(config["prompts"])
    with path.open("a",encoding="utf-8") as stream:
      number=0
      for model in config["models"]:
       for prompt in config["prompts"]:
        number+=1; iid=f'{model["id"]}:{prompt["id"]}'
        if iid in latest and (not retry_failures or latest[iid].get("status")=="success"): continue
        print(f'[{number}/{total}] model={model["id"]} prompt={prompt["id"]}',flush=True); command=build_command(config,model,prompt["text"]); started=datetime.now(timezone.utc).isoformat(); result=execute(command,float(config["timeout_seconds"])); status="timed_out" if result["timed_out"] else "success" if result["return_code"]==0 else "failed"
        record={"run_id":run_id,"invocation_id":iid,"prompt_id":prompt["id"],"prompt_text":prompt["text"],"model_logical_id":model["id"],"quantization":model["quantization"],"raw_generated_text":result["stdout"],"generated_token_count":generated_token_count(str(result["stderr"])),"status":status,"started_at":started,"completed_at":datetime.now(timezone.utc).isoformat(),"command":command,"seed":config["seed"],"temperature":config["temperature"],"max_generated_tokens":config["max_generated_tokens"],**result}
        stream.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+"\n"); stream.flush(); os.fsync(stream.fileno()); existing.append(record)
        summary_path.write_text(json.dumps({"configuration_fingerprint":fp,"planned":total,"completed":len({r["invocation_id"] for r in existing}),"successful":sum(r["status"]=="success" for r in existing)},indent=2)+"\n")
    return path

def main():
    p=argparse.ArgumentParser(); p.add_argument("config",type=Path); p.add_argument("--resume",action="store_true"); p.add_argument("--retry-failures",action="store_true"); p.add_argument("--force-resume",action="store_true"); a=p.parse_args()
    try:
        if a.retry_failures and not a.resume: raise QualityError("--retry-failures requires --resume")
        print(run(load_config(a.config.resolve()),a.resume,a.retry_failures,a.force_resume))
    except QualityError as exc: p.error(str(exc))
if __name__=="__main__": main()
