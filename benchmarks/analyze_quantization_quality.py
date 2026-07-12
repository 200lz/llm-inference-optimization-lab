#!/usr/bin/env python3
"""Describe deterministic output differences relative to F16."""
from __future__ import annotations
import argparse,json,re,unicodedata
from pathlib import Path

def normalize(text:str)->str: return " ".join(unicodedata.normalize("NFKC",text).casefold().split())
def assistant_text(raw:str)->str:
    marker="<|im_start|>assistant\n"
    text=raw.rsplit(marker,1)[-1] if marker in raw else raw
    text=re.split(r"\n\n\[ Prompt:",text,maxsplit=1)[0]
    return text.strip()
def analyze(path:Path)->dict:
    records=[]; warnings=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        try:r=json.loads(line)
        except json.JSONDecodeError: warnings.append(f"line {line_no}: malformed JSON"); continue
        records.append(r)
    latest={r.get("invocation_id"):r for r in records}
    if len(latest)!=len(records): warnings.append(f"{len(records)-len(latest)} superseded retry record(s) excluded")
    records=list(latest.values())
    refs={r.get("prompt_id"):r for r in records if r.get("quantization")=="F16" and r.get("status")=="success"}
    comparisons=[]
    for r in records:
        ref=refs.get(r.get("prompt_id")); text=assistant_text(r.get("raw_generated_text", "")) if r.get("status")=="success" else None; base=assistant_text(ref.get("raw_generated_text", "")) if ref else None
        comparisons.append({"prompt_id":r.get("prompt_id"),"quantization":r.get("quantization"),"completed_successfully":r.get("status")=="success","exact_match_to_f16":None if text is None or base is None else text==base,"normalized_match_to_f16":None if text is None or base is None else normalize(text)==normalize(base),"output_length":None if text is None else len(text),"output_length_difference_to_f16":None if text is None or base is None else len(text)-len(base)})
    return {"schema_version":1,"prompt_count":len({r.get("prompt_id") for r in records}),"all_completed_successfully":all(r.get("status")=="success" for r in records),"warnings":warnings,"token_metrics":None,"token_metrics_limitation":"llama-cli stdout does not reliably expose generated token IDs separately from chat rendering; token agreement and first-divergence metrics are omitted.","comparisons":comparisons}
def main():
    p=argparse.ArgumentParser();p.add_argument("raw_jsonl",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args(); result=analyze(a.raw_jsonl);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(f'analyzed {len(result["comparisons"])} outputs')
if __name__=="__main__":main()
