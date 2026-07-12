from __future__ import annotations
import csv, io, json
from pathlib import Path
import pytest
from benchmarks import analyze_quantization as analyzer
from benchmarks import analyze_quantization_quality as quality_analyzer
from benchmarks import run_llama_bench as harness
from benchmarks import run_quantization_quality as quality
from scripts import download_model

def multi_config(tmp_path:Path)->harness.BenchmarkConfig:
    text=f'''executable: /fake/bench
models:
  - {{id: f16, path: /models/f16.gguf, quantization: F16}}
  - {{id: q8, path: /models/q8.gguf, quantization: Q8_0}}
  - {{id: q4, path: /models/q4.gguf, quantization: Q4_K_M}}
threads: [4]
prompt_tokens: [128]
generated_tokens: [64]
batch_sizes: [512]
ubatch_sizes: [512]
context_sizes: [512]
repetitions: 1
warmup_runs: 0
timeout_seconds: 1
output_directory: {tmp_path / "out"}
'''
    path=tmp_path/"config.yaml";path.write_text(text);return harness.load_config(path)

def test_multi_model_ids_cannot_collide(tmp_path:Path):
    plan=harness.invocations(multi_config(tmp_path)); assert len(plan)==3
    assert len({x.invocation_id for x in plan})==3
    assert {x.model.quantization for x in plan}=={"F16","Q8_0","Q4_K_M"}

def test_duplicate_model_id_rejected(tmp_path:Path):
    config=multi_config(tmp_path); path=tmp_path/"bad.yaml"
    text=(tmp_path/"config.yaml").read_text().replace("id: q8","id: f16");path.write_text(text)
    with pytest.raises(harness.ConfigError,match="duplicate model ID"): harness.load_config(path)

def test_phase5_matrix_counts():
    config=harness.load_config(Path("configs/quantization_comparison.yaml")); assert len(list(harness.cases(config)))==5
    assert len(harness.invocations(config))==90

def test_failed_model_record_preserved(tmp_path:Path,monkeypatch:pytest.MonkeyPatch):
    fixture=(Path(__file__).parents[1]/"fixtures/llama_bench_success.txt").read_text(); calls=0
    def execute(*args):
        nonlocal calls; calls+=1
        return {"stdout":fixture if calls!=2 else "","stderr":"bad" if calls==2 else "","return_code":7 if calls==2 else 0,"elapsed_seconds":.1,"timed_out":False}
    monkeypatch.setattr(harness,"execute",execute); raw,_,_=harness.run(multi_config(tmp_path),tmp_path)
    records=[json.loads(x) for x in raw.read_text().splitlines()]; assert len(records)==3 and records[1]["status"]=="failed"

def write_csv(path:Path, include_f16=True):
    fields=list(harness.CSV_FIELDS); rows=[]
    for q,value,size in (("F16",100,1000),("Q8_0",150,600),("Q4_K_M",200,400)):
        if q=="F16" and not include_f16: continue
        for rep in (1,2):
            row={x:"" for x in fields}; row.update(model_logical_id=q,model_filename=q+".gguf",quantization=q,model_sha256="a"*64,model_file_size=size,invocation_id=f"{q}:{rep}",model_name=q,model_size=str(size),backend="CPU",thread_count=4,prompt_tokens=128,generated_tokens=64,prompt_tokens_per_second=value,generation_tokens_per_second=value/10,test_identifier="x",batch_size=512,ubatch_size=512,context_size=512,repetition=rep,elapsed_seconds=1)
            rows.append(row)
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def test_analyzer_relatives_and_missing_f16(tmp_path:Path):
    path=tmp_path/"a.csv";write_csv(path);rows,w=analyzer.load_rows(path);groups=analyzer.aggregate(rows,{"F16":1000,"Q8_0":600,"Q4_K_M":400})
    q8=next(x for x in groups if x["quantization"]=="Q8_0" and x["metric"]=="prompt_processing")
    assert q8["throughput_ratio_to_f16"]==1.5 and q8["file_size_reduction_percent"]==40
    write_csv(path,False);groups=analyzer.aggregate(analyzer.load_rows(path)[0],{"Q8_0":600});assert all(x["throughput_ratio_to_f16"] is None for x in groups)

def test_analyzer_malformed_duplicate_and_plots(tmp_path:Path):
    path=tmp_path/"a.csv";write_csv(path); text=path.read_text(); path.write_text(text+text.splitlines()[-1]+"\n"+text.splitlines()[-1].replace(",200,",",broken,")+"\n")
    rows,warnings=analyzer.load_rows(path); assert warnings
    made=analyzer.plots(analyzer.aggregate(rows,{"F16":1000,"Q8_0":600,"Q4_K_M":400}),{"F16":1000,"Q8_0":600,"Q4_K_M":400},tmp_path/"plots"); assert len(made)==8 and all(p.exists() for p in made)

def test_model_config_compatibility_and_checksum(tmp_path:Path):
    models=download_model.load_model_config(Path("configs/models/quantization_models.yaml")); assert set(models)=={"f16","q8_0","q4_k_m"}
    with pytest.raises(download_model.DownloadError,match="checksum mismatch"):
        download_model.download_model("owner/repo","x.gguf",tmp_path,lambda *a,**k:type("Response",(io.BytesIO,),{"status":200,"headers":{"Content-Length":"3"},"__enter__":lambda s:s,"__exit__":lambda *a:None})(b"bad"),expected_sha256="0"*64)

def quality_config(tmp_path:Path):
    path=tmp_path/"q.yaml";path.write_text(f'''executable: /fake/cli
models:
  - {{id: f, path: /m/f, quantization: F16}}
  - {{id: q8, path: /m/q8, quantization: Q8_0}}
  - {{id: q4, path: /m/q4, quantization: Q4_K_M}}
output_directory: {tmp_path/'quality'}
timeout_seconds: 1
seed: 42
temperature: 0
max_generated_tokens: 8
threads: 4
prompts: [{{id: p, text: Hello}}]
''');return quality.load_config(path)

def test_quality_command_and_resume(tmp_path:Path,monkeypatch:pytest.MonkeyPatch):
    config=quality_config(tmp_path); command=quality.build_command(config,config["models"][0],"Hello")
    assert "--no-conversation" in command and "--single-turn" in command and "<|im_start|>user" in command[command.index("-p")+1] and command[command.index("--temp")+1]=="0" and command[-4:]==["-ngl","0","-dev","none"]
    monkeypatch.setattr(quality,"execute",lambda *a:{"stdout":"Same","stderr":"eval time = 1 / 3 runs","return_code":0,"timed_out":False,"duration_seconds":.1})
    path=quality.run(config); before=path.read_text();quality.run(config,True);assert path.read_text()==before and len(before.splitlines())==3
    assert all(json.loads(x)["generated_token_count"]==3 for x in before.splitlines())

def test_quality_timeout_bytes_are_serializable(monkeypatch:pytest.MonkeyPatch):
    def timeout(*args, **kwargs):
        raise __import__("subprocess").TimeoutExpired(args[0], kwargs["timeout"], output=b"partial", stderr=b"timing")
    monkeypatch.setattr(quality.subprocess,"run",timeout)
    result=quality.execute(["fake"],1)
    assert result["timed_out"] and result["stdout"]=="partial" and result["stderr"]=="timing"
    json.dumps(result)

def test_quality_comparisons_and_failure(tmp_path:Path):
    path=tmp_path/"q.jsonl"; records=[{"invocation_id":"f:p","prompt_id":"p","quantization":"F16","status":"success","raw_generated_text":"Hello  World"},{"invocation_id":"q:p","prompt_id":"p","quantization":"Q8_0","status":"success","raw_generated_text":"hello world"},{"invocation_id":"x:p","prompt_id":"p","quantization":"Q4_K_M","status":"failed","raw_generated_text":""}]
    path.write_text("\n".join(json.dumps(x) for x in records)); result=quality_analyzer.analyze(path); q8=result["comparisons"][1]; assert not q8["exact_match_to_f16"] and q8["normalized_match_to_f16"]; assert result["comparisons"][2]["exact_match_to_f16"] is None

def test_quality_analyzer_extracts_assistant_text():
    wrapped="banner and model metadata\n<|im_start|>assistant\nThe answer.\n\n[ Prompt: 1 t/s | Generation: 2 t/s ]\nExiting..."
    assert quality_analyzer.assistant_text(wrapped)=="The answer."
