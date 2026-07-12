from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from benchmarks.analyze_cpu_profiling import (aggregate, categorize_function, comparisons,
    cumulative_share, parse_perf_report, plots)
from benchmarks.run_cpu_profiling import (Model, Workload, append_jsonl, build_target_command,
    invocation_id, parse_perf_stat, read_latest, run_process)
from scripts.probe_perf_events import classify_probe, render_markdown, run_probe


def test_probe_supported_unsupported_and_permission():
    assert classify_probe(0,"","")[0]=="supported"
    assert classify_probe(1,"","event is not supported")[0]=="unsupported"
    assert classify_probe(255,"","Permission denied; perf_event_paranoid")[0]=="permission denied"


def test_probe_not_counted_and_malformed():
    assert classify_probe(0,"","<not counted>")[0]=="noisy or unreliable"
    assert classify_probe(2,"","")[0]=="unsupported"


def test_missing_perf(monkeypatch):
    monkeypatch.setattr("shutil.which",lambda _:None)
    result=run_probe(); assert result["events"]==[] and "error" in result
    assert "missing" in render_markdown(result)


def test_perf_stat_parsing_and_derived():
    text="1000,,cycles,1,100.00,,\n2000,,instructions,2,100.00,,\n100,,branches\n5,,branch-misses\n50,,cache-references\n10,,cache-misses\n1.25,seconds,time elapsed\n"
    parsed=parse_perf_stat(text)
    assert parsed["derived"]["ipc"]==2
    assert parsed["derived"]["branch_miss_rate"]==.05
    assert parsed["derived"]["cache_miss_rate"]==.2


def test_perf_event_modifiers_are_normalized():
    parsed=parse_perf_stat("100,,cycles:u\n200,,instructions:u\n")
    assert parsed["derived"]["ipc"]==2 and "cycles" in parsed["events"]


def test_perf_stat_unsupported_missing_and_zero():
    parsed=parse_perf_stat("<not supported>,,cycles\n<not counted>,,instructions\n0,,branches\n1,,branch-misses\nbad")
    assert parsed["events"]["cycles"]["value"] is None
    assert parsed["events"]["instructions"]["reason"]=="not counted"
    assert parsed["derived"]["ipc"] is None and parsed["derived"]["branch_miss_rate"] is None
    assert parsed["malformed_lines"]==["bad"]


def test_locale_independent_comma_csv():
    # perf CSV quotes values containing grouping separators.
    parsed=parse_perf_stat('"1,234",,cycles\n"2,468",,instructions\n')
    assert parsed["derived"]["ipc"]==2


def config(tmp_path):
    return {"executable":tmp_path/"llama bench","context_size":4096,"batch":512,"ubatch":512,"mmap":True,"cpu_only_flags":["-ngl","0","-dev","none"]}


def test_command_construction_and_no_shell(tmp_path):
    command=build_target_command(config(tmp_path),Model("model-f16",tmp_path/"m.gguf","F16"),Workload("primary",1024,64,4))
    assert command[0].endswith("llama bench") and command[-4:]==["-ngl","0","-dev","none"]
    assert "-mmp" in command and "-ub" in command
    # argv execution proves spaces are not interpreted by a shell.
    code,_,_,_=run_process(["/bin/true"],1); assert code==0


def test_invocation_identity_stable_and_specific(tmp_path):
    m=Model("qwen-f16",tmp_path/"m","F16"); w=Workload("p1024",1024,64,4)
    assert invocation_id("stat",m,w,0)==invocation_id("stat",m,w,0)
    assert invocation_id("stat",m,w,0)!=invocation_id("stat",m,Workload("p128",128,64,4),0)
    assert "f16" in invocation_id("stat",m,w,0) and "p1024" in invocation_id("stat",m,w,0)


def test_incremental_persistence_and_resume_latest(tmp_path):
    path=tmp_path/"run.jsonl"; append_jsonl(path,{"invocation_id":"x","status":"failed"}); append_jsonl(path,{"invocation_id":"x","status":"success"})
    assert read_latest(path)["x"]["status"]=="success" and len(path.read_text().splitlines())==2


def test_timeout_preserved():
    code,out,err,timed=run_process(["/bin/sleep","2"],.01)
    assert code is None and timed and out=="" and err==""


REPORT="""# Samples: 10K of event 'cycles'
  40.00%  llama-bench  libggml-cpu.so  [.] ggml_vec_dot_q4_K_q8_K
  20.00%  llama-bench  libggml-cpu.so  [.] ggml_vec_dot_q4_K_q8_K
  15.00%  llama-bench  libc.so.6       [.] memcpy
   5.00%  llama-bench  [unknown]       [.] 0x1234
"""


def test_report_symbol_dso_duplicate_unresolved_and_cumulative():
    rows=parse_perf_report(REPORT)
    assert rows[0]["function"]=="ggml_vec_dot_q4_K_q8_K" and rows[0]["shared_object"]=="libggml-cpu.so"
    assert rows[0]["overhead_percent"]==60 and rows[0]["family"]=="quantized dot product"
    assert any(x["symbol_status"]=="unresolved" for x in rows)
    assert cumulative_share(rows,2)==75


@pytest.mark.parametrize(("name","family"),[("dequantize_row_q4","dequantization or unpacking"),("ggml_gemm","GEMM/GEMV"),("pthread_mutex_lock","thread synchronization"),("mystery","unknown")])
def test_family_categories(name,family): assert categorize_function(name)==family


def record(q,work="primary",cycles=100,instructions=200):
    return {"invocation_id":q,"mode":"stat","status":"success","quantization":q,"workload_id":work,"prompt_tokens":100,"generated_tokens":20,"threads":4,"batch":512,"ubatch":512,
            "perf_stat":{"events":{"cycles":{"value":cycles},"instructions":{"value":instructions},"branches":{"value":10},"branch-misses":{"value":1}},"derived":{"ipc":instructions/cycles,"branch_miss_rate":.1}}}


def test_analyzer_exact_matching_relative_and_missing():
    groups=aggregate([record("F16"),record("Q8_0",cycles=150,instructions=225),record("Q4_K_M","different")]); comp=comparisons(groups)
    q8=next(x for x in comp if x["quantization"]=="Q8_0"); assert q8["relative_to_f16"]["cycles_ratio"]==1.5
    q4=next(x for x in comp if x["quantization"]=="Q4_K_M"); assert q4["relative_to_f16"]["cycles_ratio"] is None
    f16=next(x for x in groups if x["quantization"]=="F16"); assert f16["metrics"]["cache_miss_rate"]["available"] is False


def test_unsupported_not_substituted_with_zero():
    row=aggregate([record("F16")])[0]
    assert row["metrics"]["cache-misses"].get("mean") is None
    assert row["metrics"]["cache-misses"]["reason"]


def test_plot_generation_omits_unsupported(tmp_path):
    rec=record("F16","primary-p1024-n64-t4"); groups=aggregate([rec]); key=("F16",100,20,4,512,512)
    made=plots(groups,{key:{"prompt":10}}, {},tmp_path)
    assert "quantization_cycles.png" in made and "quantization_cache_miss_rate.png" not in made
    assert all((tmp_path/name).exists() for name in made)


def test_fake_executable_failure_preserved(tmp_path):
    fake=tmp_path/"fake perf"; fake.write_text("#!/bin/sh\necho diagnostic >&2\nexit 7\n"); fake.chmod(fake.stat().st_mode|stat.S_IXUSR)
    code,out,err,timed=run_process([str(fake)],2)
    assert code==7 and "diagnostic" in err and not timed
