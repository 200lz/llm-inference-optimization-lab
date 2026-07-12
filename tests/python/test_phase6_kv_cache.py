from __future__ import annotations
import csv, json
from pathlib import Path
import pytest
from benchmarks import analyze_kv_cache as analyzer
from benchmarks import run_llama_bench as harness

def test_phase6_config_and_ids() -> None:
    config=harness.load_config(Path("configs/kv_cache_context_smoke.yaml")); cases=list(harness.cases(config))
    assert len(cases)==6 and {c.kv_key_type for c in cases}=={"f16","q8_0","q4_0"}
    ids={harness.case_id(c) for c in cases}; assert len(ids)==6 and all("-c" in i and "-kv-" in i for i in ids)
    commands=[harness.build_command(config,c) for c in cases]
    assert all("-ctk" in c and "-ctv" in c and c[-4:]==["-ngl","0","-dev","none"] for c in commands)

def test_context_and_cache_change_fingerprint(tmp_path: Path) -> None:
    base=Path("configs/kv_cache_context_smoke.yaml").read_text()
    a=tmp_path/"a.yaml"; b=tmp_path/"b.yaml"; a.write_text(base.replace("../",str(Path.cwd())+"/")); b.write_text(a.read_text().replace("context_sizes: [512, 2048]","context_sizes: [512]"))
    assert harness.config_fingerprint(harness.load_config(a)) != harness.config_fingerprint(harness.load_config(b))

def test_unknown_cache_rejected(tmp_path: Path) -> None:
    text=Path("configs/kv_cache_context_smoke.yaml").read_text().replace("q4_0, value: q4_0","made_up, value: q4_0")
    path=tmp_path/"bad.yaml"; path.write_text(text)
    with pytest.raises(harness.ConfigError,match="unknown"): harness.load_config(path)

@pytest.mark.parametrize(("text","expected"),[("Maximum resident set size (kbytes): 12345",12345),("Maximum resident set size (kbytes): bad",None),("",None)])
def test_time_parsing(text: str, expected: int|None) -> None: assert harness.parse_time_verbose(text)==expected

def test_kv_memory_parsing_and_units() -> None:
    assert harness.parse_kv_cache_bytes("CPU KV buffer size = 12.50 MiB") == 13_107_200
    assert harness.parse_kv_cache_bytes("CPU KV buffer size = nope MiB") is None

def row(iid: str, cache: str, tg: str="20", rss: str="1000") -> dict[str,str]:
    return {"invocation_id":iid,"model_logical_id":"m","context_size":"512","prompt_tokens":"256","generated_tokens":"64","thread_count":"4","batch_size":"512","ubatch_size":"512","kv_key_type":cache,"kv_value_type":cache,"prompt_tokens_per_second":"100","generation_tokens_per_second":tg,"elapsed_seconds":"2","peak_rss_kib":rss,"reported_kv_cache_bytes":"500"}

def test_analyzer_reference_ratios_and_duplicates(tmp_path: Path) -> None:
    path=tmp_path/"n.csv"; fields=list(row("a","f16"))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows([row("a","f16"),row("b","q8_0","22","800"),row("b","q8_0")])
    rows,warnings=analyzer.load_rows(path); groups=analyzer.aggregate(rows)
    q8=next(g for g in groups if g["kv_cache_id"]=="kv-q8_0-q8_0")
    assert q8["relative_to_f16"]["generation_tokens_per_second"]["ratio_to_f16"]==1.1
    assert q8["relative_to_f16"]["peak_rss_kib"]["reduction_percent"] == pytest.approx(20)
    assert warnings

def test_missing_reference_is_na() -> None:
    groups=analyzer.aggregate([{**row("b","q8_0"),"context_size":512,"prompt_tokens":256,"generated_tokens":64,"thread_count":4,"batch_size":512,"ubatch_size":512,**{m:1.0 for m in analyzer.METRICS}}])
    assert groups[0]["relative_to_f16"]["generation_tokens_per_second"]["ratio_to_f16"] is None

def test_f16_reference_requires_exact_workload_match() -> None:
    f16={**row("f","f16"),"context_size":512,"prompt_tokens":256,"generated_tokens":32,"thread_count":4,"batch_size":512,"ubatch_size":512}
    q8={**row("q","q8_0"),"context_size":512,"prompt_tokens":256,"generated_tokens":64,"thread_count":4,"batch_size":512,"ubatch_size":512}
    for record in (f16,q8): record.update({m:1.0 for m in analyzer.METRICS})
    groups=analyzer.aggregate([f16,q8]); q8_group=next(g for g in groups if g["kv_cache_id"]=="kv-q8_0-q8_0")
    assert q8_group["relative_to_f16"]["generation_tokens_per_second"]["ratio_to_f16"] is None

def test_empty_aggregation_and_stats() -> None:
    assert analyzer.aggregate([])==[] and analyzer.stats([])["count"]==0

def test_cv_is_stored_as_ratio_and_rendered_as_percent() -> None:
    result=analyzer.stats([8.0,12.0])
    assert result["sample_stdev"] == pytest.approx(2.8284271247461903)
    assert result["coefficient_of_variation"] == pytest.approx(0.282842712474619)

def _group(context: int, prompt: int, generated: int, cache: str="f16") -> dict:
    records=[]
    for index,tg in enumerate((8.0,12.0),1):
        record={**row(f"{context}-{prompt}-{generated}-{cache}-{index}",cache),"context_size":context,"prompt_tokens":prompt,
                "generated_tokens":generated,"thread_count":4,"batch_size":512,"ubatch_size":512}
        record.update({m:tg for m in analyzer.METRICS}); record["reported_kv_cache_bytes"]=None; records.append(record)
    return analyzer.aggregate(records)[0]

def test_explicit_plot_slices_do_not_mix_workloads() -> None:
    valid_a=[_group(c,p,n) for c,p,n in sorted(analyzer.SLICE_A)]
    unrelated=_group(1024,256,64)
    selected=analyzer.select_slice(valid_a+[unrelated],analyzer.SLICE_A)
    assert {(g["context_size"],g["prompt_tokens"],g["generated_tokens"]) for g in selected} == analyzer.SLICE_A
    assert unrelated not in selected
    assert {g["generated_tokens"] for g in selected} == {32}

def test_plots_convert_cv_once_and_skip_empty_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    groups=[_group(c,p,n) for c,p,n in sorted(analyzer.SLICE_C)]
    plotted=[]; monkeypatch.setattr(analyzer.plt,"plot",lambda x,y,**kwargs: plotted.append((list(x),list(y),kwargs)))
    monkeypatch.setattr(analyzer.plt,"savefig",lambda path: None)
    made=analyzer.plots(groups,tmp_path)
    assert not any("reported_kv_cache_bytes" in name for name in made)
    cv_lines=[values for _,values,_ in plotted if values and values[0] == pytest.approx(28.2842712474619)]
    assert cv_lines and all(value == pytest.approx(28.2842712474619) for value in cv_lines[0])

def test_report_regeneration_preserves_required_sections() -> None:
    groups=[]
    for cache in ("f16","q8_0"):
        groups.extend(_group(c,p,n,cache) for c,p,n in sorted(analyzer.SLICE_A | analyzer.SLICE_B | analyzer.SLICE_C))
    summary={"unique_invocations":216,"warmups":36,"measured":180,"success":216,"failed":0,"timed_out":0,"parse_error":0,
             "total_elapsed_seconds":6208.4,"environment":{"cpu_model":"cpu","kernel":"kernel","llama_cpp_git_commit":"abc"}}
    config={"models":[{"path":"model.gguf","sha256":"hash","byte_size":1}],"threads":[4],"batch_sizes":[512],"ubatch_sizes":[512],
            "timeout_seconds":900,"repetitions":5}
    report=analyzer.render(groups,summary,config,["actual_growth_elapsed_seconds.png"])
    for section in ("## Environment","## KV-cache support","## Experiment matrix","## Performance and variability summary",
                    "## Memory results","## Relative F16 comparisons","## Key observations","## CV extremes","## Results","## Plots","## Limitations"):
        assert section in report
    assert "CV ratio" in report and "CV " in report and "%" in report
    assert "reported_kv_cache_bytes` is N/A" in report
    assert "compute-bound or memory-bound" in report
