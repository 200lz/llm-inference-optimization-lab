from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_phase8_optimization import analyze, bootstrap_ci, load_pairs, make_plots
from benchmarks.run_phase8_paired import parse_bench, plan


def test_plan_is_paired_deterministic_and_interleaved() -> None:
    first = plan(8081, 20, 3); second = plan(8081, 20, 3)
    assert first == second
    measured = [row for row in first if not row["warmup"]]
    assert len(measured) == 40
    for pair in range(1, 21):
        assert {row["variant"] for row in measured if row["pair_id"] == pair} == {"baseline", "optimized"}


def test_parse_bench_extracts_both_phases() -> None:
    raw = json.dumps([{"n_prompt": 1024, "n_gen": 0, "avg_ts": 300},
                      {"n_prompt": 0, "n_gen": 64, "avg_ts": 60}])
    assert parse_bench(raw) == (300.0, 60.0)


def records(path: Path) -> None:
    rows = []
    for pair in range(1, 21):
        for offset, variant in enumerate(("baseline", "optimized")):
            rows.append({"warmup": False, "return_code": 0, "pair_id": pair, "variant": variant,
                         "execution_index": pair * 2 + offset, "prompt_tokens_per_second": 100 + (variant == "optimized"),
                         "generation_tokens_per_second": 50, "time_duration_seconds": 10 - 0.1 * (variant == "optimized")})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analysis_positive_data_and_bootstrap_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"; records(path); pairs = load_pairs(path); result = analyze(pairs)
    assert result["classification"] == "ACCEPTABLE POSITIVE TREND"
    assert result["metrics"]["prompt_throughput"]["sign_count"]["optimized_better"] == 20
    assert bootstrap_ci([1.0] * 20) == [1.0, 1.0]


def test_noisy_regression_with_ci_spanning_zero_is_inconclusive(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"; records(path); pairs = load_pairs(path)
    for index, pair in enumerate(pairs):
        pair["optimized"]["prompt_tokens_per_second"] = pair["baseline"]["prompt_tokens_per_second"] + (-5 if index % 2 else 4)
        pair["optimized"]["time_duration_seconds"] = pair["baseline"]["time_duration_seconds"] + (-1 if index % 2 else 1.3)
    assert analyze(pairs)["classification"] == "INCONCLUSIVE"


def test_incomplete_pair_rejected(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"warmup": False, "return_code": 0, "pair_id": 1,
                                "variant": "baseline"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_pairs(path)


def test_plot_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "runs.jsonl"; records(path); pairs = load_pairs(path)
    made = make_plots(pairs, analyze(pairs), tmp_path / "plots")
    assert len(made) == 4 and all(item.stat().st_size for item in made)
