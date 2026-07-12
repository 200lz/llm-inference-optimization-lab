from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.analyze_prefill_decode import aggregate, load_normalized, make_plots, render_report


def rows() -> list[dict[str, object]]:
    return [
        {"thread_count": 1, "prompt_tokens": 128, "generated_tokens": 64, "prompt_tokens_per_second": x, "generation_tokens_per_second": x / 2}
        for x in (10.0, 14.0)
    ] + [{"thread_count": 2, "prompt_tokens": 128, "generated_tokens": 64, "prompt_tokens_per_second": 24.0, "generation_tokens_per_second": 12.0}]


def test_aggregation_speedup_and_efficiency() -> None:
    result = aggregate(rows())
    prompt1 = next(r for r in result if r["phase"] == "prompt_processing" and r["thread_count"] == 1)
    prompt2 = next(r for r in result if r["phase"] == "prompt_processing" and r["thread_count"] == 2)
    assert prompt1["count"] == 2
    assert prompt1["mean"] == 12
    assert prompt1["median"] == 12
    assert prompt1["minimum"] == 10
    assert prompt1["maximum"] == 14
    assert prompt1["standard_deviation"] == pytest.approx(2.828427)
    assert prompt2["speedup"] == 2
    assert prompt2["parallel_efficiency"] == 1


def test_missing_baseline_is_explicit() -> None:
    only_two = [row for row in rows() if row["thread_count"] == 2]
    result = aggregate(only_two)
    assert all(r["speedup"] is None and r["parallel_efficiency"] is None for r in result)


def write_csv(path: Path, malformed: bool = False, duplicate: bool = False) -> None:
    fields = ["model_name", "thread_count", "prompt_tokens", "generated_tokens", "batch_size", "context_size", "repetition", "prompt_tokens_per_second", "generation_tokens_per_second"]
    row = {"model_name": "q4", "thread_count": 1, "prompt_tokens": 128, "generated_tokens": 64, "batch_size": 512, "context_size": 4096, "repetition": 1, "prompt_tokens_per_second": "bad" if malformed else 10, "generation_tokens_per_second": 5}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerow(row)
        if duplicate: writer.writerow(row)


def test_malformed_and_duplicate_csv_handling(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.csv"; write_csv(malformed, malformed=True)
    parsed, warnings = load_normalized(malformed)
    assert parsed == [] and "malformed" in warnings[0]
    duplicate = tmp_path / "duplicate.csv"; write_csv(duplicate, duplicate=True)
    parsed, warnings = load_normalized(duplicate)
    assert len(parsed) == 1 and "duplicate" in warnings[0]


def test_report_generation_handles_empty_groups() -> None:
    run = {"successful_runs": 0, "total_runs": 0, "failed_runs": 0, "timed_out_runs": 0, "parse_failures": 0, "total_duration_seconds": 0}
    report = render_report([], run, {}, [])
    assert "No usable groups" in report
    assert "Prefill" in report


def test_plot_generation_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    paths = make_plots(aggregate(rows()), tmp_path)
    assert len(paths) == 6
    assert all(path.stat().st_size > 0 for path in paths)
