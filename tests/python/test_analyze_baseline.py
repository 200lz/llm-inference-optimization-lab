from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.analyze_baseline import analyze


def test_analysis_statistics_and_grouping(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    fields = ["model_name", "thread_count", "prompt_tokens", "generated_tokens",
              "prompt_tokens_per_second", "generation_tokens_per_second"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"model_name": "model", "thread_count": 2, "prompt_tokens": 128, "generated_tokens": 64,
             "prompt_tokens_per_second": 10, "generation_tokens_per_second": 4},
            {"model_name": "model", "thread_count": 2, "prompt_tokens": 128, "generated_tokens": 64,
             "prompt_tokens_per_second": 14, "generation_tokens_per_second": 6},
        ])
    rows = analyze(path, "example-q4_k_m.gguf")
    prompt = next(row for row in rows if row["test_type"] == "prompt_processing")
    assert prompt["count"] == 2
    assert prompt["mean"] == 12
    assert prompt["median"] == 12
    assert prompt["minimum"] == 10
    assert prompt["maximum"] == 14
    assert prompt["standard_deviation"] == pytest.approx(2.828427)
    assert prompt["coefficient_of_variation"] == pytest.approx(0.23570226)
    assert prompt["quantization"] == "Q4_K_M"
