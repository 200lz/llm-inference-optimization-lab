from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks import run_llama_bench as harness


FIXTURES = Path(__file__).parents[1] / "fixtures"


def config_text(output: Path, **overrides: str) -> str:
    values = {
        "executable": "/opt/llama-bench", "model": "/models/test.gguf",
        "threads": "[2, 4]", "prompt_tokens": "[128]", "generated_tokens": "[32]",
        "batch_sizes": "[64]", "context_sizes": "[512]", "repetitions": "2",
        "warmup_runs": "1", "timeout_seconds": "30", "output_directory": str(output),
    }
    values.update(overrides)
    return "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n"


def write_config(tmp_path: Path, **overrides: str) -> harness.BenchmarkConfig:
    path = tmp_path / "config.yaml"
    path.write_text(config_text(tmp_path / "results", **overrides), encoding="utf-8")
    return harness.load_config(path)


def test_valid_config_parsing(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    assert config.threads == (2, 4)
    assert config.repetitions == 2
    assert config.output_directory == tmp_path / "results"


@pytest.mark.parametrize("override", [
    {"threads": "[]"}, {"threads": "[0]"}, {"repetitions": "0"},
    {"warmup_runs": "-1"}, {"timeout_seconds": "false"},
])
def test_invalid_config_rejection(tmp_path: Path, override: dict[str, str]) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(config_text(tmp_path, **override), encoding="utf-8")
    with pytest.raises(harness.ConfigError):
        harness.load_config(path)


def test_command_construction_is_an_argv_list(tmp_path: Path) -> None:
    config = write_config(tmp_path, executable="'/path with spaces/llama-bench'", model="'/model with spaces.gguf'")
    command = harness.build_command(config, harness.Case(4, 128, 32, 64, 512))
    assert command == [
        "/path with spaces/llama-bench", "-m", "/model with spaces.gguf", "-t", "4",
        "-p", "128", "-n", "32", "-b", "64", "-ub", "512", "-d", "352", "-r", "1",
        "-o", "md", "-mmp", "1", "-ctk", "f16", "-ctv", "f16", "-ngl", "0", "-dev", "none",
    ]


def test_pinned_cpu_options_can_be_configured(tmp_path: Path) -> None:
    config = write_config(tmp_path, mmap="false", cpu_only="false")
    command = harness.build_command(config, harness.Case(4, 128, 32, 64, 512))
    assert command[-6:] == ["-mmp", "0", "-ctk", "f16", "-ctv", "f16"]
    assert "-ngl" not in command


def test_context_must_fit_workload(tmp_path: Path) -> None:
    config = write_config(tmp_path, context_sizes="[128]")
    with pytest.raises(harness.ConfigError, match="smaller than prompt"):
        list(harness.cases(config))


def test_successful_output_parsing() -> None:
    rows = harness.parse_llama_bench((FIXTURES / "llama_bench_success.txt").read_text(encoding="utf-8"))
    assert rows == [{
        "model_name": "deterministic-fixture", "model_size": "1.23 GiB", "backend": "CPU",
        "thread_count": 4, "prompt_tokens": 128, "generated_tokens": 32,
        "prompt_tokens_per_second": 100.0, "generation_tokens_per_second": 20.0,
        "test_identifier": "pp128+tg32",
    }]


def test_pinned_depth_suffix_is_parsed() -> None:
    output = (FIXTURES / "llama_bench_success.txt").read_text(encoding="utf-8")
    output = output.replace("pp128", "pp128 @ d832").replace("tg32", "tg32 @ d832")
    rows = harness.parse_llama_bench(output)
    assert rows[0]["prompt_tokens"] == 128
    assert rows[0]["generated_tokens"] == 32


def test_malformed_output_handling() -> None:
    with pytest.raises(ValueError, match="no llama-bench result table"):
        harness.parse_llama_bench((FIXTURES / "llama_bench_malformed.txt").read_text(encoding="utf-8"))


def test_failed_subprocess_result_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_config(tmp_path, threads="[4]", repetitions="1", warmup_runs="0")
    monkeypatch.setattr(harness, "environment_metadata", lambda root: {"timestamp_utc": "2026-01-01T00:00:00+00:00"})
    monkeypatch.setattr(harness, "execute", lambda command, timeout: {
        "stdout": "partial", "stderr": "failure", "return_code": 7,
        "elapsed_seconds": 0.25, "timed_out": False,
    })
    raw, csv_path, count = harness.run(config, tmp_path)
    record = json.loads(raw.read_text(encoding="utf-8"))
    assert record["return_code"] == 7
    assert record["stderr"] == "failure"
    assert record["command"][0] == "/opt/llama-bench"
    assert count == 0
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_and_csv_output(tmp_path: Path) -> None:
    raw_path, csv_path = tmp_path / "out.jsonl", tmp_path / "out.csv"
    records = [{"return_code": 0, "stdout": "ok"}, {"return_code": 2, "stdout": "bad"}]
    normalized = [{field: f"value-{field}" for field in harness.CSV_FIELDS}]
    harness.write_results(raw_path, csv_path, records, normalized)
    assert [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] == records
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "value-model_name"


def test_explicit_experiment_matrix_is_deduplicated() -> None:
    config = harness.load_config(Path("configs/prefill_decode_scaling.yaml"))
    unique = list(harness.cases(config))
    assert len(unique) == 13
    assert len(set(unique)) == 13
    assert sum(case.threads == 4 and case.prompt_tokens == 512 and case.generated_tokens == 64 for case in unique) == 1
