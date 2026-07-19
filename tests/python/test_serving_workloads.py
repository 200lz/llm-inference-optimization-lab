import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.serving_common import (MANIFEST_SCHEMA_VERSION, WORKLOAD_CLASSES,
    WORKLOAD_SCHEMA_VERSION, ValidationError, generate_records, read_workload, write_workload)


ROOT = Path(__file__).resolve().parents[2]


def config(workload_class="chat", seed=7, count=8, arrival=None):
    return {"schema_version": WORKLOAD_SCHEMA_VERSION, "workload_class": workload_class,
            "request_count": count, "seed": seed,
            "arrival": arrival or {"mode": "exponential", "mean_interval_us": 1000}}


def test_generation_is_deterministic_and_seed_changes_exponential() -> None:
    first, manifest = generate_records(config(seed=7))
    second, _ = generate_records(config(seed=7))
    different, _ = generate_records(config(seed=8))
    assert first == second
    assert [row["arrival_time_us"] for row in first] != [row["arrival_time_us"] for row in different]
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["request_count"] == 8 and manifest["timestamp"] is None


@pytest.mark.parametrize("workload_class", sorted(WORKLOAD_CLASSES))
def test_all_workload_classes_are_valid(workload_class: str) -> None:
    records, manifest = generate_records(config(workload_class, arrival={"mode": "fixed_interval", "interval_us": 5}))
    assert len(records) == 8
    assert len({row["request_id"] for row in records}) == 8
    assert [row["arrival_time_us"] for row in records] == sorted(row["arrival_time_us"] for row in records)
    assert manifest["workload_class"] == workload_class


def test_fixed_interval_burst_empty_and_manual() -> None:
    fixed, _ = generate_records(config(arrival={"mode": "fixed_interval", "interval_us": 11, "start_time_us": 3}))
    assert [row["arrival_time_us"] for row in fixed] == [3 + 11 * i for i in range(8)]
    burst, _ = generate_records(config("burst", arrival={"mode": "burst", "start_time_us": 9}))
    assert {row["arrival_time_us"] for row in burst} == {9}
    empty, manifest = generate_records(config(count=0, arrival={"mode": "manual", "timestamps_us": []}))
    assert empty == [] and manifest["request_count"] == 0


def test_prefix_groups_construct_identical_full_blocks() -> None:
    records, _ = generate_records(config("shared_system_prompt", arrival={"mode": "burst"}))
    assert all(row["prompt_tokens"][:64] == records[0]["prompt_tokens"][:64] for row in records)
    assert all(len(row["prompt_tokens"]) == row["prompt_token_count"] for row in records)


@pytest.mark.parametrize("change", [
    {"request_count": -1},
    {"arrival": {"mode": "fixed_interval", "interval_us": 0}},
    {"arrival": {"mode": "exponential", "mean_interval_us": 0}},
    {"arrival": {"mode": "manual", "timestamps_us": [2, 1]}},
])
def test_invalid_generation_parameters(change: dict) -> None:
    value = config(count=2)
    value.update(change)
    with pytest.raises(ValidationError):
        generate_records(value)


def test_read_workload_validation_and_line_numbers(tmp_path: Path) -> None:
    good = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "a", "arrival_time_us": 0,
            "prompt_token_count": 0, "prompt_tokens": [], "max_new_tokens": 0,
            "workload_class": "chat"}
    path = tmp_path / "workload.jsonl"
    path.write_text(json.dumps(good) + "\n{bad\n")
    with pytest.raises(ValidationError, match="JSONL line 2"):
        read_workload(path)


def test_empty_workload_file_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert read_workload(path) == []


def test_signed_int32_boundaries_and_extension_fields_round_trip(tmp_path: Path) -> None:
    record = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "bounds",
              "arrival_time_us": 0, "prompt_token_count": 2,
              "prompt_tokens": [-(2**31), 2**31 - 1], "max_new_tokens": 0,
              "workload_class": "chat", "prefix_group": "group", "deadline_us": 12,
              "metadata": {"source": "test"}}
    path = tmp_path / "bounds.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert read_workload(path) == [record]
    for bad_token in (-(2**31) - 1, 2**31):
        bad = record | {"prompt_token_count": 1, "prompt_tokens": [bad_token]}
        path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValidationError, match="32-bit|>="):
            read_workload(path)


def test_unknown_top_level_field_is_rejected(tmp_path: Path) -> None:
    record = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "a", "arrival_time_us": 0,
              "prompt_token_count": 0, "max_new_tokens": 0, "workload_class": "chat", "extra": 1}
    path = tmp_path / "unknown.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValidationError, match="unknown fields"):
        read_workload(path)


@pytest.mark.parametrize("mutation,match", [
    ({"schema_version": "future"}, "incompatible schema_version"),
    ({"arrival_time_us": -1}, "arrival_time_us"),
    ({"max_new_tokens": "1"}, "max_new_tokens"),
    ({"prompt_token_count": 2, "prompt_tokens": [1]}, "must equal"),
])
def test_record_rejections(tmp_path: Path, mutation: dict, match: str) -> None:
    record = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "a", "arrival_time_us": 0,
              "prompt_token_count": 1, "max_new_tokens": 1, "workload_class": "chat"}
    record.update(mutation)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValidationError, match=match):
        read_workload(path)


def test_duplicate_and_missing_fields(tmp_path: Path) -> None:
    record = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "a", "arrival_time_us": 0,
              "prompt_token_count": 1, "max_new_tokens": 1, "workload_class": "chat"}
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValidationError, match="duplicate request_id"):
        read_workload(duplicate)
    del record["max_new_tokens"]
    duplicate.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValidationError, match="max_new_tokens"):
        read_workload(duplicate)


def test_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "workload.jsonl"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        write_workload([], output)
    write_workload([], output, force=True)
    assert output.read_text() == ""


def test_generator_cli_refuses_overwrite(tmp_path: Path) -> None:
    generator = tmp_path / "generator.json"
    generator.write_text(json.dumps(config(count=0, arrival={"mode": "burst"})))
    output = tmp_path / "output.jsonl"
    output.write_text("existing")
    result = subprocess.run([sys.executable, str(ROOT / "benchmarks/generate_serving_workload.py"),
        str(generator), "--output", str(output)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "refusing to overwrite" in result.stderr


def test_generator_cli_is_byte_identical_and_manifest_only_overwrite_is_atomic(tmp_path: Path) -> None:
    generator = tmp_path / "generator.json"
    generator.write_text(json.dumps(config()))
    outputs = []
    for name in ("first", "second"):
        output, manifest = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.manifest.json"
        result = subprocess.run([sys.executable, str(ROOT / "benchmarks/generate_serving_workload.py"),
            str(generator), "--output", str(output), "--manifest", str(manifest)],
            cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        outputs.append((output.read_bytes(), manifest.read_bytes()))
    assert outputs[0] == outputs[1]
    blocked_output, blocked_manifest = tmp_path / "blocked.jsonl", tmp_path / "blocked.manifest.json"
    blocked_manifest.write_text("existing")
    result = subprocess.run([sys.executable, str(ROOT / "benchmarks/generate_serving_workload.py"),
        str(generator), "--output", str(blocked_output), "--manifest", str(blocked_manifest)],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and not blocked_output.exists()
