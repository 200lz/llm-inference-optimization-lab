import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.analyze_serving_results import analyze
from benchmarks.run_serving_simulation import run, runner_command
from benchmarks.run_serving_matrix import validate_matrix
from benchmarks.serving_common import (MANIFEST_SCHEMA_VERSION, ROOT, ValidationError,
    WORKLOAD_SCHEMA_VERSION, canonical_json, load_json, read_jsonl, read_workload, sha256_file,
    validate_benchmark_runner, validate_benchmark_runner_consistency, validate_native_records,
    validate_output_destination, validate_serving_config)


RUNNER = ROOT / "build/debug/serving-benchmark-runner"


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_machine_runner_is_deterministic_and_provenance_complete(tmp_path: Path) -> None:
    first = run(ROOT / "configs/serving/continuous_chat_small.json", runner=RUNNER,
                output_override=tmp_path / "first.jsonl")
    second = run(ROOT / "configs/serving/continuous_chat_small.json", runner=RUNNER,
                 output_override=tmp_path / "second.jsonl")
    first_records, second_records = read_jsonl(first), read_jsonl(second)
    assert first_records == second_records
    provenance = first_records[0]
    for field in ("git_revision", "dirty_worktree", "llama_cpp_submodule_revision",
                  "config_sha256", "workload_sha256", "seed", "python_version",
                  "simulator_schema_version", "commands"):
        assert field in provenance and provenance[field] is not None
    assert all(record.get("evidence_type") in {"simulated", "SIMULATED"} for record in first_records)
    assert provenance["benchmark_runner"]["sha256"] == sha256_file(RUNNER)
    summary, _ = analyze(first)
    assert summary["completed"] == 8


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_fcfs_work_conservation_and_policy_provenance(tmp_path: Path) -> None:
    path = run(ROOT / "configs/serving/fcfs_chat_small.json", runner=RUNNER,
               output_override=tmp_path / "fcfs.jsonl")
    rows = read_jsonl(path)
    header, native_summary = rows[0], rows[-1]
    requests = [row for row in rows if row.get("record_type") == "request"]
    assert header["execution_mode"] == "single_active_fcfs"
    assert header["scheduling_policy"] is None
    assert "DecodeFirst" not in canonical_json(header["commands"])
    assert native_summary["submitted_prompt_tokens"] == sum(row["prompt_tokens_original"] for row in requests)
    assert native_summary["scheduled_prefill_tokens"] == sum(row["prompt_tokens_scheduled"] for row in requests)
    assert native_summary["generated_tokens"] == sum(row["generated_tokens"] for row in requests)
    assert native_summary["scheduled_decode_tokens"] == sum(row["generated_tokens"] for row in requests)


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_closed_native_schema_and_cross_record_mutations_are_rejected(tmp_path: Path) -> None:
    config_path = ROOT / "configs/serving/continuous_chat_small.json"
    config = load_json(config_path)
    path = run(config_path, runner=RUNNER, output_override=tmp_path / "valid.jsonl")
    valid = read_jsonl(path)[1:]
    workload = read_workload(ROOT / config["workload"]["path"])
    validate_native_records(valid, config=config, workload_records=workload)

    def reject(mutate) -> None:
        records = copy.deepcopy(valid)
        mutate(records)
        with pytest.raises(ValidationError):
            validate_native_records(records, config=config, workload_records=workload)

    request_index = next(index for index, row in enumerate(valid) if row["record_type"] == "request")
    iteration_index = next(index for index, row in enumerate(valid) if row["record_type"] == "iteration")
    summary_index = len(valid) - 1
    mutations = [
        lambda rows: rows[request_index].pop("workload_class"),
        lambda rows: rows[request_index].__setitem__("workload_class", 7),
        lambda rows: rows[request_index].__setitem__("metadata", []),
        lambda rows: rows[request_index].__setitem__("unexpected", 1),
        lambda rows: rows[request_index].pop("decode_token_times_us"),
        lambda rows: rows[request_index].__setitem__("prompt_tokens_scheduled",
                                                      rows[request_index]["prompt_tokens_scheduled"] + 1),
        lambda rows: rows[iteration_index].pop("internal_fragmentation_tokens"),
        lambda rows: rows[iteration_index].pop("cached_blocks"),
        lambda rows: rows[iteration_index].pop("shared_referenced_blocks"),
        lambda rows: rows[iteration_index].__setitem__("unexpected", 1),
        lambda rows: rows[iteration_index].__setitem__("scheduled_prefill_tokens",
                                                        rows[iteration_index]["scheduled_prefill_tokens"] + 1),
        lambda rows: rows[iteration_index].__setitem__("scheduled_decode_tokens",
                                                        rows[iteration_index]["scheduled_decode_tokens"] + 1),
        lambda rows: rows[iteration_index].__setitem__("total_scheduled_tokens",
                                                        rows[iteration_index]["total_scheduled_tokens"] + 1),
        lambda rows: rows[iteration_index].__setitem__("kv_utilization", 1.1),
        lambda rows: rows[iteration_index].__setitem__("evicted_ids", [1, 1]),
        lambda rows: rows[summary_index].pop("peak_allocated_blocks"),
        lambda rows: rows[summary_index].pop("cached_blocks"),
        lambda rows: rows[summary_index].__setitem__("unexpected", 1),
        lambda rows: rows[summary_index].__setitem__("scheduled_prefill_tokens",
                                                      rows[summary_index]["scheduled_prefill_tokens"] + 1),
        lambda rows: rows[summary_index].__setitem__("scheduled_decode_tokens",
                                                      rows[summary_index]["scheduled_decode_tokens"] + 1),
        lambda rows: rows[summary_index].__setitem__("max_scheduled_tokens",
                                                      rows[summary_index]["max_scheduled_tokens"] + 1),
        lambda rows: rows.insert(-1, copy.deepcopy(rows[-1])),
        lambda rows: rows.insert(0, rows.pop()),
        lambda rows: rows[iteration_index]["prefill_ids"].append(999999),
        lambda rows: rows[request_index].__setitem__("metadata", {"changed": True}),
    ]
    for mutate in mutations:
        reject(mutate)

    def change_decode_owner(rows: list[dict]) -> None:
        item = next(row for row in rows if row["record_type"] == "iteration" and row["decode_ids"])
        scheduled = set(item["prefill_ids"] + item["decode_ids"])
        replacement = next(row["internal_id"] for row in rows
                           if row["record_type"] == "request" and row["internal_id"] not in scheduled)
        item["decode_ids"][0] = replacement

    reject(change_decode_owner)
    reject(lambda rows: rows[-1].__setitem__("peak_allocated_blocks", 0))

    exact_config = load_json(ROOT / "configs/serving/shared_prefix_cache_on.json")
    exact_path = run(ROOT / "configs/serving/shared_prefix_cache_on.json", runner=RUNNER,
                     output_override=tmp_path / "exact.jsonl")
    exact_rows = read_jsonl(exact_path)[1:]
    exact_workload = read_workload(ROOT / exact_config["workload"]["path"])
    exact_request = next(row for row in exact_rows if row["record_type"] == "request")
    exact_request["prompt_tokens"][0] += 1
    with pytest.raises(ValidationError, match="prompt_tokens"):
        validate_native_records(exact_rows, config=exact_config, workload_records=exact_workload)


@pytest.mark.parametrize("mutation,match", [
    (("scheduling_policy", "Unknown"), "unknown scheduling_policy"),
    (("max_num_sequences", 0), "max_num_sequences"),
    (("kv_cache", {"total_blocks": 0, "block_size_tokens": 16}), "total_blocks"),
])
def test_invalid_serving_config(tmp_path: Path, mutation, match) -> None:
    config = load_json(ROOT / "configs/serving/continuous_chat_small.json")
    config[mutation[0]] = mutation[1]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValidationError, match=match):
        validate_serving_config(config, path)


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_native_runner_nonzero_on_malformed_tsv(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("bad header\n")
    result = subprocess.run([str(RUNNER), "--input", str(bad)], text=True, capture_output=True)
    assert result.returncode != 0 and "header" in result.stderr


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_native_runner_rejects_duplicate_external_id(tmp_path: Path) -> None:
    config = load_json(ROOT / "configs/serving/continuous_chat_small.json")
    source = tmp_path / "duplicate.tsv"
    source.write_text("internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens\n"
                      "1\tr\tchat\t0\t1\t-\t1\n2\tr\tchat\t0\t1\t-\t1\n")
    completed = subprocess.run(runner_command(config, RUNNER, source, tmp_path / "out.jsonl"),
                               text=True, capture_output=True)
    assert completed.returncode != 0 and "duplicate request_id" in completed.stderr


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_disabled_exact_and_count_requests_are_equivalent(tmp_path: Path) -> None:
    config = load_json(ROOT / "configs/serving/continuous_chat_small.json")
    outputs = []
    for name, tokens in (("count", "-"), ("exact", "1,2,3,4")):
        source = tmp_path / f"{name}.tsv"
        source.write_text("internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens\n"
                          f"1\tr\tchat\t0\t4\t{tokens}\t3\n")
        output = tmp_path / f"{name}.jsonl"
        completed = subprocess.run(runner_command(config, RUNNER, source, output), text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
        outputs.append(read_jsonl(output))
    count_request = next(row for row in outputs[0] if row["record_type"] == "request")
    exact_request = next(row for row in outputs[1] if row["record_type"] == "request")
    for field in ("admitted_time_us", "first_token_time_us", "finish_time_us",
                  "prompt_tokens_scheduled", "generated_tokens"):
        assert count_request[field] == exact_request[field]


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_empty_exact_prompt_and_oversized_prompt(tmp_path: Path) -> None:
    config = load_json(ROOT / "configs/serving/continuous_chat_small.json")
    empty = tmp_path / "empty.tsv"
    empty.write_text("internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens\n"
                     "1\tempty\tchat\t0\t0\t\t1\n")
    output = tmp_path / "empty.jsonl"
    assert subprocess.run(runner_command(config, RUNNER, empty, output)).returncode == 0
    oversized = tmp_path / "oversized.tsv"
    oversized.write_text("internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens\n"
                         "1\tlarge\tchat\t0\t513\t-\t1\n")
    result = subprocess.run(runner_command(config, RUNNER, oversized, tmp_path / "oversized.jsonl"),
                            text=True, capture_output=True)
    assert result.returncode != 0 and "max_batched_tokens" in result.stderr


def normalized_config(name: str) -> dict:
    value = load_json(ROOT / f"configs/serving/{name}.json")
    for key in ("name", "output_path"):
        value.pop(key)
    return value


def test_comparison_configs_change_only_intended_dimensions() -> None:
    for names in (("fcfs_chat_small", "continuous_chat_small", "fcfs_chat_continuous"),
                  ("single_active_fcfs_mixed", "decode_first", "fcfs_mixed"),
                  ("fcfs_overload", "overload_decode_first", "mixed_overload_fcfs_mixed")):
        schedulers = [normalized_config(name) for name in names]
        for value in schedulers:
            value["execution_mode"] = "same"
            value["scheduling_policy"] = "same"
        assert schedulers[0] == schedulers[1] == schedulers[2]
    off, on = normalized_config("shared_prefix_cache_off"), normalized_config("shared_prefix_cache_on")
    off["prefix_cache"]["enabled"] = on["prefix_cache"]["enabled"]
    assert off == on
    small, medium, large = map(normalized_config, ("kv_small", "kv_medium", "kv_large"))
    for value in (small, medium, large): value["kv_cache"]["total_blocks"] = 1
    assert small == medium == large
    blocks = [normalized_config(f"block_size_{size}") for size in (8, 16, 32)]
    for value in blocks: value["kv_cache"]["block_size_tokens"] = 1
    assert blocks[0] == blocks[1] == blocks[2]
    loads = [normalized_config(name) for name in ("mixed_low_load", "mixed_medium_load", "mixed_overload")]
    for value in loads:
        value["workload"] = {"path": "same", "manifest_path": "same"}
    assert loads[0] == loads[1] == loads[2]
    workload_records = [read_jsonl(ROOT / f"benchmarks/serving/workloads/mixed_{load}.jsonl")
                        for load in ("low", "medium", "overload")]
    stripped = [[{k: v for k, v in row.items() if k not in {"arrival_time_us", "request_id"}} for row in rows]
                for rows in workload_records]
    assert stripped[0] == stripped[1] == stripped[2]
    assert [[row["request_id"] for row in rows] for rows in workload_records].count(
        [row["request_id"] for row in workload_records[0]]) == 3


def test_output_roots_and_matrix_schema_are_strict(tmp_path: Path) -> None:
    for relative in ("README.md", "src/result.jsonl", "docs/result.jsonl", ".github/result.jsonl",
                     "CMakeLists.txt", "../escape.jsonl"):
        with pytest.raises(ValidationError):
            validate_output_destination(ROOT / relative)
    with pytest.raises(ValidationError):
        validate_output_destination(Path("/opt/outside.jsonl"))
    assert validate_output_destination(ROOT / ".artifacts/serving/test/result.jsonl")
    with pytest.raises(ValidationError):
        validate_output_destination(ROOT / "results/serving/raw/reference.jsonl")
    assert validate_output_destination(ROOT / "results/serving/raw/reference.jsonl", update_reference=True)
    for name in ("matrix_small", "matrix_extended"):
        matrix = load_json(ROOT / f"configs/serving/{name}.json")
        configs = validate_matrix(matrix, ROOT / f"configs/serving/{name}.json")
        assert matrix["max_default_runs"] == len(configs)
        if name == "matrix_extended":
            assert {"fcfs_chat_small", "continuous_chat_small", "fcfs_chat_continuous"} <= set(
                matrix["comparisons"]["scheduler"]["configs"])


def test_matrix_run_limit_duplicate_names_and_default_output_policy(tmp_path: Path) -> None:
    extended = ROOT / "configs/serving/matrix_extended.json"
    limited = subprocess.run([sys.executable, str(ROOT / "benchmarks/run_serving_matrix.py"),
        str(extended), "--dry-run"], cwd=ROOT, text=True, capture_output=True)
    assert limited.returncode != 0 and "raise --max-runs" in limited.stderr
    matrix = load_json(ROOT / "configs/serving/matrix_small.json")
    matrix["configs"].append(matrix["configs"][0])
    with pytest.raises(ValidationError, match="unique"):
        validate_matrix(matrix, tmp_path / "duplicate.json")


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_default_matrix_run_does_not_touch_tracked_files(tmp_path: Path) -> None:
    before = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout
    reference_hash = sha256_file(ROOT / "results/serving/raw/fcfs_chat_small.jsonl")
    result = subprocess.run([sys.executable, str(ROOT / "benchmarks/run_serving_matrix.py"),
        str(ROOT / "configs/serving/matrix_small.json"), "--select", "fcfs_chat_small",
        "--output-root", str(tmp_path / "matrix"), "--force"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    after = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                           capture_output=True, check=True).stdout
    assert after == before
    assert sha256_file(ROOT / "results/serving/raw/fcfs_chat_small.jsonl") == reference_hash


def _write_workload_bundle(path: Path, records: list[dict], seed: int = 20260719) -> tuple[Path, Path]:
    workload = path / "workload.jsonl"
    workload.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest = path / "workload.manifest.json"
    manifest.write_text(json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION,
        "workload_schema_version": WORKLOAD_SCHEMA_VERSION, "workload_class": "chat",
        "request_count": len(records), "seed": seed,
        "generator_config": {"schema_version": WORKLOAD_SCHEMA_VERSION, "workload_class": "chat",
            "request_count": len(records), "seed": seed, "arrival": {"mode": "burst"}},
        "generator": "test", "timestamp": None, "workload_sha256": sha256_file(workload)}))
    return workload, manifest


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_same_arrival_ties_use_external_id_and_metadata_round_trips(tmp_path: Path) -> None:
    base = {"schema_version": WORKLOAD_SCHEMA_VERSION, "arrival_time_us": 0,
            "prompt_token_count": 2, "max_new_tokens": 2, "workload_class": "chat"}
    records = [base | {"request_id": "2", "prefix_group": "g", "deadline_us": 50,
                       "metadata": {"rank": 2}},
               base | {"request_id": "10", "prefix_group": "g", "deadline_us": 40,
                       "metadata": {"rank": 1}}]
    native_streams = []
    for index, ordered in enumerate((records, list(reversed(records)))):
        directory = tmp_path / str(index); directory.mkdir()
        workload, manifest = _write_workload_bundle(directory, ordered)
        result = run(ROOT / "configs/serving/continuous_chat_small.json", runner=RUNNER,
                     output_override=directory / "result.jsonl", workload_override=workload,
                     manifest_override=manifest)
        rows = read_jsonl(result)[1:]
        native_streams.append(rows)
        requests = [row for row in rows if row.get("record_type") == "request"]
        assert [row["request_id"] for row in requests] == ["10", "2"]
        assert requests[0]["metadata"] == {"rank": 1}
        assert requests[0]["deadline_us"] == 40 and requests[0]["prefix_group"] == "g"
    assert native_streams[0] == native_streams[1]


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_empty_continuous_workload_has_closed_terminal_summary(tmp_path: Path) -> None:
    workload, manifest = _write_workload_bundle(tmp_path, [])
    result = run(ROOT / "configs/serving/continuous_chat_small.json", runner=RUNNER,
                 output_override=tmp_path / "empty-result.jsonl", workload_override=workload,
                 manifest_override=manifest)
    rows = read_jsonl(result)
    assert [row["record_type"] for row in rows] == ["run", "summary"]
    analyzed, _ = analyze(result)
    assert analyzed["submitted"] == 0 and analyzed["completed"] == 0
    assert analyzed["itl_available"] is False


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
@pytest.mark.parametrize("field", ["seed", "workload_class", "request_count", "generator_config", "hash"])
def test_manifest_mismatches_are_rejected(tmp_path: Path, field: str) -> None:
    record = {"schema_version": WORKLOAD_SCHEMA_VERSION, "request_id": "r", "arrival_time_us": 0,
              "prompt_token_count": 1, "max_new_tokens": 1, "workload_class": "chat"}
    workload, manifest_path = _write_workload_bundle(tmp_path, [record])
    manifest = load_json(manifest_path)
    if field == "seed": manifest["seed"] += 1
    elif field == "workload_class": manifest["workload_class"] = "burst"
    elif field == "request_count": manifest["request_count"] += 1
    elif field == "generator_config": manifest["generator_config"]["arrival"] = "invalid"
    else: manifest["workload_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError):
        run(ROOT / "configs/serving/continuous_chat_small.json", runner=RUNNER,
            output_override=tmp_path / "result.jsonl", workload_override=workload,
            manifest_override=manifest_path)


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_scheduler_comparison_conserves_equivalent_work(tmp_path: Path) -> None:
    paths = []
    for name in ("single_active_fcfs_mixed", "decode_first", "fcfs_mixed"):
        paths.append(run(ROOT / f"configs/serving/{name}.json", runner=RUNNER,
                         output_override=tmp_path / f"{name}.jsonl"))
    totals = []
    for path in paths:
        rows = [r for r in read_jsonl(path) if r.get("record_type") == "request"]
        totals.append((len(rows), sum(r["prompt_tokens_original"] for r in rows),
                       sum(r["generated_tokens"] for r in rows)))
    assert totals[0] == totals[1] == totals[2]


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_demo_uses_the_generated_workload_and_leaves_tracked_state_unchanged() -> None:
    before = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout
    output = ROOT / ".artifacts/serving/pytest-demo"
    environment = os.environ | {"PYTHON": sys.executable, "SERVING_DEMO_DIR": str(output)}
    completed = subprocess.run([str(ROOT / "scripts/run_serving_demo.sh")], cwd=ROOT,
                               env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    header = read_jsonl(output / "result.jsonl")[0]
    assert header["workload_sha256"] == sha256_file(output / "chat.jsonl")
    after = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                           capture_output=True, check=True).stdout
    assert after == before


@pytest.mark.skipif(not RUNNER.exists(), reason="Debug runner is not built")
def test_demo_missing_python_has_actionable_error() -> None:
    completed = subprocess.run([str(ROOT / "scripts/run_serving_demo.sh")], cwd=ROOT,
        env=os.environ | {"PYTHON": "/definitely/missing/python"}, text=True, capture_output=True)
    assert completed.returncode == 2
    assert "python3 -m venv .venv" in completed.stderr and "set PYTHON=" in completed.stderr


def test_checked_in_results_are_portable_and_schema_valid() -> None:
    result_root = ROOT / "results/serving"
    candidates = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard",
        "results/serving"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    for relative in candidates:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text and "/Users/" not in text
        assert "/tmp/" not in text and "/var/tmp/" not in text
        assert re.search(r"(?:^|[\"'])[A-Za-z]:[\\/]", text) is None
    manifests = [load_json(result_root / name) for name in
                 ("command_manifest.json", "extended_command_manifest.json")]
    assert all(manifest["evidence_type"] == "SIMULATED" for manifest in manifests)
    run_items = {item["result"]: item for manifest in manifests for item in manifest["runs"]}
    assert set(run_items) == {path.relative_to(ROOT).as_posix()
                              for path in (result_root / "raw").glob("*.jsonl")}
    runner_provenance = []
    for name, manifest in zip(("command_manifest.json", "extended_command_manifest.json"), manifests):
        if "benchmark_runner" in manifest:
            runner_provenance.append((name, manifest["benchmark_runner"]))
        for index, item in enumerate(manifest["runs"]):
            if "benchmark_runner" in item:
                runner_provenance.append((f"{name} run {index}", item["benchmark_runner"]))
    for item in run_items.values():
        result = ROOT / item["result"]
        assert result.is_file(), result
        summary, provenance = analyze(result)
        assert summary["submitted"] == len([row for row in read_jsonl(result)
                                            if row.get("record_type") == "request"])
        assert provenance["dirty_worktree"] is True
        runner_provenance.append((item["result"], provenance["benchmark_runner"]))
        workload = ROOT / provenance["normalized_config"]["workload"]["path"]
        assert provenance["workload_sha256"] == sha256_file(workload)
        assert provenance["workload_manifest"]["workload_sha256"] == provenance["workload_sha256"]
    validate_benchmark_runner_consistency(runner_provenance)


def test_recorded_runner_provenance_is_portable_without_local_binary() -> None:
    value = {"path": "build/missing/serving-benchmark-runner", "sha256": "a" * 64}
    assert not (ROOT / value["path"]).exists()
    assert validate_benchmark_runner(value, context="reference") == (value["path"], value["sha256"])


@pytest.mark.parametrize("path", ["/opt/serving-benchmark-runner", "../serving-benchmark-runner"])
def test_recorded_runner_path_rejects_nonportable_values(path: str) -> None:
    with pytest.raises(ValidationError, match="normalized repository-relative"):
        validate_benchmark_runner({"path": path, "sha256": "a" * 64}, context="reference")


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_recorded_runner_sha_rejects_malformed_values(digest: str) -> None:
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        validate_benchmark_runner(
            {"path": "build/debug/serving-benchmark-runner", "sha256": digest},
            context="reference")


@pytest.mark.parametrize("changed", [
    {"path": "build/release/serving-benchmark-runner", "sha256": "a" * 64},
    {"path": "build/debug/serving-benchmark-runner", "sha256": "b" * 64},
])
def test_recorded_runner_provenance_must_be_consistent(changed: dict[str, str]) -> None:
    original = {"path": "build/debug/serving-benchmark-runner", "sha256": "a" * 64}
    with pytest.raises(ValidationError, match="disagrees"):
        validate_benchmark_runner_consistency([
            ("raw reference", original), ("command_manifest.json", changed)])
