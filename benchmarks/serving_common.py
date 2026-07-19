"""Shared, dependency-free utilities for Phase S6 serving experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


WORKLOAD_SCHEMA_VERSION = "serving-workload-v1"
MANIFEST_SCHEMA_VERSION = "serving-workload-manifest-v1"
CONFIG_SCHEMA_VERSION = "serving-benchmark-config-v1"
RESULT_SCHEMA_VERSION = "serving-result-v1"
SIMULATOR_SCHEMA_VERSION = "serving-simulator-v2"
SUMMARY_SCHEMA_VERSION = "serving-summary-v1"
MATRIX_SCHEMA_VERSION = "serving-matrix-v1"
WORKLOAD_CLASSES = {
    "chat", "shared_system_prompt", "coding_agent", "mixed", "overload", "burst"
}
ARRIVAL_MODELS = {"fixed_interval", "burst", "exponential", "manual"}
ROOT = Path(__file__).resolve().parents[1]
NORMAL_OUTPUT_ROOTS = (ROOT / "build/serving-results", ROOT / ".artifacts/serving")
REFERENCE_OUTPUT_ROOT = ROOT / "results/serving"

RUN_RECORD_FIELDS = frozenset({
    "record_type", "schema_version", "evidence_type", "label", "git_revision", "branch",
    "dirty_worktree", "llama_cpp_submodule_revision", "normalized_config", "config_sha256",
    "workload_manifest", "workload_sha256", "seed", "simulator_schema_version",
    "result_schema_version", "python_version", "python_executable", "cmake_build_type",
    "benchmark_runner", "commands", "timestamp", "execution_mode", "scheduling_policy",
})
MANIFEST_RECORD_FIELDS = frozenset({
    "schema_version", "workload_schema_version", "workload_class", "request_count", "seed",
    "generator_config", "generator", "timestamp", "workload_sha256",
})
REQUEST_RECORD_FIELDS = frozenset({
    "record_type", "schema_version", "evidence_type", "internal_id", "request_id",
    "workload_class", "prefix_group", "deadline_us", "metadata", "prompt_tokens", "arrival_time_us",
    "admitted_time_us", "first_token_time_us", "finish_time_us", "queue_delay_us",
    "ttft_us", "end_to_end_latency_us", "tpot_us", "prompt_tokens_original",
    "prompt_tokens_matched", "prompt_tokens_scheduled", "generated_tokens", "max_new_tokens",
    "final_state", "reason", "decode_token_times_us", "inter_token_latencies_us",
})
ITERATION_RECORD_FIELDS = frozenset({
    "record_type", "schema_version", "evidence_type", "iteration_number", "start_time_us",
    "end_time_us", "policy", "prefill_ids", "decode_ids", "deferred",
    "scheduled_sequences", "scheduled_prefill_tokens", "scheduled_decode_tokens",
    "total_scheduled_tokens", "kv_allocated", "kv_free", "kv_utilization",
    "represented_kv_tokens", "internal_fragmentation_tokens", "cached_blocks",
    "shared_referenced_blocks", "prefix_hits", "prefix_misses", "matched_tokens",
    "evicted_ids", "stall",
})
SUMMARY_RECORD_FIELDS = frozenset({
    "record_type", "schema_version", "evidence_type", "execution_mode", "scheduling_policy",
    "run_status", "submitted", "completed", "cancelled", "rejected", "stalled",
    "makespan_us", "scheduling_iterations", "nonempty_batches", "idle_iterations",
    "total_scheduled_sequences", "average_batch_size", "max_batch_size",
    "max_scheduled_tokens", "deferred_requests", "current_allocated_blocks",
    "current_free_blocks", "peak_allocated_blocks", "kv_current_utilization",
    "kv_peak_utilization", "represented_kv_tokens", "internal_fragmentation_tokens",
    "cached_blocks", "shared_referenced_blocks", "kv_allocation_failures",
    "kv_capacity_deferrals", "prefix_lookups", "prefix_lookup_hits", "prefix_lookup_misses",
    "matched_prefix_blocks", "cache_eligible_prompt_tokens", "collision_verifications",
    "matched_prefix_tokens", "prefix_token_hit_rate", "saved_simulated_prefill_tokens",
    "eviction_count", "submitted_prompt_tokens", "scheduled_prefill_tokens",
    "generated_tokens", "scheduled_decode_tokens",
})


class ValidationError(ValueError):
    """Input is structurally invalid or incompatible with this S6 schema."""


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if value < minimum:
        raise ValidationError(f"{name} must be >= {minimum}")
    return value


def _require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValidationError(f"{name} must be a {'possibly empty ' if allow_empty else ''}string")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: top-level JSON must be an object")
    return value


def _arrival_times(config: dict[str, Any], count: int, seed: int) -> list[int]:
    arrival = config.get("arrival", {})
    if not isinstance(arrival, dict):
        raise ValidationError("arrival must be an object")
    mode = _require_str(arrival.get("mode"), "arrival.mode")
    if mode not in ARRIVAL_MODELS:
        raise ValidationError(f"arrival.mode must be one of {sorted(ARRIVAL_MODELS)}")
    start = _require_int(arrival.get("start_time_us", 0), "arrival.start_time_us")
    if mode == "burst":
        return [start] * count
    if mode == "fixed_interval":
        interval = _require_int(arrival.get("interval_us"), "arrival.interval_us", minimum=1)
        return [start + index * interval for index in range(count)]
    if mode == "manual":
        times = arrival.get("timestamps_us")
        if not isinstance(times, list) or len(times) != count:
            raise ValidationError("arrival.timestamps_us must have request_count entries")
        checked = [_require_int(value, f"arrival.timestamps_us[{index}]") for index, value in enumerate(times)]
        if checked != sorted(checked):
            raise ValidationError("manual arrival timestamps must be non-decreasing")
        return checked
    mean = _require_int(arrival.get("mean_interval_us"), "arrival.mean_interval_us", minimum=1)
    rng = random.Random(seed)
    result: list[int] = []
    current = start
    for index in range(count):
        if index:
            # Inverse-CDF exponential sample: floor(-mean * log1p(-U)).
            delta = math.floor(-mean * math.log1p(-rng.random()))
            current += delta
        result.append(current)
    return result


def _shared_tokens(group: int, length: int, base: int) -> list[int]:
    return [base + group * 10_000 + index for index in range(length)]


def generate_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = {"schema_version", "workload_class", "request_count", "seed", "arrival"}
    unknown = set(config) - allowed
    if unknown:
        raise ValidationError(f"unknown generator fields: {sorted(unknown)}")
    if config.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise ValidationError(
            f"schema_version must be {WORKLOAD_SCHEMA_VERSION!r}, got {config.get('schema_version')!r}"
        )
    workload_class = _require_str(config.get("workload_class"), "workload_class")
    if workload_class not in WORKLOAD_CLASSES:
        raise ValidationError(f"unknown workload_class {workload_class!r}")
    count = _require_int(config.get("request_count"), "request_count")
    seed = _require_int(config.get("seed"), "seed")
    arrivals = _arrival_times(config, count, seed)
    rng = random.Random(seed ^ 0x5EEDC0DE)
    records: list[dict[str, Any]] = []
    for index, arrival in enumerate(arrivals):
        record: dict[str, Any] = {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "request_id": f"{workload_class}-{index:04d}",
            "arrival_time_us": arrival,
            "prompt_token_count": 0,
            "max_new_tokens": 0,
            "workload_class": workload_class,
        }
        if workload_class == "chat":
            record.update(prompt_token_count=rng.randint(12, 80), max_new_tokens=rng.randint(16, 64))
        elif workload_class == "shared_system_prompt":
            shared = _shared_tokens(0, 64, 10_000)
            tokens = shared + _shared_tokens(index, 8 + index % 5, 20_000)
            record.update(prompt_token_count=len(tokens), prompt_tokens=tokens,
                          max_new_tokens=16 + index % 17, prefix_group="shared-system-v1")
        elif workload_class == "coding_agent":
            group = index % 2
            shared = _shared_tokens(group, 128, 30_000)
            tokens = shared + _shared_tokens(index, 24 + index % 17, 50_000)
            record.update(prompt_token_count=len(tokens), prompt_tokens=tokens,
                          max_new_tokens=64 + index % 65, prefix_group=f"coding-tools-{group}")
        elif workload_class == "burst":
            record.update(prompt_token_count=16 + (index % 4) * 8,
                          max_new_tokens=8 + index % 8)
        elif workload_class == "overload":
            record.update(prompt_token_count=48 + index % 33,
                          max_new_tokens=48 + index % 49)
        else:  # mixed
            kind = index % 4
            if kind == 0:
                record.update(prompt_token_count=16 + index % 48, max_new_tokens=16 + index % 24)
            elif kind == 1:
                record.update(prompt_token_count=192 + index % 65, max_new_tokens=12 + index % 12)
            elif kind == 2:
                shared = _shared_tokens(0, 96, 70_000)
                tokens = shared + _shared_tokens(index, 16 + index % 9, 80_000)
                record.update(prompt_token_count=len(tokens), prompt_tokens=tokens,
                              max_new_tokens=48 + index % 49, prefix_group="mixed-coding")
            else:
                record.update(prompt_token_count=24 + index % 40, max_new_tokens=80 + index % 49)
        records.append(validate_workload_record(record, line_number=index + 1))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload_schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_class": workload_class,
        "request_count": count,
        "seed": seed,
        "generator_config": config,
        "generator": "benchmarks/generate_serving_workload.py",
        "timestamp": None,
    }
    return records, manifest


def validate_workload_record(record: Any, *, line_number: int) -> dict[str, Any]:
    prefix = f"JSONL line {line_number}: "
    try:
        if not isinstance(record, dict):
            raise ValidationError("record must be an object")
        allowed = {"schema_version", "request_id", "arrival_time_us", "prompt_token_count",
                   "prompt_tokens", "max_new_tokens", "workload_class", "prefix_group",
                   "deadline_us", "metadata"}
        unknown = set(record) - allowed
        if unknown:
            raise ValidationError(f"unknown fields: {sorted(unknown)}; put extensions under metadata")
        if record.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
            raise ValidationError(
                f"incompatible schema_version {record.get('schema_version')!r}; expected {WORKLOAD_SCHEMA_VERSION!r}"
            )
        _require_str(record.get("request_id"), "request_id")
        _require_int(record.get("arrival_time_us"), "arrival_time_us")
        prompt_count = _require_int(record.get("prompt_token_count"), "prompt_token_count")
        _require_int(record.get("max_new_tokens"), "max_new_tokens")
        workload_class = _require_str(record.get("workload_class"), "workload_class")
        if workload_class not in WORKLOAD_CLASSES:
            raise ValidationError(f"unknown workload_class {workload_class!r}")
        if "prompt_tokens" in record:
            tokens = record["prompt_tokens"]
            if not isinstance(tokens, list):
                raise ValidationError("prompt_tokens must be an array")
            for index, token in enumerate(tokens):
                value = _require_int(token, f"prompt_tokens[{index}]", minimum=-(2**31))
                if value > 2**31 - 1:
                    raise ValidationError(f"prompt_tokens[{index}] exceeds signed 32-bit range")
            if len(tokens) != prompt_count:
                raise ValidationError("prompt_token_count must equal len(prompt_tokens)")
        if "prefix_group" in record:
            _require_str(record["prefix_group"], "prefix_group")
        if "deadline_us" in record:
            _require_int(record["deadline_us"], "deadline_us")
        if "metadata" in record and not isinstance(record["metadata"], dict):
            raise ValidationError("metadata must be an object")
        return record
    except ValidationError as error:
        raise ValidationError(prefix + str(error)) from error


def read_workload(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValidationError(f"cannot read workload {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValidationError(f"JSONL line {line_number}: blank lines are not allowed")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"JSONL line {line_number}: malformed JSON: {error.msg}") from error
        record = validate_workload_record(value, line_number=line_number)
        if record["request_id"] in seen:
            raise ValidationError(f"JSONL line {line_number}: duplicate request_id {record['request_id']!r}")
        seen.add(record["request_id"])
        records.append(record)
    return records


def write_workload(records: Iterable[dict[str, Any]], path: Path, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; use --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def resolve_repo_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValidationError("paths in checked-in serving configs must be relative")
    candidate = (ROOT / path).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValidationError(f"path escapes repository: {value}")
    return candidate


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path) -> bool:
    current = path
    while current != current.parent:
        if current.is_symlink():
            return True
        if current == ROOT or current == Path(tempfile.gettempdir()):
            break
        current = current.parent
    return False


def validate_output_destination(path: Path, *, update_reference: bool = False) -> Path:
    """Resolve and authorize generated-output paths without following escapes."""
    resolved = path.resolve()
    roots = list(NORMAL_OUTPUT_ROOTS) + [Path(tempfile.gettempdir()).resolve()]
    if update_reference:
        roots.append(REFERENCE_OUTPUT_ROOT)
    safe_roots = [root for root in roots if not _has_symlink_component(root)]
    if not any(_within(resolved, root) for root in safe_roots):
        allowed = ", ".join(str(root.relative_to(ROOT)) for root in NORMAL_OUTPUT_ROOTS)
        if update_reference:
            allowed += ", results/serving"
        raise ValidationError(f"output path {path} is outside approved roots: {allowed}, or system temporary directory")
    return resolved


def portable_path(value: str | Path) -> str:
    """Return stable, repository-relative or symbolic portable provenance."""
    text = os.fspath(value)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return re.split(r"[\\/]", text)[-1]
    if text in {"<temporary-input.tsv>", "<temporary-output.jsonl>"}:
        return text
    path = Path(text)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        pass
    if path.resolve() == Path(sys.executable).resolve():
        return ".venv/bin/python" if _within(path, ROOT / ".venv") else path.name
    if _within(path, Path(tempfile.gettempdir())):
        if path.suffix == ".tsv":
            return "<temporary-input.tsv>"
        if path.suffix == ".jsonl":
            return "<temporary-output.jsonl>"
        return f"<temporary>/{path.name}"
    return path.name if path.is_absolute() else path.as_posix()


def normalize_command(command: list[str]) -> list[str]:
    result: list[str] = []
    for value in command:
        if value.startswith("-") or not ("/" in value or "\\" in value):
            result.append(value)
        else:
            result.append(portable_path(value))
    return result


def validate_serving_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    required = {"schema_version", "name", "evidence_type", "execution_mode", "scheduling_policy",
                "max_num_sequences", "max_batched_tokens", "kv_cache", "prefix_cache", "backend",
                "slos", "workload", "output_path", "run_seed"}
    missing = required - set(config)
    if missing:
        raise ValidationError(f"{path}: missing serving config fields: {sorted(missing)}")
    unknown = set(config) - required
    if unknown:
        raise ValidationError(f"{path}: unknown serving config fields: {sorted(unknown)}")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValidationError(f"{path}: incompatible config schema_version")
    _require_str(config["name"], "name")
    if config["evidence_type"] != "simulated":
        raise ValidationError("evidence_type must be simulated")
    if config["execution_mode"] not in {"single_active_fcfs", "continuous_batching"}:
        raise ValidationError("unknown execution_mode")
    if config["scheduling_policy"] not in {"FCFS", "DecodeFirst", "FcfsMixed"}:
        raise ValidationError("unknown scheduling_policy")
    if config["execution_mode"] == "single_active_fcfs" and config["scheduling_policy"] != "FCFS":
        raise ValidationError("single_active_fcfs requires scheduling_policy FCFS")
    if config["execution_mode"] == "continuous_batching" and config["scheduling_policy"] == "FCFS":
        raise ValidationError("continuous_batching requires DecodeFirst or FcfsMixed")
    _require_int(config["max_num_sequences"], "max_num_sequences", minimum=1)
    _require_int(config["max_batched_tokens"], "max_batched_tokens", minimum=1)
    kv = config["kv_cache"]
    if not isinstance(kv, dict) or set(kv) != {"total_blocks", "block_size_tokens"}:
        raise ValidationError("kv_cache requires only total_blocks and block_size_tokens")
    _require_int(kv["total_blocks"], "kv_cache.total_blocks", minimum=1)
    _require_int(kv["block_size_tokens"], "kv_cache.block_size_tokens", minimum=1)
    prefix = config["prefix_cache"]
    if not isinstance(prefix, dict) or set(prefix) != {"enabled", "namespace", "salt"}:
        raise ValidationError("prefix_cache requires enabled, namespace, and salt")
    if not isinstance(prefix["enabled"], bool):
        raise ValidationError("prefix_cache.enabled must be boolean")
    _require_str(prefix["namespace"], "prefix_cache.namespace", allow_empty=True)
    _require_str(prefix["salt"], "prefix_cache.salt", allow_empty=True)
    backend_keys = {"prefill_base_us", "prefill_per_token_us", "prefill_per_active_sequence_us",
                    "decode_base_us", "decode_per_active_sequence_us", "batch_base_us",
                    "batch_prefill_per_token_us", "batch_decode_per_sequence_us",
                    "batch_active_sequence_overhead_us"}
    if not isinstance(config["backend"], dict) or set(config["backend"]) != backend_keys:
        raise ValidationError(f"backend must contain exactly {sorted(backend_keys)}")
    for key in backend_keys:
        _require_int(config["backend"][key], f"backend.{key}")
    if not isinstance(config["slos"], dict) or not set(config["slos"]) <= {
        "ttft_slo_us", "tpot_slo_us", "e2e_slo_us"
    }:
        raise ValidationError("slos contains unknown thresholds")
    for key, value in config["slos"].items():
        _require_int(value, f"slos.{key}")
    if not isinstance(config["workload"], dict) or set(config["workload"]) != {"path", "manifest_path"}:
        raise ValidationError("workload requires path and manifest_path")
    resolve_repo_path(_require_str(config["workload"]["path"], "workload.path"), path)
    resolve_repo_path(_require_str(config["workload"]["manifest_path"], "workload.manifest_path"), path)
    resolve_repo_path(_require_str(config["output_path"], "output_path"), path)
    _require_int(config["run_seed"], "run_seed")
    return config


def nearest_rank(values: Iterable[int | float], percentile: float) -> int | float | None:
    if not (0 < percentile <= 1):
        raise ValidationError("percentile must be in (0, 1]")
    samples = sorted(values)
    if not samples:
        return None
    rank = max(1, min(len(samples), math.ceil(percentile * len(samples))))
    return samples[rank - 1]


def request_metrics(record: dict[str, Any]) -> dict[str, int | float | None]:
    arrival = record["arrival_time_us"]
    admitted = record.get("admitted_time_us")
    first = record.get("first_token_time_us")
    finish = record.get("finish_time_us")
    generated = record["generated_tokens"]
    return {
        "queue_delay_us": None if admitted is None else admitted - arrival,
        "ttft_us": None if generated == 0 or first is None else first - arrival,
        "e2e_latency_us": None if finish is None else finish - arrival,
        "tpot_us": None if generated < 2 or first is None or finish is None
        else (finish - first) / (generated - 1),
    }


def safe_rate(numerator: int | float, duration_us: int) -> float | None:
    if duration_us <= 0:
        return None
    # Divide before multiplying to avoid constructing a large integer product.
    return float(numerator) / (float(duration_us) / 1_000_000.0)


def validate_workload_manifest(manifest: dict[str, Any], records: list[dict[str, Any]],
                               workload_path: Path, *, expected_seed: int | None = None) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValidationError("incompatible workload manifest schema")
    expected_schema = records[0]["schema_version"] if records else WORKLOAD_SCHEMA_VERSION
    if manifest.get("workload_schema_version") != expected_schema:
        raise ValidationError("workload manifest schema mismatch")
    if manifest.get("request_count") != len(records):
        raise ValidationError("workload manifest request_count mismatch")
    if manifest.get("workload_sha256") != sha256_file(workload_path):
        raise ValidationError("workload manifest SHA-256 mismatch")
    classes = {record["workload_class"] for record in records}
    manifest_class = manifest.get("workload_class")
    if classes and classes != {manifest_class}:
        raise ValidationError("workload manifest workload_class mismatch")
    generator_config = manifest.get("generator_config")
    if not isinstance(generator_config, dict):
        raise ValidationError("workload manifest generator_config must be an object")
    for field in ("schema_version", "workload_class", "request_count", "seed"):
        expected = manifest.get(field if field != "schema_version" else "workload_schema_version")
        if generator_config.get(field) != expected:
            raise ValidationError(f"workload manifest generator_config {field} mismatch")
    if manifest.get("seed") != generator_config.get("seed"):
        raise ValidationError("workload manifest seed mismatch")
    if expected_seed is not None and manifest.get("seed") != expected_seed:
        raise ValidationError("workload manifest seed does not match serving run_seed")
    arrival = generator_config.get("arrival")
    if not isinstance(arrival, dict) or arrival.get("mode") not in ARRIVAL_MODELS:
        raise ValidationError("workload manifest generator arrival configuration is invalid")
    generator = _require_str(manifest.get("generator"), "workload manifest generator")
    if [record["arrival_time_us"] for record in records] != _arrival_times(
            generator_config, len(records), manifest["seed"]):
        raise ValidationError("workload manifest generator arrival parameters mismatch")
    if generator == "benchmarks/generate_serving_workload.py":
        regenerated, _ = generate_records(generator_config)
        if regenerated != records:
            raise ValidationError("workload records do not match generator_config")


def _field_int(record: dict[str, Any], name: str, context: str, *, nullable: bool = False) -> int | None:
    value = record.get(name)
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{context}: {name} must be {'null or ' if nullable else ''}an integer")
    if value < 0:
        raise ValidationError(f"{context}: {name} must be non-negative")
    return value


def _exact_number(actual: Any, expected: int | float | None, name: str, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValidationError(f"{context}: {name} must be null")
    elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
        raise ValidationError(f"{context}: contradictory {name}; expected {expected!r}")


def _closed_fields(record: dict[str, Any], expected: frozenset[str], context: str) -> None:
    missing = expected - set(record)
    extra = set(record) - expected
    if missing:
        raise ValidationError(f"{context}: missing field {sorted(missing)[0]!r}")
    if extra:
        raise ValidationError(f"{context}: unknown field {sorted(extra)[0]!r}")


def _nullable_str(value: Any, name: str, context: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValidationError(f"{context}: {name} must be null or a string")


def _number_in_unit_interval(value: Any, name: str, context: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError(f"{context}: {name} must be {'null or ' if nullable else ''}a finite number in [0,1]")


def validate_run_record(record: dict[str, Any], *, line_number: int = 1) -> None:
    context = f"run record at result line {line_number}"
    _closed_fields(record, RUN_RECORD_FIELDS, context)
    if record["record_type"] != "run" or record["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValidationError(f"{context}: incompatible run record/schema")
    if record["evidence_type"] != "simulated" or record["label"] != "SIMULATED":
        raise ValidationError(f"{context}: evidence label must be simulated/SIMULATED")
    for name in ("git_revision", "branch", "llama_cpp_submodule_revision", "cmake_build_type", "timestamp"):
        _nullable_str(record[name], name, context)
    if not isinstance(record["dirty_worktree"], bool):
        raise ValidationError(f"{context}: dirty_worktree must be boolean")
    for name in ("config_sha256", "workload_sha256"):
        if not isinstance(record[name], str) or re.fullmatch(r"[0-9a-f]{64}", record[name]) is None:
            raise ValidationError(f"{context}: {name} must be a lowercase SHA-256")
    if not isinstance(record["normalized_config"], dict) or not isinstance(record["workload_manifest"], dict):
        raise ValidationError(f"{context}: normalized_config/workload_manifest must be objects")
    validate_serving_config(record["normalized_config"], ROOT / "result-provenance.json")
    if record["config_sha256"] != sha256_bytes(canonical_json(record["normalized_config"]).encode()):
        raise ValidationError(f"{context}: config_sha256 does not match normalized_config")
    _field_int(record, "seed", context)
    for name in ("simulator_schema_version", "result_schema_version", "python_version", "python_executable"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValidationError(f"{context}: {name} must be a nonempty string")
    if record["simulator_schema_version"] != SIMULATOR_SCHEMA_VERSION or record["result_schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValidationError(f"{context}: incompatible nested schema version")
    manifest = record["workload_manifest"]
    _closed_fields(manifest, MANIFEST_RECORD_FIELDS, f"{context} workload_manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION or manifest["workload_schema_version"] != WORKLOAD_SCHEMA_VERSION:
        raise ValidationError(f"{context}: incompatible workload manifest schema")
    if manifest["workload_class"] not in WORKLOAD_CLASSES:
        raise ValidationError(f"{context}: invalid workload manifest class")
    _field_int(manifest, "request_count", context)
    _field_int(manifest, "seed", context)
    if not isinstance(manifest["generator_config"], dict):
        raise ValidationError(f"{context}: workload manifest generator_config must be an object")
    _require_str(manifest["generator"], f"{context} workload_manifest.generator")
    _nullable_str(manifest["timestamp"], "workload_manifest.timestamp", context)
    if manifest["workload_sha256"] != record["workload_sha256"]:
        raise ValidationError(f"{context}: workload manifest hash disagrees with provenance")
    generated, generated_manifest = generate_records(manifest["generator_config"])
    for name in ("workload_class", "request_count", "seed", "workload_schema_version"):
        if manifest[name] != generated_manifest[name]:
            raise ValidationError(f"{context}: workload manifest {name} contradicts generator_config")
    if len(generated) != manifest["request_count"]:
        raise ValidationError(f"{context}: workload manifest generated request count mismatch")
    if record["seed"] != manifest["seed"] or record["seed"] != record["normalized_config"]["run_seed"]:
        raise ValidationError(f"{context}: run/config/workload seed mismatch")
    runner = record["benchmark_runner"]
    if not isinstance(runner, dict) or set(runner) != {"path", "sha256"} or not all(
            isinstance(runner[name], str) and runner[name] for name in runner):
        raise ValidationError(f"{context}: benchmark_runner must contain path and sha256 strings")
    if re.fullmatch(r"[0-9a-f]{64}", runner["sha256"]) is None:
        raise ValidationError(f"{context}: benchmark_runner.sha256 must be a lowercase SHA-256")
    commands = record["commands"]
    if not isinstance(commands, list) or not commands or any(
            not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command)
            for command in commands):
        raise ValidationError(f"{context}: commands must be a nonempty array of string arrays")
    mode, policy = record["execution_mode"], record["scheduling_policy"]
    if mode not in {"single_active_fcfs", "continuous_batching"}:
        raise ValidationError(f"{context}: invalid execution_mode")
    if mode == "single_active_fcfs" and policy is not None:
        raise ValidationError(f"{context}: FCFS scheduling_policy must be null")
    if mode == "continuous_batching" and policy not in {"DecodeFirst", "FcfsMixed"}:
        raise ValidationError(f"{context}: invalid continuous scheduling_policy")


def _id_array(value: Any, name: str, context: str, known_ids: set[int]) -> list[int]:
    if not isinstance(value, list):
        raise ValidationError(f"{context}: {name} must be an array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValidationError(f"{context}: {name} must contain non-negative integers")
    if len(value) != len(set(value)):
        raise ValidationError(f"{context}: {name} contains duplicate IDs")
    unknown = set(value) - known_ids
    if unknown:
        raise ValidationError(f"{context}: {name} references unknown request ID {min(unknown)}")
    return value


def _conserved(actual: Any, expected: Any, name: str, context: str) -> None:
    if actual != expected:
        raise ValidationError(
            f"{context}: {name} conservation mismatch; expected {expected!r}, got {actual!r}")


def validate_native_records(
        records: list[dict[str, Any]], *, config: dict[str, Any] | None = None,
        workload_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate the closed native stream and all request/iteration/summary conservation."""
    if not isinstance(records, list):
        raise ValidationError("native result must be an array of records")
    if config is not None:
        validate_serving_config(config, ROOT / "native-result-config.json")
    allowed_types = {"request", "iteration", "summary"}
    indexed: list[tuple[int, dict[str, Any]]] = []
    for line, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValidationError(f"result line {line}: record must be an object")
        record_type = record.get("record_type")
        if record_type not in allowed_types:
            raise ValidationError(f"result line {line}: unknown native record_type {record_type!r}")
        if record.get("schema_version") != SIMULATOR_SCHEMA_VERSION:
            raise ValidationError(
                f"{record_type} record at result line {line}: incompatible simulator schema; "
                f"expected {SIMULATOR_SCHEMA_VERSION!r}")
        if record.get("evidence_type") != "SIMULATED":
            raise ValidationError(f"{record_type} record at result line {line}: evidence_type must be SIMULATED")
        indexed.append((line, record))
    summary_items = [(line, record) for line, record in indexed if record["record_type"] == "summary"]
    if len(summary_items) != 1:
        raise ValidationError(f"native result must contain exactly one summary record; got {len(summary_items)}")
    if not indexed or indexed[-1][1]["record_type"] != "summary":
        raise ValidationError("summary record must be terminal")

    request_items = [(line, record) for line, record in indexed if record["record_type"] == "request"]
    iteration_items = [(line, record) for line, record in indexed if record["record_type"] == "iteration"]
    requests = [record for _, record in request_items]
    iterations = [record for _, record in iteration_items]
    summary = summary_items[0][1]
    external_ids: set[str] = set()
    native_ids: set[int] = set()
    valid_states = {"Waiting", "Prefilling", "Decoding", "Finished", "Cancelled"}

    for line, record in request_items:
        context = f"request record at result line {line}"
        _closed_fields(record, REQUEST_RECORD_FIELDS, context)
        request_id = record["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise ValidationError(f"{context}: request_id must be a nonempty string")
        if request_id in external_ids:
            raise ValidationError(f"{context}: duplicate request_id {request_id!r}")
        external_ids.add(request_id)
        internal_id = _field_int(record, "internal_id", context)
        if internal_id in native_ids:
            raise ValidationError(f"{context}: duplicate internal_id {internal_id}")
        native_ids.add(internal_id)
        workload_class = record["workload_class"]
        if not isinstance(workload_class, str) or not workload_class:
            raise ValidationError(f"{context}: workload_class must be a nonempty string")
        if workload_class not in WORKLOAD_CLASSES:
            raise ValidationError(f"{context}: unknown workload_class {workload_class!r}")
        _nullable_str(record["prefix_group"], "prefix_group", context)
        if record["prefix_group"] == "":
            raise ValidationError(f"{context}: prefix_group must be null or nonempty")
        _field_int(record, "deadline_us", context, nullable=True)
        if not isinstance(record["metadata"], dict):
            raise ValidationError(f"{context}: metadata must be an object")
        prompt_tokens = record["prompt_tokens"]
        if prompt_tokens is not None:
            if not isinstance(prompt_tokens, list):
                raise ValidationError(f"{context}: prompt_tokens must be null or an array")
            for index, token in enumerate(prompt_tokens):
                if isinstance(token, bool) or not isinstance(token, int) or not -(2**31) <= token <= 2**31 - 1:
                    raise ValidationError(f"{context}: prompt_tokens[{index}] must be a signed int32")
        for name in ("arrival_time_us", "prompt_tokens_original", "prompt_tokens_matched",
                     "prompt_tokens_scheduled", "generated_tokens", "max_new_tokens"):
            _field_int(record, name, context)
        arrival = record["arrival_time_us"]
        admitted = _field_int(record, "admitted_time_us", context, nullable=True)
        first = _field_int(record, "first_token_time_us", context, nullable=True)
        finish = _field_int(record, "finish_time_us", context, nullable=True)
        generated = record["generated_tokens"]
        original, matched, scheduled = (record[name] for name in
            ("prompt_tokens_original", "prompt_tokens_matched", "prompt_tokens_scheduled"))
        if prompt_tokens is not None and len(prompt_tokens) != original:
            raise ValidationError(f"{context}: prompt_tokens length disagrees with prompt_tokens_original")
        if matched > original or scheduled > original:
            raise ValidationError(f"{context}: matched/scheduled prompt tokens exceed original")
        if admitted is not None and matched + scheduled != original:
            raise ValidationError(
                f"{context}: prompt accounting contradiction; expected original == matched + scheduled")
        if admitted is None and (matched or scheduled or generated):
            raise ValidationError(f"{context}: unadmitted request reports executed work")
        if generated > record["max_new_tokens"]:
            raise ValidationError(f"{context}: generated_tokens exceeds max_new_tokens")
        if admitted is not None and admitted < arrival:
            raise ValidationError(f"{context}: admitted_time_us precedes arrival_time_us")
        if first is not None and (admitted is None or first < admitted):
            raise ValidationError(f"{context}: first_token_time_us precedes admission")
        if finish is not None and (finish < arrival or (admitted is not None and finish < admitted) or
                                   (first is not None and finish < first)):
            raise ValidationError(f"{context}: finish_time_us contradicts lifecycle order")
        state = record["final_state"]
        if state not in valid_states:
            raise ValidationError(f"{context}: invalid final_state {state!r}")
        _nullable_str(record["reason"], "reason", context)
        if state == "Finished" and (admitted is None or finish is None or record["reason"] is not None):
            raise ValidationError(f"{context}: Finished lifecycle/reason fields are inconsistent")
        if state != "Finished" and finish is not None:
            raise ValidationError(f"{context}: unfinished request has finish_time_us")
        if state == "Cancelled" and not record["reason"]:
            raise ValidationError(f"{context}: Cancelled request requires a reason")
        if state != "Cancelled" and record["reason"] is not None:
            raise ValidationError(f"{context}: non-Cancelled request must have null reason")
        if state == "Waiting" and (admitted is not None or first is not None or generated):
            raise ValidationError(f"{context}: Waiting request has admitted/decode state")
        if state == "Prefilling" and (admitted is None or first is not None or generated):
            raise ValidationError(f"{context}: Prefilling lifecycle fields are inconsistent")
        if state == "Decoding" and admitted is None:
            raise ValidationError(f"{context}: Decoding request lacks admission")
        if generated == 0 and first is not None:
            raise ValidationError(f"{context}: zero-output request has first_token_time_us")
        if generated > 0 and first is None:
            raise ValidationError(f"{context}: generated request lacks first_token_time_us")
        times, gaps = record["decode_token_times_us"], record["inter_token_latencies_us"]
        if not isinstance(times, list) or not isinstance(gaps, list):
            raise ValidationError(f"{context}: decode timing fields must be arrays")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in times + gaps):
            raise ValidationError(f"{context}: decode timing values must be non-negative integers")
        _conserved(len(times), generated, "decode timestamp count", context)
        _conserved(len(gaps), max(generated - 1, 0), "inter-token gap count", context)
        if times != sorted(times):
            raise ValidationError(f"{context}: decode timestamps must be non-decreasing")
        if times and times[0] != first:
            raise ValidationError(f"{context}: first decode timestamp disagrees with first_token_time_us")
        if state == "Finished" and times and times[-1] != finish:
            raise ValidationError(f"{context}: final decode timestamp disagrees with finish_time_us")
        _conserved(gaps, [right - left for left, right in zip(times, times[1:])],
                   "inter-token gaps", context)
        metrics = request_metrics(record)
        _exact_number(record["queue_delay_us"], metrics["queue_delay_us"], "queue_delay_us", context)
        _exact_number(record["ttft_us"], metrics["ttft_us"], "ttft_us", context)
        _exact_number(record["end_to_end_latency_us"], metrics["e2e_latency_us"],
                      "end_to_end_latency_us", context)
        _exact_number(record["tpot_us"], metrics["tpot_us"], "tpot_us", context)

    decode_appearances = {request_id: 0 for request_id in native_ids}
    prefill_appearances = {request_id: 0 for request_id in native_ids}
    decode_iteration_times = {request_id: [] for request_id in native_ids}
    prefill_iteration_times = {request_id: [] for request_id in native_ids}
    prefill_iteration_end_times = {request_id: [] for request_id in native_ids}
    previous_iteration = 0
    previous_end = 0
    for line, record in iteration_items:
        context = f"iteration record at result line {line}"
        _closed_fields(record, ITERATION_RECORD_FIELDS, context)
        number = _field_int(record, "iteration_number", context, nullable=False)
        start = _field_int(record, "start_time_us", context)
        end = _field_int(record, "end_time_us", context)
        if number <= previous_iteration:
            raise ValidationError(f"{context}: iteration_number must be unique and strictly increasing")
        if start < previous_end or end < start:
            raise ValidationError(f"{context}: iteration timestamps are not monotonic")
        previous_iteration, previous_end = number, end
        if record["policy"] not in {"DecodeFirst", "FcfsMixed"}:
            raise ValidationError(f"{context}: invalid scheduling policy {record['policy']!r}")
        prefill = _id_array(record["prefill_ids"], "prefill_ids", context, native_ids)
        decode = _id_array(record["decode_ids"], "decode_ids", context, native_ids)
        if set(prefill) & set(decode):
            raise ValidationError(f"{context}: request appears in both prefill_ids and decode_ids")
        deferred = record["deferred"]
        if not isinstance(deferred, list):
            raise ValidationError(f"{context}: deferred must be an array")
        deferred_ids: set[int] = set()
        for index, item in enumerate(deferred):
            if not isinstance(item, dict) or set(item) != {"request_id", "reason"}:
                raise ValidationError(f"{context}: deferred[{index}] must contain request_id and reason")
            deferred_id = item["request_id"]
            if isinstance(deferred_id, bool) or not isinstance(deferred_id, int) or deferred_id not in native_ids:
                raise ValidationError(f"{context}: deferred[{index}] references an unknown request")
            if deferred_id in deferred_ids:
                raise ValidationError(f"{context}: duplicate deferred request ID {deferred_id}")
            if item["reason"] not in {"sequence budget", "token budget", "KV capacity"}:
                raise ValidationError(f"{context}: invalid deferred reason {item['reason']!r}")
            deferred_ids.add(deferred_id)
        if deferred_ids & (set(prefill) | set(decode)):
            raise ValidationError(f"{context}: deferred request also appears in scheduled work")
        for name in ("scheduled_sequences", "scheduled_prefill_tokens", "scheduled_decode_tokens",
                     "total_scheduled_tokens", "kv_allocated", "kv_free", "represented_kv_tokens",
                     "internal_fragmentation_tokens", "cached_blocks", "shared_referenced_blocks",
                     "prefix_hits", "prefix_misses", "matched_tokens"):
            _field_int(record, name, context)
        _conserved(record["scheduled_sequences"], len(prefill) + len(decode),
                   "scheduled_sequences", context)
        _conserved(record["scheduled_decode_tokens"], len(decode),
                   "scheduled_decode_tokens", context)
        _conserved(record["total_scheduled_tokens"],
                   record["scheduled_prefill_tokens"] + record["scheduled_decode_tokens"],
                   "total_scheduled_tokens", context)
        _number_in_unit_interval(record["kv_utilization"], "kv_utilization", context)
        if not isinstance(record["stall"], bool):
            raise ValidationError(f"{context}: stall must be boolean")
        evicted = record["evicted_ids"]
        if not isinstance(evicted, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in evicted):
            raise ValidationError(f"{context}: evicted_ids must contain non-negative integers")
        if len(evicted) != len(set(evicted)):
            raise ValidationError(f"{context}: evicted_ids contains duplicate physical IDs")
        if record["stall"] and (prefill or decode or record["total_scheduled_tokens"]):
            raise ValidationError(f"{context}: stalled iteration contains scheduled work")
        if config is not None:
            total_blocks = config["kv_cache"]["total_blocks"]
            block_size = config["kv_cache"]["block_size_tokens"]
            _conserved(record["kv_allocated"] + record["kv_free"], total_blocks,
                       "configured KV block capacity", context)
            _exact_number(record["kv_utilization"], record["kv_allocated"] / total_blocks,
                          "kv_utilization", context)
            expected_fragmentation = record["kv_allocated"] * block_size - record["represented_kv_tokens"]
            _conserved(record["internal_fragmentation_tokens"], expected_fragmentation,
                       "internal_fragmentation_tokens", context)
        if record["cached_blocks"] > record["kv_allocated"] or record["shared_referenced_blocks"] > record["kv_allocated"]:
            raise ValidationError(f"{context}: cached/shared block gauges exceed allocated blocks")
        for request_id in prefill:
            prefill_appearances[request_id] += 1
            prefill_iteration_times[request_id].append(start)
            prefill_iteration_end_times[request_id].append(end)
        for request_id in decode:
            decode_appearances[request_id] += 1
            decode_iteration_times[request_id].append(end)

    context = f"summary record at result line {summary_items[0][0]}"
    _closed_fields(summary, SUMMARY_RECORD_FIELDS, context)
    if summary["execution_mode"] not in {"single_active_fcfs", "continuous_batching"}:
        raise ValidationError(f"{context}: invalid execution_mode")
    mode = summary["execution_mode"]
    if mode == "single_active_fcfs" and summary["scheduling_policy"] is not None:
        raise ValidationError(f"{context}: FCFS scheduling_policy must be null")
    if mode == "continuous_batching" and summary["scheduling_policy"] not in {"DecodeFirst", "FcfsMixed"}:
        raise ValidationError(f"{context}: invalid continuous scheduling_policy")
    if config is not None:
        _conserved(mode, config["execution_mode"], "execution_mode", context)
        expected_policy = None if mode == "single_active_fcfs" else config["scheduling_policy"]
        _conserved(summary["scheduling_policy"], expected_policy, "scheduling_policy", context)
    if summary["run_status"] not in {"completed", "stalled"}:
        raise ValidationError(f"{context}: invalid run_status {summary['run_status']!r}")
    required_ints = ("submitted", "completed", "cancelled", "rejected", "stalled", "makespan_us",
        "max_batch_size", "deferred_requests", "kv_allocation_failures", "kv_capacity_deferrals",
        "prefix_lookups", "prefix_lookup_hits", "prefix_lookup_misses", "matched_prefix_blocks",
        "cache_eligible_prompt_tokens", "collision_verifications", "matched_prefix_tokens",
        "saved_simulated_prefill_tokens", "eviction_count", "submitted_prompt_tokens",
        "scheduled_prefill_tokens", "generated_tokens", "scheduled_decode_tokens")
    for name in required_ints:
        _field_int(summary, name, context)
    nullable_ints = ("scheduling_iterations", "nonempty_batches", "idle_iterations",
        "total_scheduled_sequences", "max_scheduled_tokens", "current_allocated_blocks",
        "current_free_blocks", "peak_allocated_blocks", "represented_kv_tokens",
        "internal_fragmentation_tokens", "cached_blocks", "shared_referenced_blocks")
    for name in nullable_ints:
        _field_int(summary, name, context, nullable=True)
    average = summary["average_batch_size"]
    if average is not None and (isinstance(average, bool) or
            not isinstance(average, (int, float)) or not math.isfinite(average) or average < 0):
        raise ValidationError(f"{context}: average_batch_size must be null or a finite non-negative number")
    for name in ("kv_current_utilization", "kv_peak_utilization", "prefix_token_hit_rate"):
        _number_in_unit_interval(summary[name], name, context, nullable=True)

    _conserved(summary["submitted"], len(requests), "submitted request count", context)
    state_counts = {state: sum(record["final_state"] == state for record in requests)
                    for state in valid_states}
    _conserved(summary["completed"], state_counts["Finished"], "completed request count", context)
    _conserved(summary["cancelled"], state_counts["Cancelled"], "cancelled request count", context)
    if summary["rejected"] != 0:
        raise ValidationError(f"{context}: rejected count requires explicit rejection records")
    unfinished = len(requests) - summary["completed"] - summary["cancelled"] - summary["rejected"]
    if summary["run_status"] == "completed" and (unfinished or summary["stalled"]):
        raise ValidationError(f"{context}: completed status contradicts unfinished/stalled work")
    if summary["run_status"] == "stalled" and (unfinished == 0 or summary["stalled"] == 0):
        raise ValidationError(f"{context}: stalled status requires unfinished requests and stalled iterations")
    latest_time = max([record["finish_time_us"] for record in requests
                       if record["finish_time_us"] is not None] +
                      [record["end_time_us"] for record in iterations] + [0])
    _conserved(summary["makespan_us"], latest_time, "makespan_us", context)
    request_sums = {
        "submitted_prompt_tokens": sum(record["prompt_tokens_original"] for record in requests),
        "scheduled_prefill_tokens": sum(record["prompt_tokens_scheduled"] for record in requests),
        "generated_tokens": sum(record["generated_tokens"] for record in requests),
        "scheduled_decode_tokens": sum(record["generated_tokens"] for record in requests),
        "matched_prefix_tokens": sum(record["prompt_tokens_matched"] for record in requests),
    }
    for name, expected in request_sums.items():
        _conserved(summary[name], expected, name, context)
    _conserved(summary["prefix_lookups"],
               summary["prefix_lookup_hits"] + summary["prefix_lookup_misses"],
               "prefix_lookups", context)
    expected_hit_rate = (None if summary["cache_eligible_prompt_tokens"] == 0 else
                         summary["matched_prefix_tokens"] / summary["cache_eligible_prompt_tokens"])
    _exact_number(summary["prefix_token_hit_rate"], expected_hit_rate,
                  "prefix_token_hit_rate", context)
    _conserved(summary["saved_simulated_prefill_tokens"], summary["matched_prefix_tokens"],
               "saved_simulated_prefill_tokens", context)

    if mode == "single_active_fcfs":
        if iterations:
            raise ValidationError(f"{context}: single_active_fcfs must not contain iteration records")
        for name in ("scheduling_iterations", "nonempty_batches", "idle_iterations",
                     "total_scheduled_sequences", "max_scheduled_tokens", "current_allocated_blocks",
                     "current_free_blocks", "peak_allocated_blocks", "kv_current_utilization",
                     "kv_peak_utilization", "represented_kv_tokens", "internal_fragmentation_tokens",
                     "cached_blocks", "shared_referenced_blocks"):
            if summary[name] is not None:
                raise ValidationError(f"{context}: FCFS field {name} must be null")
        expected_average = None if not requests else 1
        _exact_number(summary["average_batch_size"], expected_average, "average_batch_size", context)
        _conserved(summary["max_batch_size"], 0 if not requests else 1, "max_batch_size", context)
    else:
        for name in nullable_ints:
            if summary[name] is None:
                raise ValidationError(f"{context}: continuous field {name} must not be null")
        if summary["average_batch_size"] is None and summary["nonempty_batches"]:
            raise ValidationError(f"{context}: nonempty run requires average_batch_size")
        _conserved(summary["scheduling_iterations"], len(iterations), "scheduling_iterations", context)
        iteration_sums = {
            "scheduled_prefill_tokens": sum(item["scheduled_prefill_tokens"] for item in iterations),
            "scheduled_decode_tokens": sum(item["scheduled_decode_tokens"] for item in iterations),
            "total_scheduled_sequences": sum(item["scheduled_sequences"] for item in iterations),
            "nonempty_batches": sum(item["scheduled_sequences"] > 0 for item in iterations),
            "stalled": sum(item["stall"] for item in iterations),
            "deferred_requests": sum(len(item["deferred"]) for item in iterations),
            "kv_capacity_deferrals": sum(
                deferred["reason"] == "KV capacity" for item in iterations for deferred in item["deferred"]),
            "prefix_lookup_hits": sum(item["prefix_hits"] for item in iterations),
            "prefix_lookup_misses": sum(item["prefix_misses"] for item in iterations),
            "matched_prefix_tokens": sum(item["matched_tokens"] for item in iterations),
            "eviction_count": sum(len(item["evicted_ids"]) for item in iterations),
        }
        for name, expected in iteration_sums.items():
            _conserved(summary[name], expected, name, context)
        expected_idle = sum(item["scheduled_sequences"] == 0 and not item["stall"] for item in iterations)
        _conserved(summary["idle_iterations"], expected_idle, "idle_iterations", context)
        _conserved(summary["max_batch_size"], max(
            (item["scheduled_sequences"] for item in iterations), default=0), "max_batch_size", context)
        _conserved(summary["max_scheduled_tokens"], max(
            (item["total_scheduled_tokens"] for item in iterations), default=0),
            "max_scheduled_tokens", context)
        expected_average = (None if summary["nonempty_batches"] == 0 else
                            summary["total_scheduled_sequences"] / summary["nonempty_batches"])
        _exact_number(summary["average_batch_size"], expected_average, "average_batch_size", context)
        observed_peak_blocks = max((item["kv_allocated"] for item in iterations), default=0)
        observed_peak_utilization = max((item["kv_utilization"] for item in iterations), default=0)
        if summary["peak_allocated_blocks"] < observed_peak_blocks:
            raise ValidationError(
                f"{context}: peak_allocated_blocks must cover the iteration peak; "
                f"expected at least {observed_peak_blocks}, got {summary['peak_allocated_blocks']}")
        if summary["kv_peak_utilization"] < observed_peak_utilization:
            raise ValidationError(
                f"{context}: kv_peak_utilization must cover the iteration peak; "
                f"expected at least {observed_peak_utilization}, got {summary['kv_peak_utilization']}")
        if config is not None:
            total_blocks = config["kv_cache"]["total_blocks"]
            if summary["peak_allocated_blocks"] > total_blocks:
                raise ValidationError(f"{context}: peak_allocated_blocks exceeds configured KV capacity")
            _exact_number(summary["kv_peak_utilization"],
                          summary["peak_allocated_blocks"] / total_blocks,
                          "kv_peak_utilization", context)
        last = iterations[-1] if iterations else None
        for summary_name, iteration_name in (("current_allocated_blocks", "kv_allocated"),
                ("current_free_blocks", "kv_free"), ("kv_current_utilization", "kv_utilization"),
                ("represented_kv_tokens", "represented_kv_tokens"),
                ("internal_fragmentation_tokens", "internal_fragmentation_tokens"),
                ("cached_blocks", "cached_blocks"),
                ("shared_referenced_blocks", "shared_referenced_blocks")):
            if last is not None:
                expected = last[iteration_name]
            elif summary_name == "current_free_blocks" and config is not None:
                expected = config["kv_cache"]["total_blocks"]
            else:
                expected = summary[summary_name] if summary_name == "current_free_blocks" else 0
            _conserved(summary[summary_name], expected, summary_name, context)
        if config is not None:
            total_blocks = config["kv_cache"]["total_blocks"]
            block_size = config["kv_cache"]["block_size_tokens"]
            _conserved(summary["current_allocated_blocks"] + summary["current_free_blocks"],
                       total_blocks, "summary configured KV block capacity", context)
            _exact_number(summary["kv_current_utilization"],
                          summary["current_allocated_blocks"] / total_blocks,
                          "kv_current_utilization", context)
            _conserved(summary["internal_fragmentation_tokens"],
                       summary["current_allocated_blocks"] * block_size - summary["represented_kv_tokens"],
                       "summary internal_fragmentation_tokens", context)
        if summary["cached_blocks"] > summary["current_allocated_blocks"] or \
                summary["shared_referenced_blocks"] > summary["current_allocated_blocks"]:
            raise ValidationError(f"{context}: cached/shared block gauges exceed current allocated blocks")
        request_by_native = {record["internal_id"]: record for record in requests}
        for native_id, request in request_by_native.items():
            _conserved(decode_appearances[native_id], request["generated_tokens"],
                       f"request {native_id} decode appearances", context)
            _conserved(decode_iteration_times[native_id], request["decode_token_times_us"],
                       f"request {native_id} decode completion timestamps", context)
            expected_prefill = 1 if request["admitted_time_us"] is not None else 0
            _conserved(prefill_appearances[native_id], expected_prefill,
                       f"request {native_id} prefill appearances", context)
            if expected_prefill:
                _conserved(prefill_iteration_times[native_id][0], request["admitted_time_us"],
                           f"request {native_id} prefill admission timestamp", context)
                if request["final_state"] == "Finished" and request["generated_tokens"] == 0:
                    _conserved(prefill_iteration_end_times[native_id][0], request["finish_time_us"],
                               f"request {native_id} zero-output finish timestamp", context)

    if workload_records is not None:
        if not isinstance(workload_records, list):
            raise ValidationError("workload envelope must be an array")
        workload_by_id = {record["request_id"]: record for record in workload_records}
        if len(workload_by_id) != len(workload_records):
            raise ValidationError("workload envelope contains duplicate external request IDs")
        _conserved(set(workload_by_id), external_ids, "workload/result external ID set", "workload envelope")
        ordered = sorted(workload_records, key=lambda item: (item["arrival_time_us"], item["request_id"]))
        expected_native = {item["request_id"]: index for index, item in enumerate(ordered, 1)}
        for result in requests:
            source = workload_by_id[result["request_id"]]
            context = f"workload envelope for request {result['request_id']!r}"
            expected = {
                "internal_id": expected_native[result["request_id"]],
                "workload_class": source["workload_class"],
                "prefix_group": source.get("prefix_group"),
                "deadline_us": source.get("deadline_us"),
                "metadata": source.get("metadata", {}),
                "prompt_tokens": source.get("prompt_tokens"),
                "arrival_time_us": source["arrival_time_us"],
                "prompt_tokens_original": source["prompt_token_count"],
                "max_new_tokens": source["max_new_tokens"],
            }
            for name, value in expected.items():
                _conserved(result[name], value, name, context)
    return requests, iterations, summary


def analyze_records(records: list[dict[str, Any]], slos: dict[str, int], *,
                    config: dict[str, Any] | None = None,
                    workload_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    requests, iterations, summary = validate_native_records(
        records, config=config, workload_records=workload_records)
    enriched = []
    for record in requests:
        needed = {"request_id", "arrival_time_us", "generated_tokens", "final_state",
                  "prompt_tokens_original", "prompt_tokens_matched", "prompt_tokens_scheduled"}
        missing = needed - set(record)
        if missing:
            raise ValidationError(f"request {record.get('request_id')!r} missing {sorted(missing)}")
        enriched.append(record | request_metrics(record))
    finished = [record for record in enriched if record["final_state"] == "Finished"]
    first_arrival = min((record["arrival_time_us"] for record in enriched), default=0)
    last_arrival = max((record["arrival_time_us"] for record in enriched), default=first_arrival)
    last_finish = max((record["finish_time_us"] for record in finished), default=first_arrival)
    duration = last_finish - first_arrival
    good = 0
    violations = {"ttft": 0, "tpot": 0, "e2e": 0}
    for record in finished:
        passed = True
        for name, metric_name in (("ttft", "ttft_us"), ("tpot", "tpot_us"), ("e2e", "e2e_latency_us")):
            threshold = slos.get(f"{name}_slo_us")
            if threshold is None:
                continue
            value = record[metric_name]
            # An inapplicable metric (zero/one output token TPOT, zero-output TTFT)
            # does not fail an otherwise applicable configured SLO.
            if value is not None and value > threshold:
                violations[name] += 1
                passed = False
        good += int(passed)
    def percentiles(field: str) -> dict[str, int | float | None]:
        values = [record[field] for record in finished if record[field] is not None]
        return {f"p{int(p * 100)}": nearest_rank(values, p) for p in (0.5, 0.9, 0.95, 0.99)}
    output_tokens = sum(record["generated_tokens"] for record in finished)
    total_tokens = output_tokens + sum(record["prompt_tokens_scheduled"] for record in finished)
    queue_window_end = summary.get("makespan_us", last_finish)
    queue_depth_events: list[tuple[int, int]] = []
    for record in enriched:
        if record["arrival_time_us"] <= queue_window_end:
            queue_depth_events.append((record["arrival_time_us"], 1))
        if record.get("admitted_time_us") is not None and record["admitted_time_us"] <= queue_window_end:
            queue_depth_events.append((record["admitted_time_us"], -1))
    depth = maximum_depth = 0
    weighted_depth = 0
    previous = first_arrival
    for timestamp, delta in sorted(queue_depth_events, key=lambda item: (item[0], -item[1])):
        weighted_depth += depth * max(0, timestamp - previous)
        previous = timestamp
        depth += delta
        maximum_depth = max(maximum_depth, depth)
    weighted_depth += depth * max(0, queue_window_end - previous)
    queue_duration = queue_window_end - first_arrival
    average_depth = None if queue_duration <= 0 else weighted_depth / queue_duration
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evidence_type": "SIMULATED",
        "request_count": len(enriched),
        "completed": len(finished),
        "run_status": summary["run_status"],
        "submitted": summary["submitted"],
        "unfinished": len(enriched) - len(finished),
        "completion_ratio": None if not enriched else len(finished) / len(enriched),
        "arrival_rate_per_s": safe_rate(max(0, len(enriched) - 1), last_arrival - first_arrival),
        "duration_us": duration,
        "request_throughput_per_s": safe_rate(len(finished), duration),
        "output_token_throughput_per_s": safe_rate(output_tokens, duration),
        "total_token_throughput_per_s": safe_rate(total_tokens, duration),
        "good_requests": good,
        "good_requests_per_s": safe_rate(good, duration),
        "goodput_ratio": None if not finished else good / len(finished),
        "slo_violations": violations,
        "ttft_us": percentiles("ttft_us"),
        "e2e_latency_us": percentiles("e2e_latency_us"),
        "tpot_us": percentiles("tpot_us"),
        "queue_delay_us": percentiles("queue_delay_us"),
        "average_queue_depth": average_depth,
        "maximum_queue_depth": maximum_depth,
        "average_batch_size": summary.get("average_batch_size"),
        "maximum_batch_size": summary.get("max_batch_size"),
        "kv_peak_utilization": summary.get("kv_peak_utilization"),
        "prefix_lookup_hit_rate": None if summary.get("prefix_lookup_hits", 0) + summary.get("prefix_lookup_misses", 0) == 0 else
            summary["prefix_lookup_hits"] / (summary["prefix_lookup_hits"] + summary["prefix_lookup_misses"]),
        "prefix_token_hit_rate": summary.get("prefix_token_hit_rate"),
        "kv_capacity_deferrals": summary.get("kv_capacity_deferrals"),
        "stalled_iterations": summary.get("stalled"),
        "eviction_count": summary.get("eviction_count"),
        "saved_simulated_prefill_tokens": summary.get("saved_simulated_prefill_tokens"),
        "scheduled_prefill_tokens": summary.get("scheduled_prefill_tokens"),
        "matched_prefix_tokens": summary.get("matched_prefix_tokens"),
        "internal_fragmentation_tokens": summary.get("internal_fragmentation_tokens"),
        "peak_allocated_blocks": summary.get("peak_allocated_blocks"),
        "iteration_count": len(iterations),
        "itl_available": bool(requests) and all("decode_token_times_us" in record for record in requests),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"result line {line_number}: malformed JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValidationError(f"result line {line_number}: record must be an object")
        records.append(value)
    return records


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


def provenance(config: dict[str, Any], workload_path: Path, manifest: dict[str, Any],
               commands: list[list[str]], *, runner: Path) -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    build_type = None
    cache = ROOT / "build/debug/CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:STRING="):
                build_type = line.partition("=")[2]
                break
    return {
        "evidence_type": "simulated",
        "label": "SIMULATED",
        "git_revision": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "dirty_worktree": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "llama_cpp_submodule_revision": git_value("-C", "third_party/llama.cpp", "rev-parse", "HEAD"),
        "normalized_config": config,
        "config_sha256": sha256_bytes(canonical_json(config).encode()),
        "workload_manifest": manifest,
        "workload_sha256": sha256_file(workload_path),
        "seed": config["run_seed"],
        "simulator_schema_version": SIMULATOR_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "python_version": sys.version,
        "python_executable": portable_path(sys.executable),
        "cmake_build_type": build_type,
        "benchmark_runner": {
            "path": portable_path(runner),
            "sha256": sha256_file(runner),
        },
        "commands": [normalize_command(command) for command in commands],
        "timestamp": None,
    }
