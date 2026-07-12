#!/usr/bin/env python3
"""Run a reproducible llama-bench parameter matrix and normalize its output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


@dataclass(frozen=True)
class BenchmarkConfig:
    executable: Path
    model: Path
    threads: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    context_sizes: tuple[int, ...]
    ubatch_sizes: tuple[int, ...]
    repetitions: int
    warmup_runs: int
    timeout_seconds: float
    output_directory: Path
    mmap: bool = True
    cpu_only: bool = True
    smoke_fixture: Path | None = None
    explicit_cases: tuple[Case, ...] | None = None


@dataclass(frozen=True)
class Case:
    threads: int
    prompt_tokens: int
    generated_tokens: int
    batch_size: int
    context_size: int
    ubatch_size: int = 512


CSV_FIELDS = (
    "timestamp_utc", "model_name", "model_size", "backend", "thread_count",
    "prompt_tokens", "generated_tokens", "prompt_tokens_per_second",
    "generation_tokens_per_second", "test_identifier", "batch_size",
    "context_size", "repetition", "elapsed_seconds",
)


@dataclass(frozen=True)
class RunOptions:
    resume: bool = False
    retry_failures: bool = False
    force_resume: bool = False
    allow_partial_analysis: bool = False
    write_csv: bool = False
    max_total_seconds: float | None = None
    fsync: bool = True


@dataclass(frozen=True)
class Invocation:
    number: int
    total: int
    case: Case
    case_id: str
    invocation_id: str
    warmup: bool
    repetition: int


def _positive_list(data: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value or any(type(item) is not int or item <= 0 for item in value):
        raise ConfigError(f"{key} must be a non-empty list of positive integers")
    return tuple(value)


def load_config(path: Path) -> BenchmarkConfig:
    """Load and strictly validate a YAML benchmark configuration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    required = {
        "executable", "model", "threads", "prompt_tokens", "generated_tokens",
        "batch_sizes", "context_sizes", "repetitions", "warmup_runs",
        "timeout_seconds", "output_directory",
    }
    optional = {"smoke_fixture", "mmap", "cpu_only", "explicit_cases", "ubatch_sizes"}
    missing = sorted(required - raw.keys())
    unknown = sorted(raw.keys() - required - optional)
    if missing:
        raise ConfigError(f"missing required keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"unknown keys: {', '.join(unknown)}")

    def path_value(key: str) -> Path:
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty path string")
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()

    for key, minimum in (("repetitions", 1), ("warmup_runs", 0)):
        value = raw[key]
        if type(value) is not int or value < minimum:
            raise ConfigError(f"{key} must be an integer >= {minimum}")
    timeout = raw["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigError("timeout_seconds must be a positive number")
    smoke = raw.get("smoke_fixture")
    if smoke is not None and (not isinstance(smoke, str) or not smoke.strip()):
        raise ConfigError("smoke_fixture must be a non-empty path string")
    for key in ("mmap", "cpu_only"):
        if key in raw and type(raw[key]) is not bool:
            raise ConfigError(f"{key} must be a boolean")
    ubatches = _positive_list(raw, "ubatch_sizes") if "ubatch_sizes" in raw else (512,)
    explicit: tuple[Case, ...] | None = None
    if "explicit_cases" in raw:
        entries = raw["explicit_cases"]
        if not isinstance(entries, list) or not entries:
            raise ConfigError("explicit_cases must be a non-empty list")
        allowed = {"threads", "prompt_tokens", "generated_tokens", "batch_size", "context_size", "ubatch_size"}
        built = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != allowed:
                raise ConfigError(f"explicit_cases[{index}] must contain exactly {', '.join(sorted(allowed))}")
            if any(type(entry[key]) is not int or entry[key] <= 0 for key in allowed):
                raise ConfigError(f"explicit_cases[{index}] values must be positive integers")
            built.append(Case(**entry))
        explicit = tuple(built)
    return BenchmarkConfig(
        executable=path_value("executable"), model=path_value("model"),
        threads=_positive_list(raw, "threads"),
        prompt_tokens=_positive_list(raw, "prompt_tokens"),
        generated_tokens=_positive_list(raw, "generated_tokens"),
        batch_sizes=_positive_list(raw, "batch_sizes"),
        context_sizes=_positive_list(raw, "context_sizes"),
        ubatch_sizes=ubatches,
        repetitions=raw["repetitions"], warmup_runs=raw["warmup_runs"],
        timeout_seconds=float(timeout), output_directory=path_value("output_directory"),
        mmap=raw.get("mmap", True), cpu_only=raw.get("cpu_only", True),
        smoke_fixture=((path.parent / smoke).resolve() if smoke and not Path(smoke).is_absolute() else Path(smoke) if smoke else None),
        explicit_cases=explicit,
    )


def cases(config: BenchmarkConfig) -> Iterable[Case]:
    result = list(config.explicit_cases) if config.explicit_cases is not None else [Case(*values) for values in itertools.product(
        config.threads, config.prompt_tokens, config.generated_tokens,
        config.batch_sizes, config.context_sizes, config.ubatch_sizes,
    )]
    result = list(dict.fromkeys(result))
    invalid = [case for case in result if case.context_size < case.prompt_tokens + case.generated_tokens]
    if invalid:
        case = invalid[0]
        raise ConfigError(
            f"context size {case.context_size} is smaller than prompt + generated tokens "
            f"({case.prompt_tokens + case.generated_tokens})"
        )
    return iter(result)


def build_command(config: BenchmarkConfig, case: Case) -> list[str]:
    """Construct an argv list; no value is interpreted by a shell."""
    depth = case.context_size - case.prompt_tokens - case.generated_tokens
    command = [
        str(config.executable), "-m", str(config.model), "-t", str(case.threads),
        "-p", str(case.prompt_tokens), "-n", str(case.generated_tokens),
        "-b", str(case.batch_size), "-ub", str(case.ubatch_size), "-d", str(depth), "-r", "1",
        "-o", "md", "-mmp", "1" if config.mmap else "0",
    ]
    if config.cpu_only:
        command.extend(["-ngl", "0", "-dev", "none"])
    return command


def _version(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else None


def _git_commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True,
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata(project_root: Path) -> dict[str, Any]:
    cpu_model = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operating_system": platform.system(), "kernel": platform.release(),
        "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "cmake_version": _version(["cmake", "--version"]),
        "compiler_version": _version([os.environ.get("CXX", "/usr/bin/g++"), "--version"]),
        "llama_cpp_git_commit": _git_commit(project_root / "third_party" / "llama.cpp"),
        "project_git_commit": _git_commit(project_root),
    }


def execute(command: Sequence[str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "return_code": process.returncode,
                "elapsed_seconds": time.perf_counter() - started, "timed_out": False}
    except subprocess.TimeoutExpired:
        assert process is not None
        process.kill()
        stdout, stderr = process.communicate()
        return {"stdout": stdout or "", "stderr": stderr or "", "return_code": None,
                "elapsed_seconds": time.perf_counter() - started, "timed_out": True}
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    except OSError as exc:
        return {"stdout": "", "stderr": str(exc), "return_code": None,
                "elapsed_seconds": time.perf_counter() - started, "timed_out": False}


def parse_llama_bench(stdout: str) -> list[dict[str, Any]]:
    """Parse llama-bench's stable Markdown table output and combine pp/tg rows."""
    table: list[dict[str, str]] = []
    headers: list[str] | None = None
    for line in stdout.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if headers is None and "model" in cells and "test" in cells:
            headers = cells
            continue
        if headers and len(cells) == len(headers) and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            table.append(dict(zip(headers, cells)))
    if not table:
        raise ValueError("no llama-bench result table found")
    combined: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in table:
        test = row.get("test", "")
        match = re.fullmatch(r"(pp|tg)(\d+)(?: @ d\d+)?", test)
        if not match:
            continue
        key = (row.get("model", ""), row.get("backend", ""), row.get("threads", ""), row.get("size", ""))
        item = combined.setdefault(key, {
            "model_name": key[0], "model_size": key[3] or None, "backend": key[1],
            "thread_count": int(key[2]), "prompt_tokens": None, "generated_tokens": None,
            "prompt_tokens_per_second": None, "generation_tokens_per_second": None,
            "test_identifier": "",
        })
        rate_text = row.get("t/s", "").split("±", 1)[0].strip()
        try:
            rate = float(rate_text)
        except ValueError as exc:
            raise ValueError(f"invalid throughput in test {test}: {rate_text!r}") from exc
        count = int(match.group(2))
        if match.group(1) == "pp":
            item["prompt_tokens"], item["prompt_tokens_per_second"] = count, rate
        else:
            item["generated_tokens"], item["generation_tokens_per_second"] = count, rate
        item["test_identifier"] = "+".join(filter(None, [item["test_identifier"], test]))
    rows = list(combined.values())
    if not rows:
        raise ValueError("result table contains no ppN or tgN tests")
    return rows


def write_results(raw_path: Path, csv_path: Path, records: Sequence[Mapping[str, Any]], normalized: Sequence[Mapping[str, Any]]) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


def case_id(case: Case) -> str:
    return f"p{case.prompt_tokens}-n{case.generated_tokens}-t{case.threads}-b{case.batch_size}-ub{case.ubatch_size}-c{case.context_size}"


def invocations(config: BenchmarkConfig) -> list[Invocation]:
    specs: list[tuple[Case, bool, int]] = []
    for case in cases(config):
        specs.extend((case, True, index) for index in range(1, config.warmup_runs + 1))
        specs.extend((case, False, index) for index in range(1, config.repetitions + 1))
    total = len(specs)
    return [Invocation(number, total, case, case_id(case),
                       f"{case_id(case)}:{'warmup' if warmup else 'measured'}:{repetition}",
                       warmup, repetition)
            for number, (case, warmup, repetition) in enumerate(specs, 1)]


def config_fingerprint(config: BenchmarkConfig) -> str:
    payload = {
        "executable": str(config.executable), "model": str(config.model),
        "cases": [asdict(case) for case in cases(config)], "repetitions": config.repetitions,
        "warmup_runs": config.warmup_runs, "timeout_seconds": config.timeout_seconds,
        "mmap": config.mmap, "cpu_only": config.cpu_only,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            suffix = " (possibly truncated final line)" if index == len(lines) else ""
            raise ConfigError(f"invalid JSONL line {index}{suffix}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("invocation_id"), str):
            raise ConfigError(f"invalid JSONL record on line {index}")
        records.append(value)
    return records


def append_record(stream: Any, record: Mapping[str, Any], fsync: bool = True) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    if fsync:
        os.fsync(stream.fileno())


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _latest(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(record["invocation_id"]): record for record in records}


def _successful(record: Mapping[str, Any]) -> bool:
    return record.get("status") == "success"


def normalized_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in _latest(records).values():
        if record.get("warmup") or not _successful(record):
            continue
        case = record["case"]
        for row in record.get("parsed_results", []):
            normalized.append({**row, "timestamp_utc": record["started_at"],
                               "batch_size": case["batch_size"], "context_size": case["context_size"],
                               "repetition": record["repetition"], "elapsed_seconds": record["elapsed_seconds"]})
    return normalized


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], incomplete: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        if incomplete:
            stream.write("# incomplete: true\n")
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(config: BenchmarkConfig, project_root: Path, smoke: bool = False,
        options: RunOptions | None = None) -> tuple[Path, Path, int]:
    options = options or RunOptions()
    fixture = config.smoke_fixture
    if smoke and (fixture is None or not fixture.is_file()):
        raise ConfigError("smoke mode requires an existing smoke_fixture")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = config.output_directory / "run.jsonl"
    csv_path = config.output_directory / "normalized.csv"
    summary_path = config.output_directory / "run-summary.json"
    fingerprint = config_fingerprint(config)
    plan = invocations(config)
    existing = read_jsonl(raw_path) if options.resume else []
    if options.resume and not summary_path.exists():
        raise ConfigError("resume requires run-summary.json")
    old_summary = json.loads(summary_path.read_text(encoding="utf-8")) if options.resume else {}
    if options.resume and old_summary.get("config_fingerprint") != fingerprint and not options.force_resume:
        raise ConfigError("configuration is incompatible with existing run; use --force-resume to override")
    if not options.resume and (raw_path.exists() or summary_path.exists()):
        raise ConfigError(f"output directory already contains a run; use --resume or choose another directory")
    run_id = str(old_summary.get("run_id") or uuid.uuid4())
    metadata = environment_metadata(project_root)
    latest = _latest(existing)
    start = time.monotonic()
    successful = sum(_successful(record) for record in latest.values())
    failed = len(latest) - successful
    timed_out = sum(record.get("status") == "timed_out" for record in latest.values())
    summary: dict[str, Any] = {
        "run_id": run_id, "config_fingerprint": fingerprint, "planned_invocations": len(plan),
        "completed_invocations": len(latest), "successful_invocations": successful,
        "failed_invocations": failed, "timed_out_invocations": timed_out,
        "last_completed_case": old_summary.get("last_completed_case"),
        "elapsed_seconds": float(old_summary.get("elapsed_seconds", 0)), "run_status": "running",
        "max_total_seconds": options.max_total_seconds,
    }
    _write_summary(summary_path, summary)
    total_limit = "unlimited" if options.max_total_seconds is None else f"{options.max_total_seconds:g}s"
    print(f"run_id={run_id} planned={len(plan)} whole_matrix_timeout={total_limit} fsync={options.fsync}", flush=True)
    interrupted = False
    try:
        with raw_path.open("a", encoding="utf-8") as stream:
            for invocation in plan:
                previous = latest.get(invocation.invocation_id)
                if previous and (not options.retry_failures or _successful(previous)):
                    continue
                elapsed = float(old_summary.get("elapsed_seconds", 0)) + time.monotonic() - start
                if options.max_total_seconds is not None and elapsed >= options.max_total_seconds:
                    summary["run_status"] = "failed"
                    break
                phase = "warmup" if invocation.warmup else "measured"
                repetition_text = f"{invocation.repetition}/{config.warmup_runs if invocation.warmup else config.repetitions}"
                print(f"[{invocation.number}/{invocation.total}] case={invocation.case_id} {phase}={repetition_text}", flush=True)
                print(f"prompt={invocation.case.prompt_tokens} gen={invocation.case.generated_tokens} "
                      f"threads={invocation.case.threads} timeout={config.timeout_seconds:g}s "
                      f"elapsed={elapsed:.1f}s", flush=True)
                started_at = datetime.now(timezone.utc).isoformat()
                command = build_command(config, invocation.case)
                try:
                    result = ({"stdout": fixture.read_text(encoding="utf-8"), "stderr": "", "return_code": 0,
                               "elapsed_seconds": 0.0, "timed_out": False} if smoke else execute(command, config.timeout_seconds))
                except KeyboardInterrupt:
                    interrupted = True
                    raise
                parsed: list[dict[str, Any]] = []
                parse_error = None
                status = "timed_out" if result["timed_out"] else "failed" if result["return_code"] != 0 else "success"
                if status == "success":
                    try:
                        parsed = parse_llama_bench(str(result["stdout"]))
                    except ValueError as exc:
                        parse_error, status = str(exc), "parse_error"
                record = {"run_id": run_id, "case_id": invocation.case_id,
                          "invocation_id": invocation.invocation_id, "phase": phase,
                          "repetition": invocation.repetition, "warmup": invocation.warmup,
                          "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                          "status": status, "environment": metadata, "command": command,
                          "case": asdict(invocation.case), **result, "parsed_results": parsed}
                if parse_error:
                    record["parse_error"] = parse_error
                append_record(stream, record, options.fsync)
                existing.append(record); latest[invocation.invocation_id] = record
                successful = sum(_successful(item) for item in latest.values())
                failed = len(latest) - successful
                timed_out = sum(item.get("status") == "timed_out" for item in latest.values())
                summary.update({"completed_invocations": len(latest), "successful_invocations": successful,
                                "failed_invocations": failed, "timed_out_invocations": timed_out,
                                "last_completed_case": invocation.case_id,
                                "elapsed_seconds": float(old_summary.get("elapsed_seconds", 0)) + time.monotonic() - start})
                _write_summary(summary_path, summary)
                throughput = ""
                if parsed:
                    row = parsed[0]
                    throughput = f" pp={row.get('prompt_tokens_per_second')} tg={row.get('generation_tokens_per_second')} tokens/s"
                print(f"return_code={result['return_code']} duration={result['elapsed_seconds']:.3f}s "
                      f"parse={status}{throughput} successful={successful} failed={failed}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        summary["elapsed_seconds"] = float(old_summary.get("elapsed_seconds", 0)) + time.monotonic() - start
        measured_ids = {item.invocation_id for item in plan if not item.warmup}
        complete = measured_ids.issubset(latest)
        summary["run_status"] = "interrupted" if interrupted else "completed" if complete else summary.get("run_status", "failed")
        _write_summary(summary_path, summary)
    rows = normalized_records(existing)
    if complete or options.write_csv or options.allow_partial_analysis:
        _write_csv(csv_path, rows, incomplete=not complete)
    elif csv_path.exists():
        csv_path.unlink()
    if interrupted:
        print(f"run interrupted; preserved {len(latest)} completed invocation(s) in {raw_path}", file=sys.stderr, flush=True)
        raise KeyboardInterrupt
    return raw_path, csv_path, len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML benchmark configuration")
    parser.add_argument("--smoke", action="store_true", help="parse the configured fixture without llama-bench or a model")
    parser.add_argument("--resume", action="store_true", help="resume the durable run in the output directory")
    parser.add_argument("--retry-failures", action="store_true", help="with --resume, rerun failed, timed-out, and parse-error records")
    parser.add_argument("--force-resume", action="store_true", help="permit resume despite a configuration fingerprint mismatch")
    parser.add_argument("--write-csv", action="store_true", help="explicitly write CSV even when the measured matrix is incomplete")
    parser.add_argument("--allow-partial-analysis", action="store_true", help="write an incomplete-labeled partial CSV")
    parser.add_argument("--max-total-seconds", type=float, help="whole-matrix elapsed-time limit; default is unlimited")
    parser.add_argument("--no-fsync", action="store_true", help="flush without fsync (not recommended for long runs)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        if args.retry_failures and not args.resume:
            raise ConfigError("--retry-failures requires --resume")
        if args.max_total_seconds is not None and args.max_total_seconds <= 0:
            raise ConfigError("--max-total-seconds must be positive")
        options = RunOptions(args.resume, args.retry_failures, args.force_resume,
                             args.allow_partial_analysis, args.write_csv,
                             args.max_total_seconds, not args.no_fsync)
        raw, normalized, count = run(config, Path(__file__).resolve().parents[1], args.smoke, options)
    except ConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        return 130
    print(f"raw results: {raw}")
    if normalized.exists():
        print(f"normalized results: {normalized} ({count} rows)")
    else:
        print("normalized results: not generated (run incomplete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
