#!/usr/bin/env python3
"""Run a reproducible llama-bench parameter matrix and normalize its output."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
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
    repetitions: int
    warmup_runs: int
    timeout_seconds: float
    output_directory: Path
    mmap: bool = True
    cpu_only: bool = True
    smoke_fixture: Path | None = None


@dataclass(frozen=True)
class Case:
    threads: int
    prompt_tokens: int
    generated_tokens: int
    batch_size: int
    context_size: int


CSV_FIELDS = (
    "timestamp_utc", "model_name", "model_size", "backend", "thread_count",
    "prompt_tokens", "generated_tokens", "prompt_tokens_per_second",
    "generation_tokens_per_second", "test_identifier", "batch_size",
    "context_size", "repetition", "elapsed_seconds",
)


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
    optional = {"smoke_fixture", "mmap", "cpu_only"}
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
    return BenchmarkConfig(
        executable=path_value("executable"), model=path_value("model"),
        threads=_positive_list(raw, "threads"),
        prompt_tokens=_positive_list(raw, "prompt_tokens"),
        generated_tokens=_positive_list(raw, "generated_tokens"),
        batch_sizes=_positive_list(raw, "batch_sizes"),
        context_sizes=_positive_list(raw, "context_sizes"),
        repetitions=raw["repetitions"], warmup_runs=raw["warmup_runs"],
        timeout_seconds=float(timeout), output_directory=path_value("output_directory"),
        mmap=raw.get("mmap", True), cpu_only=raw.get("cpu_only", True),
        smoke_fixture=((path.parent / smoke).resolve() if smoke and not Path(smoke).is_absolute() else Path(smoke) if smoke else None),
    )


def cases(config: BenchmarkConfig) -> Iterable[Case]:
    result = [Case(*values) for values in itertools.product(
        config.threads, config.prompt_tokens, config.generated_tokens,
        config.batch_sizes, config.context_sizes,
    )]
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
        "-b", str(case.batch_size), "-d", str(depth), "-r", "1",
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
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.returncode,
                "elapsed_seconds": time.perf_counter() - started, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"stdout": exc.stdout or "", "stderr": exc.stderr or "", "return_code": None,
                "elapsed_seconds": time.perf_counter() - started, "timed_out": True}
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


def run(config: BenchmarkConfig, project_root: Path, smoke: bool = False) -> tuple[Path, Path, int]:
    metadata = environment_metadata(project_root)
    records: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    fixture = config.smoke_fixture
    if smoke and (fixture is None or not fixture.is_file()):
        raise ConfigError("smoke mode requires an existing smoke_fixture")
    for case in cases(config):
        for iteration in range(config.warmup_runs + config.repetitions):
            is_warmup = iteration < config.warmup_runs
            repetition = iteration - config.warmup_runs + 1
            command = build_command(config, case)
            result = ({"stdout": fixture.read_text(encoding="utf-8"), "stderr": "", "return_code": 0,
                       "elapsed_seconds": 0.0, "timed_out": False} if smoke else execute(command, config.timeout_seconds))
            record = {"environment": metadata, "command": command, "case": asdict(case),
                      "warmup": is_warmup, "repetition": None if is_warmup else repetition, **result}
            records.append(record)
            if is_warmup or result["return_code"] != 0:
                continue
            try:
                parsed = parse_llama_bench(str(result["stdout"]))
            except ValueError as exc:
                record["parse_error"] = str(exc)
                continue
            for row in parsed:
                normalized.append({**row, "timestamp_utc": metadata["timestamp_utc"],
                                   "batch_size": case.batch_size, "context_size": case.context_size,
                                   "repetition": repetition, "elapsed_seconds": result["elapsed_seconds"]})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = config.output_directory / f"llama-bench-{stamp}.jsonl"
    csv_path = config.output_directory / f"llama-bench-{stamp}.csv"
    write_results(raw_path, csv_path, records, normalized)
    return raw_path, csv_path, len(normalized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML benchmark configuration")
    parser.add_argument("--smoke", action="store_true", help="parse the configured fixture without llama-bench or a model")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        raw, normalized, count = run(config, Path(__file__).resolve().parents[1], args.smoke)
    except ConfigError as exc:
        parser.error(str(exc))
    print(f"raw results: {raw}")
    print(f"normalized results: {normalized} ({count} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
