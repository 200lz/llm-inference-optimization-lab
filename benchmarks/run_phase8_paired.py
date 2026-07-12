#!/usr/bin/env python3
"""Run the focused Phase 8 paired llama-bench confirmation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL line {line_number}") from exc
        if record.get("return_code") == 0 and not record.get("parse_error"):
            completed.add(str(record["invocation_id"]))
    return completed


def plan(seed: int, pairs: int, warmups: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    execution_index = 0
    for warmup in range(1, warmups + 1):
        for variant in ("baseline", "optimized") if warmup % 2 else ("optimized", "baseline"):
            execution_index += 1
            result.append({"invocation_id": f"warmup-{warmup}-{variant}", "warmup": True,
                           "pair_id": None, "order_within_pair": None,
                           "execution_index": execution_index, "variant": variant})
    rng = random.Random(seed)
    for pair_id in range(1, pairs + 1):
        order = ["baseline", "optimized"]
        rng.shuffle(order)
        for position, variant in enumerate(order, 1):
            execution_index += 1
            result.append({"invocation_id": f"pair-{pair_id}-{variant}", "warmup": False,
                           "pair_id": pair_id, "order_within_pair": position,
                           "execution_index": execution_index, "variant": variant})
    return result


def parse_bench(stdout: str) -> tuple[float, float]:
    rows = json.loads(stdout)
    prompt = [row for row in rows if row.get("n_prompt") == 1024 and row.get("n_gen") == 0]
    generation = [row for row in rows if row.get("n_prompt") == 0 and row.get("n_gen") == 64]
    if len(prompt) != 1 or len(generation) != 1:
        raise ValueError("expected one prompt and one generation result")
    return float(prompt[0]["avg_ts"]), float(generation[0]["avg_ts"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--affinity", default="0,2,4,6")
    parser.add_argument("--seed", type=int, default=8081)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.pairs < 20 or args.warmups < 3:
        parser.error("at least 20 pairs and 3 warm-ups are required")
    binaries = {"baseline": args.baseline.resolve(), "optimized": args.optimized.resolve()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    records_path = args.output_dir / "runs.jsonl"
    manifest = {
        "schema_version": 1, "seed": args.seed, "pairs": args.pairs, "warmups_per_variant": args.warmups,
        "affinity": args.affinity, "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
        "binaries": {name: {"path": str(path), "sha256": sha256(path)} for name, path in binaries.items()},
        "workload": {"prompt_tokens": 1024, "generated_tokens": 64, "threads": 4,
                     "batch": 512, "ubatch": 512, "cpu_only": True, "ngl": 0,
                     "device": "none", "mmap": True},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit("existing manifest does not match requested experiment")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    completed = load_completed(records_path)
    for item in plan(args.seed, args.pairs, args.warmups):
        if item["invocation_id"] in completed:
            continue
        binary = binaries[item["variant"]]
        command = ["taskset", "-c", args.affinity, "/usr/bin/time", "-f", "%e %M", "-o",
                   str(args.output_dir / ".time.tmp"), str(binary), "-m", str(args.model.resolve()),
                   "-p", "1024", "-n", "64", "-t", "4", "-b", "512", "-ub", "512",
                   "-dev", "none", "-ngl", "0", "-mmp", "1", "-r", "1", "-o", "json"]
        started = utc_now(); monotonic = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout, check=False)
        elapsed = time.monotonic() - monotonic; finished = utc_now()
        record = dict(item, started_utc=started, finished_utc=finished, exact_command=command,
                      binary_sha256=manifest["binaries"][item["variant"]]["sha256"],
                      return_code=result.returncode, wall_duration_seconds=elapsed,
                      stdout=result.stdout, stderr=result.stderr)
        try:
            timing = (args.output_dir / ".time.tmp").read_text(encoding="utf-8").split()
            record["time_duration_seconds"] = float(timing[0])
            record["peak_rss_kib"] = int(timing[1])
            record["prompt_tokens_per_second"], record["generation_tokens_per_second"] = parse_bench(result.stdout)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            record["parse_error"] = str(exc)
        append_jsonl(records_path, record)
        print(f"completed {item['invocation_id']} at execution {item['execution_index']}", flush=True)
        if result.returncode or record.get("parse_error"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
