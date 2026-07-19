#!/usr/bin/env python3
"""Run a bounded serving experiment matrix, failing fast on invalid results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_serving_simulation import run
from benchmarks.serving_common import (MATRIX_SCHEMA_VERSION, ROOT, ValidationError, load_json,
    portable_path, validate_output_destination)


def validate_matrix(matrix: dict, path: Path) -> list[str]:
    allowed = {"schema_version", "evidence_type", "max_default_runs", "command_manifest",
               "comparisons", "configs"}
    if set(matrix) - allowed:
        raise ValidationError(f"unknown serving matrix fields: {sorted(set(matrix) - allowed)}")
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION or matrix.get("evidence_type") != "SIMULATED":
        raise ValidationError("invalid serving matrix schema/evidence type")
    configs = matrix.get("configs")
    if not isinstance(configs, list) or any(not isinstance(value, str) for value in configs):
        raise ValidationError("matrix configs must be an array of paths")
    names = [Path(value).stem for value in configs]
    if len(names) != len(set(names)):
        raise ValidationError("matrix config names must be unique")
    for value in configs:
        candidate = (ROOT / value).resolve()
        if ROOT not in candidate.parents or not candidate.is_file():
            raise ValidationError(f"matrix config does not exist inside repository: {value}")
    comparisons = matrix.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValidationError("matrix comparisons must be an object")
    for group, definition in comparisons.items():
        if not isinstance(definition, dict) or set(definition) != {"configs", "metrics"}:
            raise ValidationError(f"matrix comparison {group!r} is invalid")
        members = definition["configs"]
        if not isinstance(members, list) or any(member not in names for member in members):
            raise ValidationError(f"matrix comparison {group!r} references a config not in this matrix")
    expected_count = matrix.get("max_default_runs")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count != len(configs):
        raise ValidationError("matrix max_default_runs must equal its actual config count")
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/debug/serving-benchmark-runner")
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--update-reference", action="store_true")
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        matrix = load_json(args.matrix)
        configs = validate_matrix(matrix, args.matrix)
        selected = [value for value in configs if not args.select or Path(value).stem in args.select]
        if len(selected) > args.max_runs:
            raise ValidationError(f"matrix has {len(selected)} runs; raise --max-runs explicitly")
        if len(args.select) != len(set(args.select)) or (args.select and len(selected) != len(args.select)):
            raise ValidationError("one or more --select names were not found uniquely")
        manifest = []
        if args.update_reference and args.output_root is not None:
            raise ValidationError("--update-reference uses config reference paths; do not combine it with --output-root")
        output_root = args.output_root or ROOT / ".artifacts/serving" / args.matrix.stem
        if not args.update_reference:
            output_root = validate_output_destination(output_root)
        for index, value in enumerate(selected, 1):
            config_path = ROOT / value
            print(f"[{index}/{len(selected)}] {config_path}")
            if not args.dry_run:
                output_override = None if args.update_reference else output_root / f"{config_path.stem}.jsonl"
                output = run(config_path, runner=args.runner, output_override=output_override,
                             force=args.force, update_reference=args.update_reference)
                manifest.append({"config": value, "result": portable_path(output)})
            else:
                manifest.append({"config": value, "result": None})
        if not args.dry_run:
            destination = ((ROOT / matrix["command_manifest"]) if args.update_reference
                           else output_root / "command_manifest.json")
            destination = validate_output_destination(destination, update_reference=args.update_reference)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"evidence_type": "SIMULATED", "matrix": portable_path(args.matrix),
                "runs": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ValidationError, FileExistsError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
