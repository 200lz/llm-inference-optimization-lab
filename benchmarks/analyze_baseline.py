#!/usr/bin/env python3
"""Aggregate normalized llama-bench CSV measurements."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = ("test_type", "prompt_token_count", "generated_token_count",
                "thread_count", "model", "quantization")


def quantization_from_model(model: str) -> str:
    lowered = model.lower()
    for quantization in ("q2_k", "q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0"):
        if quantization in lowered:
            return quantization.upper()
    return "unknown"


def analyze(path: Path, model_filename: str | None = None) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            model = model_filename or row["model_name"]
            common = (int(row["prompt_tokens"]), int(row["generated_tokens"]),
                      int(row["thread_count"]), model, quantization_from_model(model))
            for test_type, field in (("prompt_processing", "prompt_tokens_per_second"),
                                     ("generation", "generation_tokens_per_second")):
                if row[field]:
                    grouped[(test_type, *common)].append(float(row[field]))
    results = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        mean = statistics.fmean(values)
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        results.append(dict(zip(GROUP_FIELDS, key)) | {
            "count": len(values), "mean": mean, "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean if mean else None,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="normalized benchmark CSV")
    parser.add_argument("--model-filename", help="exact model artifact used")
    parser.add_argument("--output", type=Path, help="write aggregate JSON here")
    args = parser.parse_args()
    results = analyze(args.csv, args.model_filename)
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
