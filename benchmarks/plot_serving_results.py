#!/usr/bin/env python3
"""Create deterministic, separately labeled plots from normalized S6 summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from benchmarks.serving_common import ValidationError


def plot(rows: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def bar(filename: str, title: str, ylabel: str, values: list[tuple[str, float | int | None]]) -> None:
        available = [(name, value) for name, value in values if value is not None]
        if not available:
            return
        figure, axis = plt.subplots(figsize=(max(6, len(available) * 0.8), 4))
        axis.bar([name for name, _ in available], [value for _, value in available])
        axis.set_title(f"SIMULATED — {title}")
        axis.set_xlabel("Configuration")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path)
        plt.close(figure)
        created.append(path)

    names = [(row["config_name"], row["summary"]) for row in rows]
    bar("scheduler_goodput.png", "scheduler/configuration goodput", "Goodput ratio",
        [(name, summary.get("goodput_ratio")) for name, summary in names])
    bar("configuration_p99_ttft.png", "configuration P99 TTFT", "P99 TTFT (us)",
        [(name, summary.get("ttft_us", {}).get("p99")) for name, summary in names])
    bar("configuration_request_throughput.png", "completed request throughput",
        "Requests / simulated second", [(name, summary.get("request_throughput_per_s")) for name, summary in names])
    bar("prefix_scheduled_prefill_tokens.png", "prefix workload scheduled prefill work",
        "Scheduled prefill tokens", [(name, summary.get("scheduled_prefill_tokens")) for name, summary in names
                                      if "prefix" in name])
    bar("kv_blocks_stalls.png", "KV configurations and stalled iterations", "Stalled iterations",
        [(name, summary.get("stalled_iterations")) for name, summary in names if name.startswith("kv_")])
    bar("block_size_fragmentation.png", "block size and final fragmentation",
        "Internal fragmentation tokens", [(name, summary.get("internal_fragmentation_tokens"))
                                           for name, summary in names if name.startswith("block_size_")])
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = json.loads(args.summary.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get("evidence_type") != "SIMULATED" for row in rows):
            raise ValidationError("summary must be a SIMULATED normalized summary array")
        created = plot(rows, args.output_dir)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        parser.error(str(error))
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
