#!/usr/bin/env python3
"""Analyze paired Phase 8 optimization records and create drift plots."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


METRICS = {
    "prompt_throughput": ("prompt_tokens_per_second", True),
    "generation_throughput": ("generation_tokens_per_second", True),
    "duration": ("time_duration_seconds", False),
}


def load_pairs(path: Path) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        if record.get("warmup") or record.get("return_code") != 0 or record.get("parse_error"):
            continue
        pair_id = int(record["pair_id"]); variant = str(record["variant"])
        if variant in grouped[pair_id]:
            raise ValueError(f"line {line_number}: duplicate {variant} in pair {pair_id}")
        grouped[pair_id][variant] = record
    incomplete = [pair for pair, variants in grouped.items() if set(variants) != {"baseline", "optimized"}]
    if incomplete:
        raise ValueError(f"incomplete pairs: {incomplete}")
    return [{"pair_id": pair, **grouped[pair]} for pair in sorted(grouped)]


def bootstrap_ci(values: list[float], *, seed: int = 20260801, resamples: int = 50_000,
                 statistic: Callable[[list[float]], float] = statistics.median) -> list[float]:
    rng = random.Random(seed); count = len(values)
    samples = sorted(statistic([values[rng.randrange(count)] for _ in range(count)]) for _ in range(resamples))
    return [samples[int(0.025 * resamples)], samples[int(0.975 * resamples)]]


def slope(xs: list[float], ys: list[float]) -> float:
    xmean, ymean = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xmean) ** 2 for x in xs)
    return sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / denominator if denominator else 0.0


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return 0.0
    return statistics.correlation(xs, ys)


def analyze(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("no complete measured pairs")
    output: dict[str, Any] = {"pair_count": len(pairs), "bootstrap_seed": 20260801,
                              "bootstrap_resamples": 50_000, "metrics": {}}
    for name, (field, higher_is_better) in METRICS.items():
        baseline = [float(pair["baseline"][field]) for pair in pairs]
        optimized = [float(pair["optimized"][field]) for pair in pairs]
        differences = [o - b for b, o in zip(baseline, optimized)]
        percents = [100 * (o - b) / b for b, o in zip(baseline, optimized)]
        signs = {"optimized_better": 0, "baseline_better": 0, "tie": 0}
        for difference in differences:
            if difference == 0: signs["tie"] += 1
            elif (difference > 0) == higher_is_better: signs["optimized_better"] += 1
            else: signs["baseline_better"] += 1
        half = len(pairs) // 2
        output["metrics"][name] = {
            "higher_is_better": higher_is_better,
            "baseline": {"mean": statistics.fmean(baseline), "median": statistics.median(baseline),
                         "sample_standard_deviation": statistics.stdev(baseline),
                         "coefficient_of_variation_percent": 100 * statistics.stdev(baseline) / statistics.fmean(baseline)},
            "optimized": {"mean": statistics.fmean(optimized), "median": statistics.median(optimized),
                          "sample_standard_deviation": statistics.stdev(optimized),
                          "coefficient_of_variation_percent": 100 * statistics.stdev(optimized) / statistics.fmean(optimized)},
            "paired_absolute_difference": differences,
            "paired_percent_difference": percents,
            "median_paired_absolute_difference": statistics.median(differences),
            "mean_paired_absolute_difference": statistics.fmean(differences),
            "median_paired_percent_difference": statistics.median(percents),
            "mean_paired_percent_difference": statistics.fmean(percents),
            "minimum_paired_difference": min(differences), "maximum_paired_difference": max(differences),
            "paired_difference_sample_standard_deviation": statistics.stdev(differences),
            "bootstrap_95_ci_median_paired_percent_difference": bootstrap_ci(percents), "sign_count": signs,
            "first_half_median_paired_percent_difference": statistics.median(percents[:half]),
            "second_half_median_paired_percent_difference": statistics.median(percents[half:]),
            "paired_percent_vs_pair_index_slope": slope([float(p["pair_id"]) for p in pairs], percents),
            "paired_percent_vs_pair_index_correlation": correlation([float(p["pair_id"]) for p in pairs], percents),
            "baseline_vs_execution_index_slope": slope([float(p["baseline"]["execution_index"]) for p in pairs], baseline),
            "optimized_vs_execution_index_slope": slope([float(p["optimized"]["execution_index"]) for p in pairs], optimized),
        }
    prompt = output["metrics"]["prompt_throughput"]
    generation = output["metrics"]["generation_throughput"]
    duration = output["metrics"]["duration"]
    prompt_ci = prompt["bootstrap_95_ci_median_paired_percent_difference"]
    drift = (abs(prompt["paired_percent_vs_pair_index_correlation"]) >= 0.5 or
             abs(prompt["first_half_median_paired_percent_difference"] - prompt["second_half_median_paired_percent_difference"]) > 1.0)
    positive = (prompt["median_paired_percent_difference"] > 0 and prompt_ci[0] > 0 and
                prompt["sign_count"]["optimized_better"] / len(pairs) >= 0.70 and
                generation["median_paired_percent_difference"] >= -1.0 and
                duration["median_paired_percent_difference"] <= 1.0 and
                max(prompt["baseline"]["coefficient_of_variation_percent"], prompt["optimized"]["coefficient_of_variation_percent"]) <= 3.0 and not drift)
    generation_ci = generation["bootstrap_95_ci_median_paired_percent_difference"]
    duration_ci = duration["bootstrap_95_ci_median_paired_percent_difference"]
    reject = ((prompt["median_paired_percent_difference"] < 0 and prompt_ci[1] < 0) or
              (generation["median_paired_percent_difference"] < -1.0 and generation_ci[1] < 0) or
              (duration["median_paired_percent_difference"] > 1.0 and duration_ci[0] > 0))
    output["drift_material"] = drift
    output["classification"] = "ACCEPTABLE POSITIVE TREND" if positive else "REJECT" if reject else "INCONCLUSIVE"
    return output


def make_plots(pairs: list[dict[str, Any]], analysis: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True); paths = []
    for metric, filename, ylabel in (
        ("prompt_throughput", "paired_prompt_difference.png", "Optimized - baseline (tokens/s)"),
        ("generation_throughput", "paired_generation_difference.png", "Optimized - baseline (tokens/s)"),
        ("duration", "paired_duration_difference.png", "Optimized - baseline (seconds)"),
    ):
        fig, ax = plt.subplots(); values = analysis["metrics"][metric]["paired_absolute_difference"]
        ax.plot([p["pair_id"] for p in pairs], values, marker="o"); ax.axhline(0)
        ax.set(xlabel="Pair", ylabel=ylabel); ax.grid(True); fig.tight_layout()
        path = output / filename; fig.savefig(path); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots()
    for variant in ("baseline", "optimized"):
        ax.plot([p[variant]["execution_index"] for p in pairs],
                [p[variant]["prompt_tokens_per_second"] for p in pairs], marker="o", label=variant)
    ax.set(xlabel="Execution index", ylabel="Prompt throughput (tokens/s)"); ax.grid(True); ax.legend(); fig.tight_layout()
    path = output / "prompt_throughput_execution_order.png"; fig.savefig(path); plt.close(fig); paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); pairs = load_pairs(args.records); result = analyze(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_plots(pairs, result, args.output_dir)
    print(json.dumps({"pair_count": len(pairs), "classification": result["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
