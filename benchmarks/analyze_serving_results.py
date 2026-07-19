#!/usr/bin/env python3
"""Validate S6 records and produce normalized JSON/CSV/Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.serving_common import (ROOT, ValidationError, analyze_records, load_json,
    portable_path, read_jsonl, read_workload, sha256_file, validate_run_record)


def results_from_manifest(path: Path) -> list[Path]:
    manifest = load_json(path)
    if manifest.get("evidence_type") != "SIMULATED" or not isinstance(manifest.get("runs"), list):
        raise ValidationError("command manifest must contain SIMULATED runs")
    results: list[Path] = []
    for index, run in enumerate(manifest["runs"]):
        if not isinstance(run, dict) or not isinstance(run.get("result"), str):
            raise ValidationError(f"command manifest run {index} lacks a result path")
        candidate = (ROOT / run["result"]).resolve()
        if ROOT not in candidate.parents:
            raise ValidationError("command manifest result path escapes repository")
        results.append(candidate)
    return results


def analyze(path: Path) -> tuple[dict, dict]:
    records = read_jsonl(path)
    runs = [record for record in records if record.get("record_type") == "run"]
    if len(runs) != 1 or records[0] is not runs[0]:
        raise ValidationError("result must begin with one compatible run provenance record")
    validate_run_record(runs[0])
    config = runs[0].get("normalized_config")
    if not isinstance(config, dict) or not isinstance(config.get("slos"), dict):
        raise ValidationError("run provenance lacks normalized config/SLOs")
    workload_records = None
    workload_value = config.get("workload", {}).get("path")
    if isinstance(workload_value, str) and not Path(workload_value).is_absolute():
        workload_path = (ROOT / workload_value).resolve()
        if workload_path.is_file() and sha256_file(workload_path) == runs[0]["workload_sha256"]:
            workload_records = read_workload(workload_path)
    summary = analyze_records(
        [record for record in records if record.get("record_type") != "run"], config["slos"],
        config=config, workload_records=workload_records)
    return summary, runs[0]


def render_markdown(rows: list[tuple[Path, dict, dict]], comparisons: list[str] | None = None) -> str:
    lines = ["# Serving simulation summary", "", "All values are **SIMULATED** metadata-model results, not hardware measurements.",
             "Normalized JSON values are authoritative. Tables display rates and ratios to two decimal places and integer microseconds without additional rounding.", "",
             "| Configuration | Evidence | Status | Submitted | Completed | Completion ratio | Stalled iterations | Req/s | P99 TTFT (us) | P99 E2E (us) | Goodput ratio | KV deferrals | Prefix token hit rate |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def display(value: object) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    for path, summary, run in rows:
        lines.append("| " + " | ".join([
            str(run["normalized_config"]["name"]), "SIMULATED", summary["run_status"],
            display(summary["submitted"]), display(summary["completed"]), display(summary["completion_ratio"]),
            display(summary["stalled_iterations"]),
            display(summary["request_throughput_per_s"]), display(summary["ttft_us"]["p99"]),
            display(summary["e2e_latency_us"]["p99"]), display(summary["goodput_ratio"]),
            display(summary["kv_capacity_deferrals"]), display(summary["prefix_token_hit_rate"])]) + " |")
    if comparisons:
        by_name = {run["normalized_config"]["name"]: (summary, run) for _, summary, run in rows}
        if len(by_name) != len(rows):
            raise ValidationError("comparison inputs contain duplicate configuration names")
        lines += ["", "## Selected comparisons", "",
                  "| Baseline -> candidate | Evidence | Completed | Request/s delta | P99 TTFT delta (us) | Scheduled prefill-token delta |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for comparison in comparisons:
            try:
                baseline_name, candidate_name = comparison.split(":", 1)
                (baseline, baseline_run), (candidate, candidate_run) = by_name[baseline_name], by_name[candidate_name]
            except (ValueError, KeyError) as error:
                raise ValidationError(f"invalid --compare {comparison!r}; use baseline:candidate names") from error
            for field in ("workload_sha256", "seed"):
                if baseline_run.get(field) != candidate_run.get(field):
                    raise ValidationError(f"comparison {comparison!r} has non-equivalent {field}")
            for field in ("backend", "slos"):
                if baseline_run["normalized_config"].get(field) != candidate_run["normalized_config"].get(field):
                    raise ValidationError(f"comparison {comparison!r} has non-equivalent {field}")
            def delta(field: str, nested: str | None = None):
                left = baseline[field] if nested is None else baseline[field][nested]
                right = candidate[field] if nested is None else candidate[field][nested]
                return None if left is None or right is None else right - left
            equivalent_completion = (baseline["completed"] == candidate["completed"] and
                                     baseline["request_count"] == candidate["request_count"])
            lines.append("| " + " | ".join([f"{baseline_name} -> {candidate_name}", "SIMULATED",
                f"{baseline['completed']} -> {candidate['completed']}",
                display(delta("request_throughput_per_s") if equivalent_completion else None),
                display(delta("ttft_us", "p99") if equivalent_completion else None),
                display(delta("scheduled_prefill_tokens"))]) + " |")
        lines += ["", "Rate and latency deltas are N/A when completion populations differ; scheduled-work deltas then describe incompleteness, not an optimization win."]
    lines += ["", "Nearest-rank percentiles use `ceil(p * N)` with one-based clamped ranks; empty populations are N/A.",
              "Exact ITL is available only when decode completion timestamps were emitted; TPOT is otherwise the supported cadence metric."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="*")
    parser.add_argument("--command-manifest", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--compare", action="append", default=[], metavar="BASELINE:CANDIDATE")
    args = parser.parse_args(argv)
    try:
        paths = list(args.results)
        if args.command_manifest:
            paths += results_from_manifest(args.command_manifest)
        if not paths:
            raise ValidationError("provide result files or --command-manifest")
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValidationError("duplicate result path")
        rows = [(path, *analyze(path)) for path in paths]
        payload = [{"result_path": portable_path(path), "summary": summary,
                    "config_name": run["normalized_config"]["name"], "evidence_type": "SIMULATED"}
                   for path, summary, run in rows]
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["config_name", "evidence_type", "run_status",
                    "submitted", "completed", "completion_ratio", "stalled_iterations",
                    "request_throughput_per_s", "p99_ttft_us", "p99_e2e_us", "goodput_ratio",
                    "kv_capacity_deferrals", "prefix_token_hit_rate"])
                writer.writeheader()
                for _, summary, run in rows:
                    writer.writerow({"config_name": run["normalized_config"]["name"], "evidence_type": "SIMULATED",
                        "run_status": summary["run_status"], "submitted": summary["submitted"],
                        "completed": summary["completed"], "completion_ratio": summary["completion_ratio"],
                        "stalled_iterations": summary["stalled_iterations"],
                        "request_throughput_per_s": summary["request_throughput_per_s"],
                        "p99_ttft_us": summary["ttft_us"]["p99"], "p99_e2e_us": summary["e2e_latency_us"]["p99"],
                        "goodput_ratio": summary["goodput_ratio"], "kv_capacity_deferrals": summary["kv_capacity_deferrals"],
                        "prefix_token_hit_rate": summary["prefix_token_hit_rate"]})
        markdown = render_markdown(rows, args.compare)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown, encoding="utf-8")
        else:
            print(markdown, end="")
    except (ValidationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
