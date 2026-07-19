#!/usr/bin/env python3
"""Validate JSON inputs, invoke the strict native TSV boundary, and add provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.serving_common import (ROOT, RESULT_SCHEMA_VERSION, SIMULATOR_SCHEMA_VERSION,
    ValidationError, load_json, provenance, read_jsonl, read_workload, resolve_repo_path,
    validate_native_records, validate_output_destination, validate_serving_config,
    validate_workload_manifest)


def runner_command(config: dict, runner: Path, input_tsv: Path, output: Path) -> list[str]:
    backend = config["backend"]
    prefix = config["prefix_cache"]
    kv = config["kv_cache"]
    policy = config["scheduling_policy"]
    return [str(runner), "--input", str(input_tsv), "--output", str(output),
        "--mode", config["execution_mode"], "--policy", policy,
        "--max-num-sequences", str(config["max_num_sequences"]),
        "--max-batched-tokens", str(config["max_batched_tokens"]),
        "--kv-total-blocks", str(kv["total_blocks"]), "--kv-block-size", str(kv["block_size_tokens"]),
        "--prefix-cache", "1" if prefix["enabled"] else "0", "--namespace", prefix["namespace"],
        "--salt", prefix["salt"], "--prefill-base-us", str(backend["prefill_base_us"]),
        "--prefill-per-token-us", str(backend["prefill_per_token_us"]),
        "--prefill-per-sequence-us", str(backend["prefill_per_active_sequence_us"]),
        "--decode-base-us", str(backend["decode_base_us"]),
        "--decode-per-sequence-us", str(backend["decode_per_active_sequence_us"]),
        "--batch-base-us", str(backend["batch_base_us"]),
        "--batch-prefill-per-token-us", str(backend["batch_prefill_per_token_us"]),
        "--batch-decode-per-sequence-us", str(backend["batch_decode_per_sequence_us"]),
        "--batch-active-overhead-us", str(backend["batch_active_sequence_overhead_us"])]


def write_tsv(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens\n")
        ordered = sorted(records, key=lambda record: (record["arrival_time_us"], record["request_id"]))
        for internal_id, record in enumerate(ordered, 1):
            fields = [str(internal_id), record["request_id"], record["workload_class"],
                      str(record["arrival_time_us"]), str(record["prompt_token_count"]),
                      ",".join(map(str, record["prompt_tokens"])) if "prompt_tokens" in record else "-",
                      str(record["max_new_tokens"])]
            if any(any(control in field for control in ("\t", "\n", "\r")) for field in fields):
                raise ValidationError("TSV string fields cannot contain tabs or line breaks")
            stream.write("\t".join(fields) + "\n")


def run(config_path: Path, *, runner: Path, output_override: Path | None = None,
        workload_override: Path | None = None, manifest_override: Path | None = None,
        force: bool = False, update_reference: bool = False) -> Path:
    config = validate_serving_config(load_json(config_path), config_path)
    workload_path = workload_override or resolve_repo_path(config["workload"]["path"], config_path)
    manifest_path = manifest_override or resolve_repo_path(config["workload"]["manifest_path"], config_path)
    output = output_override or resolve_repo_path(config["output_path"], config_path)
    output = validate_output_destination(output, update_reference=update_reference)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; use --force")
    records = read_workload(workload_path)
    manifest = load_json(manifest_path)
    validate_workload_manifest(manifest, records, workload_path, expected_seed=config["run_seed"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="serving-s6-") as directory:
        input_tsv = Path(directory) / "input.tsv"
        native_output = Path(directory) / "native.jsonl"
        write_tsv(records, input_tsv)
        command = runner_command(config, runner, input_tsv, native_output)
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"native runner failed ({completed.returncode}): {completed.stderr.strip()}")
        native_records = read_jsonl(native_output)
        if not native_records or native_records[-1].get("record_type") != "summary":
            raise ValidationError("native runner did not emit a terminal summary")
        original_by_id = {record["request_id"]: record for record in records}
        for record in native_records:
            if record.get("record_type") == "request":
                source = original_by_id.get(record.get("request_id"))
                if source is None:
                    raise ValidationError("native runner emitted unknown request_id")
                record["workload_class"] = source["workload_class"]
                record["prefix_group"] = source.get("prefix_group")
                record["deadline_us"] = source.get("deadline_us")
                record["metadata"] = source.get("metadata", {})
                record["prompt_tokens"] = source.get("prompt_tokens")
        validate_native_records(native_records, config=config, workload_records=records)
        invocation = [sys.executable, "benchmarks/run_serving_simulation.py", str(config_path),
                      "--runner", str(runner), "--output", str(output)]
        if workload_override is not None:
            invocation += ["--workload", str(workload_path)]
        if manifest_override is not None:
            invocation += ["--manifest", str(manifest_path)]
        if force:
            invocation.append("--force")
        if update_reference:
            invocation.append("--update-reference")
        header = {"record_type": "run", "schema_version": RESULT_SCHEMA_VERSION,
                  "execution_mode": config["execution_mode"],
                  "scheduling_policy": None if config["execution_mode"] == "single_active_fcfs" else config["scheduling_policy"],
                  **provenance(config, workload_path, manifest, [invocation, command], runner=runner)}
        with output.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            for record in native_records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/debug/serving-benchmark-runner")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update-reference", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = run(args.config, runner=args.runner, output_override=args.output,
                     workload_override=args.workload, manifest_override=args.manifest,
                     force=args.force, update_reference=args.update_reference)
    except (ValidationError, FileExistsError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"SIMULATED result: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
