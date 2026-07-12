#!/usr/bin/env python3
"""Probe Linux perf events one at a time and emit JSON plus Markdown."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EVENTS = (
    "task-clock", "cpu-clock", "cycles", "instructions", "branches", "branch-misses",
    "cache-references", "cache-misses", "context-switches", "cpu-migrations", "page-faults",
    "minor-faults", "major-faults", "stalled-cycles-frontend", "stalled-cycles-backend",
    "L1-dcache-loads", "L1-dcache-load-misses", "LLC-loads", "LLC-load-misses",
)


def classify_probe(returncode: int, stdout: str, stderr: str) -> tuple[str, str]:
    text = f"{stdout}\n{stderr}".lower()
    if "permission" in text or "access to performance monitoring" in text or "perf_event_paranoid" in text:
        return "permission denied", (stderr or stdout).strip()
    if "not supported" in text or "unsupported" in text:
        return "unsupported", (stderr or stdout).strip()
    if "unknown event" in text or "event syntax error" in text or "cannot find pmu" in text:
        return "unavailable in WSL2", (stderr or stdout).strip()
    if "<not supported>" in text:
        return "unsupported", (stderr or stdout).strip()
    if "<not counted>" in text or "not counted" in text:
        return "noisy or unreliable", (stderr or stdout).strip()
    if returncode == 0:
        return "supported", ""
    return "unsupported", (stderr or stdout).strip() or f"perf exited {returncode}"


def probe_event(perf: str, event: str, command: Sequence[str]) -> dict[str, Any]:
    argv = [perf, "stat", "-x", ",", "-e", event, "--", *command]
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=15, check=False)
        status, diagnostic = classify_probe(result.returncode, result.stdout, result.stderr)
        return {"event": event, "classification": status, "perf_exit_code": result.returncode,
                "diagnostic": diagnostic, "command": argv}
    except subprocess.TimeoutExpired as exc:
        return {"event": event, "classification": "noisy or unreliable", "perf_exit_code": None,
                "diagnostic": f"probe timed out: {exc}", "command": argv}


def render_markdown(data: dict[str, Any]) -> str:
    lines = ["# perf event capability probe", "", f"Generated: `{data['timestamp_utc']}`", "",
             f"perf: `{data.get('perf_path') or 'missing'}`", "",
             "| Event | Classification | Diagnostic |", "| --- | --- | --- |"]
    for item in data["events"]:
        diagnostic = item["diagnostic"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{item['event']}` | {item['classification']} | {diagnostic or '—'} |")
    return "\n".join(lines) + "\n"


def run_probe(perf: str | None = None, command: Sequence[str] = ("true",)) -> dict[str, Any]:
    perf_path = perf or shutil.which("perf")
    data: dict[str, Any] = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "perf_path": perf_path, "events": []}
    if not perf_path:
        data["error"] = "perf binary not found; install linux-tools-common and the linux-tools package matching the current kernel"
        return data
    data["events"] = [probe_event(perf_path, event, command) for event in EVENTS]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--perf")
    args = parser.parse_args()
    data = run_probe(args.perf)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(data), encoding="utf-8")
    print(json.dumps(data))
    return 127 if data.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
