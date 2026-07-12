#!/usr/bin/env python3
"""Deterministic llama-bench stand-in selected with FAKE_BENCH_BEHAVIOR."""

from __future__ import annotations

import os
import sys
import time


def value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


behavior = os.environ.get("FAKE_BENCH_BEHAVIOR", "success")
if behavior in {"delayed", "timeout"}:
    time.sleep(float(os.environ.get("FAKE_BENCH_DELAY", "0.2" if behavior == "delayed" else "60")))
if behavior == "nonzero":
    print("deterministic failure", file=sys.stderr)
    raise SystemExit(7)
if behavior == "malformed":
    print("not a benchmark table")
    raise SystemExit(0)

prompt, generated, threads = value("-p", "128"), value("-n", "64"), value("-t", "1")
depth = value("-d", "0")
print("| model | size | backend | threads | test | t/s |")
print("| --- | ---: | --- | ---: | ---: | ---: |")
print(f"| fake Q4_K_M | 1 MiB | CPU | {threads} | pp{prompt} @ d{depth} | 100.0 ± 0.0 |")
print(f"| fake Q4_K_M | 1 MiB | CPU | {threads} | tg{generated} @ d{depth} | 20.0 ± 0.0 |")
