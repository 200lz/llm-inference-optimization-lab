#!/usr/bin/env python3
"""Verify the required llama.cpp Release executables without running inference."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXECUTABLES = ("llama-bench", "llama-cli", "llama-server")
DEFAULT_BUILD_DIR = Path("third_party/llama.cpp/build-release")


@dataclass(frozen=True)
class VerificationResult:
    name: str
    path: Path
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.timed_out and self.return_code == 0


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def verify_executable(path: Path, timeout: float) -> VerificationResult:
    if not path.is_file():
        return VerificationResult(path.name, path, "", "", None, error="file does not exist")
    if not os.access(path, os.X_OK):
        return VerificationResult(path.name, path, "", "", None, error="file is not executable")

    try:
        completed = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return VerificationResult(
            path.name,
            path,
            _text(exc.stdout),
            _text(exc.stderr),
            None,
            timed_out=True,
            error=f"--help timed out after {timeout:g} seconds",
        )
    except OSError as exc:
        return VerificationResult(path.name, path, "", "", None, error=str(exc))

    return VerificationResult(
        path.name, path, completed.stdout, completed.stderr, completed.returncode
    )


def verify_build(build_dir: Path, timeout: float) -> list[VerificationResult]:
    binary_dir = build_dir / "bin"
    return [verify_executable(binary_dir / name, timeout) for name in EXECUTABLES]


def print_result(result: VerificationResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    return_code = "not available" if result.return_code is None else str(result.return_code)
    print(f"[{status}] {result.name}")
    print(f"path: {result.path}")
    print(f"return code: {return_code}")
    print(f"timed out: {str(result.timed_out).lower()}")
    if result.error:
        print(f"error: {result.error}")
    print("--- stdout ---")
    print(result.stdout, end="" if result.stdout.endswith("\n") or not result.stdout else "\n")
    print("--- stderr ---")
    print(result.stderr, end="" if result.stderr.endswith("\n") or not result.stderr else "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"llama.cpp build directory (default: {DEFAULT_BUILD_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="timeout in seconds for each --help invocation (default: 10)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = verify_build(args.build_dir, args.timeout)
    for index, result in enumerate(results):
        if index:
            print()
        print_result(result)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
