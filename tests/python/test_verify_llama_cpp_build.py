from __future__ import annotations

import stat
from pathlib import Path

import pytest

from scripts import verify_llama_cpp_build as verifier


def write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_success_preserves_output_and_return_code(tmp_path: Path) -> None:
    path = tmp_path / "llama-cli"
    write_executable(path, "printf 'help output\\n'; printf 'diagnostic\\n' >&2; exit 0")

    result = verifier.verify_executable(path, timeout=1)

    assert result.passed
    assert result.stdout == "help output\n"
    assert result.stderr == "diagnostic\n"
    assert result.return_code == 0


def test_nonzero_help_fails_and_preserves_return_code(tmp_path: Path) -> None:
    path = tmp_path / "llama-bench"
    write_executable(path, "printf 'partial\\n'; printf 'failure\\n' >&2; exit 7")

    result = verifier.verify_executable(path, timeout=1)

    assert not result.passed
    assert result.stdout == "partial\n"
    assert result.stderr == "failure\n"
    assert result.return_code == 7


def test_missing_and_non_executable_files_fail(tmp_path: Path) -> None:
    missing = verifier.verify_executable(tmp_path / "missing", timeout=1)
    plain = tmp_path / "plain"
    plain.write_text("not executable", encoding="utf-8")
    non_executable = verifier.verify_executable(plain, timeout=1)

    assert not missing.passed
    assert missing.error == "file does not exist"
    assert not non_executable.passed
    assert non_executable.error == "file is not executable"


def test_timeout_fails_and_preserves_partial_output(tmp_path: Path) -> None:
    path = tmp_path / "llama-server"
    write_executable(path, "printf 'started\\n'; sleep 2")

    result = verifier.verify_executable(path, timeout=0.05)

    assert not result.passed
    assert result.timed_out
    assert result.stdout == "started\n"
    assert result.return_code is None


def test_main_checks_all_required_executables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in verifier.EXECUTABLES:
        write_executable(tmp_path / "bin" / name, f"printf '{name} help\\n'")

    assert verifier.main(["--build-dir", str(tmp_path), "--timeout", "1"]) == 0
    output = capsys.readouterr().out
    assert all(f"[PASS] {name}" in output for name in verifier.EXECUTABLES)
