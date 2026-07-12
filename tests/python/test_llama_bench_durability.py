from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import run_llama_bench as harness


FIXTURE = Path(__file__).parents[1] / "fixtures" / "llama_bench_success.txt"


def config(tmp_path: Path, *, repetitions: int = 2, warmups: int = 0, timeout: float = 1) -> harness.BenchmarkConfig:
    return harness.BenchmarkConfig(
        executable=Path("/fake/llama-bench"), model=Path("/fake/model.gguf"), threads=(1,),
        prompt_tokens=(128,), generated_tokens=(32,), batch_sizes=(64,), context_sizes=(512,),
        ubatch_sizes=(64,), repetitions=repetitions, warmup_runs=warmups,
        timeout_seconds=timeout, output_directory=tmp_path, mmap=True, cpu_only=True,
    )


def success() -> dict[str, object]:
    return {"stdout": FIXTURE.read_text(encoding="utf-8"), "stderr": "", "return_code": 0,
            "elapsed_seconds": 0.01, "timed_out": False}


def failed() -> dict[str, object]:
    return {"stdout": "", "stderr": "failed", "return_code": 7,
            "elapsed_seconds": 0.01, "timed_out": False}


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_are_persisted_and_visible_before_next_invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    def execute(command: object, timeout: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert len(records(tmp_path / "run.jsonl")) == 1
        return success()
    monkeypatch.setattr(harness, "execute", execute)
    raw, _, _ = harness.run(config(tmp_path), tmp_path)
    assert len(records(raw)) == 2


def test_append_flushes_and_fsyncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(harness.os, "fsync", lambda fd: calls.append(fd))
    path = tmp_path / "records.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        harness.append_record(stream, {"invocation_id": "one"})
        assert json.loads(path.read_text(encoding="utf-8"))["invocation_id"] == "one"
    assert len(calls) == 1


def test_resume_skips_completed_without_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    def interrupt_second(command: object, timeout: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2: raise KeyboardInterrupt
        return success()
    monkeypatch.setattr(harness, "execute", interrupt_second)
    with pytest.raises(KeyboardInterrupt): harness.run(config(tmp_path), tmp_path)
    monkeypatch.setattr(harness, "execute", lambda command, timeout: success())
    raw, _, _ = harness.run(config(tmp_path), tmp_path, options=harness.RunOptions(resume=True))
    result = records(raw)
    assert len(result) == 2
    assert len({row["invocation_id"] for row in result}) == 2


def test_retry_failures_reruns_only_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter((success(), failed()))
    monkeypatch.setattr(harness, "execute", lambda command, timeout: next(outcomes))
    raw, _, _ = harness.run(config(tmp_path), tmp_path)
    calls = 0
    def retry(command: object, timeout: float) -> dict[str, object]:
        nonlocal calls; calls += 1; return success()
    monkeypatch.setattr(harness, "execute", retry)
    harness.run(config(tmp_path), tmp_path, options=harness.RunOptions(resume=True, retry_failures=True))
    result = records(raw)
    assert calls == 1 and len(result) == 3
    assert sum(row["status"] == "success" for row in result) == 2


def test_truncated_final_jsonl_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"; path.write_text('{"invocation_id":"one"}\n{"broken":', encoding="utf-8")
    with pytest.raises(harness.ConfigError, match="truncated final line"):
        harness.read_jsonl(path)


def test_incompatible_resume_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "execute", lambda command, timeout: success())
    harness.run(config(tmp_path, repetitions=1), tmp_path)
    with pytest.raises(harness.ConfigError, match="incompatible"):
        harness.run(config(tmp_path, repetitions=2), tmp_path, options=harness.RunOptions(resume=True))


def test_keyboard_interrupt_preserves_records_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter((success(), KeyboardInterrupt()))
    def execute(command: object, timeout: float) -> dict[str, object]:
        value = next(outcomes)
        if isinstance(value, BaseException): raise value
        return value
    monkeypatch.setattr(harness, "execute", execute)
    with pytest.raises(KeyboardInterrupt): harness.run(config(tmp_path), tmp_path)
    assert len(records(tmp_path / "run.jsonl")) == 1
    summary = json.loads((tmp_path / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"] == "interrupted" and summary["completed_invocations"] == 1
    assert not (tmp_path / "normalized.csv").exists()


def test_timeout_record_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "execute", lambda command, timeout: {
        "stdout": "", "stderr": "", "return_code": None, "elapsed_seconds": timeout, "timed_out": True})
    raw, _, _ = harness.run(config(tmp_path, repetitions=1), tmp_path)
    assert records(raw)[0]["status"] == "timed_out"
    summary = json.loads((tmp_path / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["timed_out_invocations"] == 1


def test_progress_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(harness, "execute", lambda command, timeout: success())
    harness.run(config(tmp_path, repetitions=1), tmp_path)
    output = capsys.readouterr().out
    assert "whole_matrix_timeout=unlimited" in output
    assert "[1/1] case=p128-n32-t1" in output
    assert "measured=1/1" in output and "timeout=1s" in output
    assert "return_code=0" in output and "pp=100.0 tg=20.0" in output


def test_incomplete_run_prevents_csv_unless_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "execute", lambda command, timeout: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): harness.run(config(tmp_path), tmp_path)
    assert not (tmp_path / "normalized.csv").exists()


def test_complete_run_generates_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "execute", lambda command, timeout: success())
    _, csv_path, count = harness.run(config(tmp_path, repetitions=1), tmp_path)
    assert count == 1 and csv_path.exists()
    assert "prompt_tokens_per_second" in csv_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(("behavior", "status"), [
    ("success", "success"), ("delayed", "success"), ("nonzero", "failed"),
    ("timeout", "timed_out"), ("malformed", "parse_error"),
])
def test_fake_executable_behaviors(behavior: str, status: str, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = Path(__file__).parents[1] / "fixtures" / "fake_llama_bench.py"
    monkeypatch.setenv("FAKE_BENCH_BEHAVIOR", behavior)
    monkeypatch.setenv("FAKE_BENCH_DELAY", "0.02")
    result = harness.execute([str(executable), "-p", "128", "-n", "32", "-t", "1", "-d", "352"],
                             0.005 if behavior == "timeout" else 1)
    actual = "timed_out" if result["timed_out"] else "failed" if result["return_code"] != 0 else "success"
    if actual == "success":
        try: harness.parse_llama_bench(str(result["stdout"]))
        except ValueError: actual = "parse_error"
    assert actual == status
