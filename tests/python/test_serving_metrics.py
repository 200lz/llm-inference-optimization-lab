import math
import copy

import pytest

from benchmarks.plot_serving_results import plot
from benchmarks.analyze_serving_results import render_markdown
from benchmarks.serving_common import (SIMULATOR_SCHEMA_VERSION, ValidationError, analyze_records,
    nearest_rank, request_metrics, safe_rate)


def request(generated=4, arrival=10, admitted=20, first=50, finish=110):
    times = [] if generated == 0 else ([first] if generated == 1 else
        [first + round((finish - first) * index / (generated - 1)) for index in range(generated)])
    tpot = None if generated < 2 else (finish - first) / (generated - 1)
    return {"record_type": "request", "schema_version": SIMULATOR_SCHEMA_VERSION,
        "evidence_type": "SIMULATED", "request_id": "r", "internal_id": 1,
        "workload_class": "chat", "prefix_group": None, "deadline_us": None, "metadata": {},
        "prompt_tokens": None,
        "arrival_time_us": arrival,
        "admitted_time_us": admitted, "first_token_time_us": first, "finish_time_us": finish,
        "prompt_tokens_original": 2, "prompt_tokens_matched": 0, "prompt_tokens_scheduled": 2,
        "generated_tokens": generated, "max_new_tokens": generated, "final_state": "Finished",
        "reason": None, "decode_token_times_us": times,
        "inter_token_latencies_us": [b - a for a, b in zip(times, times[1:])],
        "queue_delay_us": admitted - arrival,
        "ttft_us": None if generated == 0 else first - arrival,
        "end_to_end_latency_us": finish - arrival, "tpot_us": tpot}


def summary(completed=1):
    return {"record_type": "summary", "schema_version": SIMULATOR_SCHEMA_VERSION,
        "evidence_type": "SIMULATED", "execution_mode": "single_active_fcfs",
        "scheduling_policy": None, "run_status": "completed", "submitted": 1,
        "completed": completed, "cancelled": 0, "rejected": 0, "average_batch_size": 1,
        "max_batch_size": 1, "kv_peak_utilization": None, "kv_current_utilization": None,
        "prefix_token_hit_rate": None, "prefix_lookups": 0,
        "prefix_lookup_hits": 0, "prefix_lookup_misses": 0, "kv_capacity_deferrals": 0,
        "stalled": 0, "eviction_count": 0, "saved_simulated_prefill_tokens": 0,
        "scheduled_prefill_tokens": 2, "scheduled_decode_tokens": 4,
        "submitted_prompt_tokens": 2, "generated_tokens": 4,
        "matched_prefix_tokens": 0, "makespan_us": 110, "scheduling_iterations": None,
        "nonempty_batches": None, "idle_iterations": None, "total_scheduled_sequences": None,
        "max_scheduled_tokens": None, "deferred_requests": 0,
        "current_allocated_blocks": None, "current_free_blocks": None,
        "represented_kv_tokens": None, "cached_blocks": None, "shared_referenced_blocks": None,
        "kv_allocation_failures": 0, "prefix_lookups": 0, "matched_prefix_blocks": 0,
        "cache_eligible_prompt_tokens": 0, "collision_verifications": 0,
        "internal_fragmentation_tokens": None, "peak_allocated_blocks": None}


@pytest.mark.parametrize("generated,expected", [(0, None), (1, None), (2, 60), (4, 20)])
def test_request_metric_formulas(generated, expected) -> None:
    row = request(generated=generated, first=None if generated == 0 else 50)
    metrics = request_metrics(row)
    assert metrics["queue_delay_us"] == 10
    assert metrics["e2e_latency_us"] == 100
    assert metrics["ttft_us"] == (None if generated == 0 else 40)
    assert metrics["tpot_us"] == expected


@pytest.mark.parametrize("p,expected", [(0.5, 2), (0.9, 4), (0.95, 4), (0.99, 4)])
def test_nearest_rank_exact_small_population(p, expected) -> None:
    assert nearest_rank([1, 2, 3, 4], p) == expected
    assert nearest_rank([7], p) == 7
    assert nearest_rank([2, 2, 2], p) == 2


def test_empty_percentile_and_invalid_percentile() -> None:
    assert nearest_rank([], 0.99) is None
    with pytest.raises(ValidationError):
        nearest_rank([1], 0)
    with pytest.raises(ValidationError):
        nearest_rank([], 0)


def test_throughput_goodput_and_partial_slos() -> None:
    result = analyze_records([request(), summary()], {"ttft_slo_us": 39})
    assert result["request_throughput_per_s"] == pytest.approx(10_000)
    assert result["output_token_throughput_per_s"] == pytest.approx(40_000)
    assert result["goodput_ratio"] == 0
    assert result["slo_violations"] == {"ttft": 1, "tpot": 0, "e2e": 0}
    assert result["tpot_us"]["p99"] == 20


def test_undefined_metrics_are_null_not_zero() -> None:
    zero = request(generated=0, first=None)
    zero_summary = summary()
    zero_summary["scheduled_decode_tokens"] = zero_summary["generated_tokens"] = 0
    result = analyze_records([zero, zero_summary], {"tpot_slo_us": 1})
    assert result["ttft_us"]["p50"] is None
    assert result["tpot_us"]["p99"] is None
    assert result["goodput_ratio"] == 1


def test_zero_duration_and_overflow_safe_rate() -> None:
    assert safe_rate(1, 0) is None
    assert math.isfinite(safe_rate(2**63, 2**63))


def test_malformed_result_records() -> None:
    bad = request()
    del bad["generated_tokens"]
    with pytest.raises(ValidationError, match="generated_tokens"):
        analyze_records([bad, summary()], {})
    with pytest.raises(ValidationError, match="exactly one summary"):
        analyze_records([request()], {})


@pytest.mark.parametrize("case", [
    "duplicate_external", "duplicate_native", "unknown_type", "reversed_admission",
    "negative_derived", "invalid_state", "generated_over_limit", "timestamp_count",
    "gap_mismatch", "derived_mismatch", "summary_count", "prompt_conservation",
    "output_conservation", "duplicate_summary",
])
def test_strict_result_validation_rejects_inconsistent_records(case: str) -> None:
    row, total = request(), summary()
    records = [row, total]
    if case == "duplicate_external":
        other = copy.deepcopy(row); other["internal_id"] = 2
        total["submitted"] = total["completed"] = 2
        total["submitted_prompt_tokens"] = total["scheduled_prefill_tokens"] = 4
        total["generated_tokens"] = total["scheduled_decode_tokens"] = 8
        records.insert(1, other)
    elif case == "duplicate_native":
        other = copy.deepcopy(row); other["request_id"] = "other"
        total["submitted"] = total["completed"] = 2
        total["submitted_prompt_tokens"] = total["scheduled_prefill_tokens"] = 4
        total["generated_tokens"] = total["scheduled_decode_tokens"] = 8
        records.insert(1, other)
    elif case == "unknown_type":
        records.insert(1, {"record_type": "mystery"})
    elif case == "reversed_admission":
        row["admitted_time_us"] = 9
    elif case == "negative_derived":
        row["queue_delay_us"] = -1
    elif case == "invalid_state":
        row["final_state"] = "Rejected"
    elif case == "generated_over_limit":
        row["max_new_tokens"] = 3
    elif case == "timestamp_count":
        row["decode_token_times_us"].pop()
    elif case == "gap_mismatch":
        row["inter_token_latencies_us"][0] += 1
    elif case == "derived_mismatch":
        row["ttft_us"] += 1
    elif case == "summary_count":
        total["completed"] = 0
    elif case == "prompt_conservation":
        total["submitted_prompt_tokens"] = 1
    elif case == "output_conservation":
        total["generated_tokens"] = 3
    elif case == "duplicate_summary":
        records.append(copy.deepcopy(total))
    with pytest.raises(ValidationError):
        analyze_records(records, {})


def test_empty_native_result_has_no_vacuous_itl_claim() -> None:
    empty = summary(completed=0)
    empty.update(submitted=0, scheduled_prefill_tokens=0, submitted_prompt_tokens=0,
                 scheduled_decode_tokens=0, generated_tokens=0,
                 average_batch_size=None, max_batch_size=0, makespan_us=0)
    result = analyze_records([empty], {})
    assert result["itl_available"] is False


def test_iteration_validation_and_totals() -> None:
    iteration = {"record_type": "iteration", "schema_version": SIMULATOR_SCHEMA_VERSION,
        "evidence_type": "SIMULATED", "iteration_number": 1, "start_time_us": 20,
        "end_time_us": 30, "policy": "DecodeFirst", "prefill_ids": [1], "decode_ids": [],
        "deferred": [], "scheduled_sequences": 1, "scheduled_prefill_tokens": 2,
        "scheduled_decode_tokens": 0, "total_scheduled_tokens": 2,
        "kv_allocated": 1, "kv_free": 3, "kv_utilization": 0.25,
        "represented_kv_tokens": 2, "internal_fragmentation_tokens": 0,
        "cached_blocks": 0, "shared_referenced_blocks": 0,
        "prefix_hits": 0, "prefix_misses": 0, "matched_tokens": 0,
        "evicted_ids": [], "stall": False}
    row = request(generated=0, first=None, finish=30)
    total = summary()
    total.update(execution_mode="continuous_batching", scheduling_policy="DecodeFirst",
        scheduling_iterations=1, nonempty_batches=1, idle_iterations=0,
        total_scheduled_sequences=1, average_batch_size=1, max_scheduled_tokens=2,
        current_allocated_blocks=1, current_free_blocks=3, peak_allocated_blocks=1,
        kv_current_utilization=0.25, kv_peak_utilization=0.25, represented_kv_tokens=2,
        internal_fragmentation_tokens=0, cached_blocks=0, shared_referenced_blocks=0,
        scheduled_decode_tokens=0, generated_tokens=0, makespan_us=30)
    assert analyze_records([iteration, row, total], {})["iteration_count"] == 1
    for field, value in (("iteration_number", 0), ("end_time_us", 19),
                         ("scheduled_sequences", 2), ("scheduled_decode_tokens", 1),
                         ("kv_utilization", 1.1)):
        bad = copy.deepcopy(iteration); bad[field] = value
        with pytest.raises(ValidationError):
            analyze_records([bad, copy.deepcopy(row), copy.deepcopy(total)], {})


def test_plots_skip_missing_values_and_label_simulated(tmp_path) -> None:
    rows = [{"config_name": "kv_small", "evidence_type": "SIMULATED", "summary": {
        "goodput_ratio": None, "ttft_us": {"p99": 5}, "request_throughput_per_s": 2,
        "stalled_iterations": 1, "internal_fragmentation_tokens": None}}]
    created = plot(rows, tmp_path)
    assert {path.name for path in created} == {
        "configuration_p99_ttft.png", "configuration_request_throughput.png", "kv_blocks_stalls.png"
    }


def test_plot_labels_are_complete_and_simulated(tmp_path, monkeypatch) -> None:
    axes = []
    class Axis:
        def bar(self, *_args, **_kwargs): pass
        def set_title(self, value): self.title = value
        def set_xlabel(self, value): self.xlabel = value
        def set_ylabel(self, value): self.ylabel = value
        def tick_params(self, **_kwargs): pass
    class Figure:
        def tight_layout(self): pass
        def savefig(self, path): path.write_bytes(b"plot")
    def subplots(**_kwargs):
        axis = Axis(); axes.append(axis); return Figure(), axis
    monkeypatch.setattr("benchmarks.plot_serving_results.plt.subplots", subplots)
    monkeypatch.setattr("benchmarks.plot_serving_results.plt.close", lambda _figure: None)
    plot([{"config_name": "one", "evidence_type": "SIMULATED", "summary": {
        "goodput_ratio": 1, "ttft_us": {"p99": 2}, "request_throughput_per_s": 3}}], tmp_path)
    assert axes
    assert all("SIMULATED" in axis.title and axis.xlabel and axis.ylabel for axis in axes)


def test_markdown_display_rounding_and_partial_status_are_explicit(tmp_path) -> None:
    analyzed = analyze_records([request(), summary()], {})
    analyzed["request_throughput_per_s"] = 1.235
    run = {"normalized_config": {"name": "sample"}}
    markdown = render_markdown([(tmp_path / "result.jsonl", analyzed, run)])
    assert "| sample | SIMULATED | completed | 1 | 1 | 1.00 | 0 | 1.24 |" in markdown
    assert "Normalized JSON values are authoritative" in markdown
