# Serving simulation summary

All values are **SIMULATED** metadata-model results, not hardware measurements.
Normalized JSON values are authoritative. Tables display rates and ratios to two decimal places and integer microseconds without additional rounding.

| Configuration | Evidence | Status | Submitted | Completed | Completion ratio | Stalled iterations | Req/s | P99 TTFT (us) | P99 E2E (us) | Goodput ratio | KV deferrals | Prefix token hit rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fcfs_chat_small | SIMULATED | completed | 8 | 8 | 1.00 | 0 | 68.94 | 946 | 3745 | 1.00 | 0 | N/A |
| continuous_chat_small | SIMULATED | completed | 8 | 8 | 1.00 | 0 | 67.97 | 521 | 6230 | 1.00 | 0 | N/A |
| shared_prefix_cache_off | SIMULATED | completed | 8 | 8 | 1.00 | 0 | 110.20 | 556 | 2592 | 1.00 | 0 | N/A |
| shared_prefix_cache_on | SIMULATED | completed | 8 | 8 | 1.00 | 0 | 110.69 | 536 | 2272 | 1.00 | 0 | 0.88 |
| kv_small | SIMULATED | stalled | 12 | 1 | 0.08 | 1 | 605.69 | 256 | 1651 | 1.00 | 1 | N/A |
| kv_large | SIMULATED | completed | 12 | 12 | 1.00 | 0 | 101.08 | 1181 | 8721 | 1.00 | 0 | N/A |
| mixed_low_load | SIMULATED | completed | 12 | 12 | 1.00 | 0 | 35.43 | 1181 | 8721 | 1.00 | 0 | 0.57 |
| mixed_overload | SIMULATED | completed | 12 | 12 | 1.00 | 0 | 458.31 | 11123 | 20683 | 1.00 | 0 | 0.57 |

## Selected comparisons

| Baseline -> candidate | Evidence | Completed | Request/s delta | P99 TTFT delta (us) | Scheduled prefill-token delta |
| --- | --- | ---: | ---: | ---: | ---: |
| fcfs_chat_small -> continuous_chat_small | SIMULATED | 8 -> 8 | -0.97 | -425 | 0 |
| shared_prefix_cache_off -> shared_prefix_cache_on | SIMULATED | 8 -> 8 | 0.49 | -20 | -448 |
| kv_large -> kv_small | SIMULATED | 12 -> 1 | N/A | N/A | -1073 |

Rate and latency deltas are N/A when completion populations differ; scheduled-work deltas then describe incompleteness, not an optimization win.

Nearest-rank percentiles use `ceil(p * N)` with one-based clamped ranks; empty populations are N/A.
Exact ITL is available only when decode completion timestamps were emitted; TPOT is otherwise the supported cadence metric.
