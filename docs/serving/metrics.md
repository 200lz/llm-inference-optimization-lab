# Serving metrics and evidence semantics

## Clock and population conventions

Simulation metrics use integer simulated timestamps; real backend metrics use a monotonic clock. For request `r`:

- `a_r`: arrival time
- `s_r`: first admission/backend-start time
- `f_r`: terminal time (`Finished`; cancellation time for cancellation-only metrics)
- `t_r,i`: emission time of output token `i`, indexed from zero
- `n_r`: number of emitted output tokens

The default latency population contains successfully `Finished` requests. Cancelled requests are counted and reported separately, never silently dropped into successful latency percentiles. Preempted requests remain part of the successful population if they later finish. Report count, arithmetic mean, and p50/p90/p95/p99 using nearest-rank percentiles; do not report a percentile for an empty population. Durations use the configured tick unit in raw output and an explicit converted unit in summaries.

## Request and token latency

| Metric | Definition | Edge cases |
| --- | --- | --- |
| Time to first token (TTFT) | `t_r,0 - a_r` | Undefined when `n_r = 0` |
| Time per output token (TPOT) | `(t_r,n-1 - t_r,0) / (n_r - 1)` | Undefined when `n_r < 2`; do not substitute end-to-end latency |
| Inter-token latency (ITL) | Per-token samples `t_r,i - t_r,i-1` for `i >= 1` | Report request-weighted and token-weighted aggregates separately |
| End-to-end latency | `f_r - a_r` | Successful requests only by default |
| Initial queue delay | `s_r - a_r` | Includes all time before first service, including cache/capacity blocking |
| Total waiting time | Sum of all `Waiting` intervals | Separate companion metric; includes waits after preemption |

TTFT includes queueing and prefill. TPOT describes the cadence after the first token; ITL preserves stalls that an average can hide. When streaming emission is modeled separately from backend completion, `t_r,i` is the visible emission event; otherwise the completion timestamp is used and this limitation is recorded.

## Run-level rate and SLO metrics

Let the observation window be `[T_start, T_end)`, normally the first arrival through the last terminal event after drain. Warm-up or steady-state windows must be explicit. Requests/tokens crossing a clipped window are handled by a declared inclusion policy; the default full-drain report includes completed work only and also reports unfinished counts.

- **Output-token throughput:** emitted output tokens divided by window duration.
- **Total-token throughput:** processed uncached prefill tokens plus emitted decode tokens divided by window duration. Always name this separately from output-token throughput.
- **Request throughput:** finished requests divided by window duration.
- **Goodput:** finished requests satisfying every configured SLO divided by window duration. The run configuration must state threshold values (for example TTFT, TPOT, and/or end-to-end), conjunction logic, and whether requests lacking enough tokens for TPOT are ineligible or evaluated without that constraint. Also report `SLO-met / SLO-eligible` counts.

Throughput never uses the sum of per-request latencies as its denominator. A run with no positive-duration window reports rates as undefined.

## Cache and scheduler metrics

### Prefix-cache hit rate

The primary metric is token-weighted reusable-prefix hit rate:

`reused block-aligned prompt tokens / cache-eligible prompt tokens`.

Also report request hit rate (`requests reusing at least one token / eligible requests`) and lookup counts. Token-count-only synthetic prompts are ineligible, not misses. A lookup hit is not counted unless its blocks are successfully retained for the request. Saved compute is a separate analytical estimate unless confirmed by a real measurement.

### KV utilization

Instantaneous utilization is `allocated KV blocks / total KV blocks`. S4 has
only private blocks; a future shared prefix block would still be counted once
per physical block. The run summary is time-weighted utilization:

`sum(allocated_blocks * interval_duration) / (total_blocks * observation_duration)`.

S4 reports peak allocated blocks and peak utilization. Its absolute internal
fragmentation metric is
`allocated_blocks * block_size_tokens - represented_tokens`; this counts unused
modeled token slots, not bytes. A later summary may additionally report that
value divided by total allocated token slots as a ratio. Planner
`KVCapacity` deferrals count once per request per iteration and remain separate
from manager allocation failures. The allocation-failure count includes only
otherwise valid prompt or decode-boundary operations that lack physical blocks;
validation, arithmetic, epoch, and host-container failures are excluded. A
deferred-only stalled iteration counts its KV deferrals and stall occurrence
but performs no backend work and advances no simulated time. A zero-capacity or zero-duration
configuration is invalid/undefined rather than zero utilization.

Prefix caching reports successful deterministic eviction count only. There is
no separate eviction-failure metric: planner `KVCapacity` deferral is expected
control flow and is not an eviction failure.

### Preemption count

This is a future metric and is not emitted by S4. A future implementation would
count each active-to-`Preempted` transition and report total preemptions,
requests preempted at least once, preemptions per finished request, and
recomputed prompt tokens. Ordinary iteration boundaries and voluntary
completion are not preemptions.

### Batch utilization

For iteration `j`, let `u_j` be scheduled token work (uncached prefill tokens plus decode-token slots), `c_j` the configured maximum token work for that iteration, and `d_j` its duration. Primary batch utilization is compute-time weighted:

`sum((u_j / c_j) * d_j) / sum(d_j)`.

Report mean and distribution of `u_j / c_j`, active sequences per iteration, and prefill/decode work mix. If backend constraints make `c_j` shape-dependent, record the effective capacity on every iteration. Queue-empty idle time is reported as backend idle fraction, not folded into batch utilization.

## Aggregation rules

- Preserve raw request, token, iteration, cache, and event records; derive summaries from them.
- State whether an aggregate is request-weighted, token-weighted, iteration-weighted, or time-weighted.
- Use a documented nearest-rank percentile implementation and stable request/event ordering.
- Report numerator, denominator, eligible count, excluded/cancelled/unfinished counts, and units alongside rates.
- Never coerce undefined values to zero. CSV uses an empty field plus a reason column; structured output uses `null` plus a reason.
- Validate conservation: arrivals equal finished + cancelled + unfinished; emitted tokens equal token records; block allocation/release deltas reconcile with final occupancy.

## Evidence labeling

Metric names describe what was calculated, not how trustworthy or physical it is. Each record therefore includes `evidence_kind`:

- `simulated`: calculated from deterministic simulator events. Example: simulated TTFT under FCFS.
- `analytical_estimate`: calculated directly from a formula/fit. Example: estimated compute time saved by a prefix hit.
- `llama_cpp_measurement`: observed through llama.cpp with recorded clock, host, model, build, command, warm-up, and repetition methodology.

The same field also carries `backend`, `scheduler`, `workload_fingerprint`, and `configuration_fingerprint` where applicable. Summaries may compare evidence kinds in adjacent, visibly labeled columns, but must not pool their samples or describe agreement as validation without an error analysis.

## Minimum output schema

All result tables include `run_id`, schema version, evidence kind, metric name, scope (`request`, `token`, `iteration`, `cache`, or `run`), value, unit, aggregation, numerator/denominator where relevant, sample count, and configuration/workload fingerprints. Request rows add request ID and lifecycle timestamps; iteration rows add batch composition and effective capacity. This is a design contract, not a Phase S0 implementation.

## S5 prefix-cache metrics

S5 records lookup/hit/miss counts, matched blocks/tokens, eligible exact-token
prompt tokens, reused requests, collision verifications, cached and referenced
shared block counts, and evictions. `prefix_token_hit_rate` is matched tokens
divided by cache-eligible complete-block prompt tokens looked up; it is
undefined for a zero denominator.

Physical represented tokens sum block occupancy once, whether a cached block
has zero, one, or several request references. Logical matched tokens count per
request reuse. Original prompt tokens, matched tokens, and scheduled prefill
tokens are separate trace fields. `saved_simulated_prefill_tokens` is modeled
work avoided; it is not actual latency, bandwidth, energy, GPU, or MN-Core
evidence.

`AllEligibleBlocksHit` counts a lookup where every complete exact-token prompt
block matched. A partial prompt tail may remain and contribute scheduled
prefill tokens. Count-only requests are publication-ineligible and do not
contribute cache-eligible prompt tokens or cache misses.

## S6 implemented aggregation

S6 uses nearest-rank percentiles. For sorted population size `N`, the one-based
rank is `ceil(p * N)`, clamped to `[1, N]`. Empty populations produce `null`/
N/A; one observation returns that observation; duplicate observations are not
deduplicated. The implemented percentiles are p50, p90, p95, and p99 and do not
use NumPy defaults.

Normalized JSON values are authoritative. Generated Markdown and README tables
display rates and ratios to two decimal places and integer-microsecond
percentiles as integers; display rounding never changes comparison inputs.

The full-drain observation window is first arrival through last successful
finish. Request, output-token, and total-token throughput divide by that
positive simulated duration; zero duration produces null. Queue depth is
derived from ordered arrival/admission transitions. Native decode-completion
timestamps are included, so modeled token gaps are reconstructable; reports
still use TPOT as the primary cadence metric.

Checked-in SLO values are educational simulated thresholds, never production
defaults. A finished request contributes to goodput when every configured,
applicable TTFT, TPOT, and E2E threshold passes. A structurally undefined
metric (zero-output TTFT or fewer-than-two-token TPOT) is inapplicable rather
than silently zero. Reports include good requests per simulated second, the
good/finished ratio, and violation counts by threshold.

The versioned request workload is JSONL. Unknown top-level fields are rejected;
extensions belong under the preserved `metadata` object. The Python layer
reports JSONL line numbers, rejects schema mismatches and duplicate IDs, and
translates validated data through a strict temporary TSV interface. Native
request, iteration, and summary JSONL uses the closed
`serving-simulator-v2` schema; the Python run provenance row uses the separately
closed `serving-result-v1` schema. Required fields cannot be omitted, nullable
fields must still be present, and unknown fields are rejected. The analyzer
validates the entire stream before calculating any metric.

For continuous batching, iteration sums must equal summary scheduled prefill,
decode, and sequence totals. Nonempty, idle, stalled, deferred, maximum-batch,
maximum-token, prefix, eviction, and KV-capacity counts are reconciled with the
trace. Each admitted request has one explicit prefill appearance (including a
zero-work full-prefix hit), and its decode appearances equal its generated
token count. Iteration request IDs must exist in the request records. Request
prompt/output totals and lifecycle states reconcile with the single terminal
summary and, when the submitted workload is available, its immutable envelope.
Single-active FCFS emits no iteration records, so its prompt/decode conservation
is checked directly between request records and summary totals. See the
[closed result envelope](workload_schema.md#closed-result-envelope).
