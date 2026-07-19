# Deterministic serving workloads

## Record schema

Each nonblank JSONL line is one `serving-workload-v1` object:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Must equal `serving-workload-v1` |
| `request_id` | nonempty string | yes | Unique within the file |
| `arrival_time_us` | nonnegative integer | yes | Simulated arrival |
| `prompt_token_count` | nonnegative integer | yes | Count-only or exact length |
| `prompt_tokens` | signed int32 array | no | Exact prompt; count must match; empty is valid |
| `max_new_tokens` | nonnegative integer | yes | Output limit; zero is valid |
| `workload_class` | stable class string | yes | One of the six checked-in classes |
| `prefix_group` | nonempty string | no | Generator locality label, not a cache key |
| `deadline_us` | nonnegative integer | no | Workload annotation |
| `metadata` | object | no | Preserved extension data, merged back into request results by Python |

Unknown top-level fields are rejected; extensions belong under `metadata`.
Schema mismatch, wrong types, duplicate IDs, negative arrivals, count/token
mismatch, malformed JSON, and blank lines fail with the JSONL line number.
Count-only records remain supported; exact-token records enable cache studies.
`prefix_group`, `deadline_us`, and `metadata` are carried in the Python result
request envelope by external ID even though the dependency-free native core does
not use them for scheduling. Equal-arrival requests are ordered lexicographically
by the exact external UTF-8 request-ID string (`"10"` sorts before `"2"`),
independently of JSONL line order.
An oversized full prompt is rejected unless a committed prefix hit reduces its
unmatched work to the configured full-prefill budget.

## Generator and manifest

The generator supports `fixed_interval`, `burst`, `exponential`, and `manual`.
Exponential gaps use
`floor(-mean * log1p(-random.Random(seed).random()))`; accumulated integer
microseconds are non-decreasing. Output is identical for the same config/seed
and is not overwritten without `--force`.

The sidecar `serving-workload-manifest-v1` records exact generator config,
seed, schema, class, count, generator path, and workload SHA-256. Timestamp is
null by design. Checked-in classes are:

- `chat`: short/medium count-only prompts, medium output, low reuse;
- `shared_system_prompt`: 64 identical exact tokens plus a private suffix;
- `coding_agent`: synthetic shared system/tool tokens, suffix, longer output;
- `mixed`: chat, summary-shaped, coding-like, prefix/non-prefix traffic;
- `overload`: service-demand-heavy requests with faster arrivals;
- `burst`: multiple request IDs at the same timestamp.

Synthetic token IDs encode no copyrighted or proprietary prompt text.
