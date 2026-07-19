# Serving workload schema

The canonical replay format is `serving-workload-v1`: one validated request per
nonblank JSONL line, accompanied by a `serving-workload-manifest-v1` sidecar.
Count-only prompts and exact signed-int32 token arrays are supported. Request IDs
are unique, arrivals and output limits are nonnegative, and unknown top-level
fields are rejected in favor of the `metadata` object. Parse failures identify
the JSONL line; incompatible versions are rejected explicitly.

The complete field table, deterministic arrival algorithms, workload classes,
manifest contract, and external-ID tie rule are documented in
[Deterministic serving workloads](workloads.md).

## Closed result envelope

The corresponding native replay output uses `serving-simulator-v2`. It is a
closed JSONL schema: every request, iteration, and terminal summary field is
required, except that fields explicitly documented as nullable must still be
present with `null`. Unknown fields, missing fields, Boolean values in integer
positions, invalid enums, and incompatible versions are rejected before metric
calculation. The Python provenance wrapper remains `serving-result-v1` and is
closed separately.

Request records preserve `workload_class`, nullable `prefix_group`, nullable
`deadline_us`, `metadata`, and the nullable exact `prompt_tokens` array from the
submitted workload. Count-only requests carry `prompt_tokens: null`. Lifecycle
and derived timestamps are nullable only when the lifecycle makes them
undefined. Decode timestamp and inter-token-gap arrays are always present,
including as empty arrays for zero-output requests.

Continuous runs require iteration records and non-null iteration/KV summary
gauges. Single-active FCFS has no iteration records; iteration-only and KV
gauges in its summary are present as `null`, not fabricated as zero. Exactly
one summary is required and it must be the final native record. Validation
reconciles workload envelopes, the external/native ID bijection, per-request
prompt and decode work, iteration work and request appearances, and terminal
summary counts and totals. The authoritative field sets live in
`benchmarks/serving_common.py`; the native runner and regression fixtures are
validated against those same descriptors.
