# Prefill versus decode scaling

## Run status

- **Verified conclusion:** The planned real benchmark was started on 2026-07-12 and interrupted at the user's direction after the background terminal produced no output or observable state change for more than ten minutes.
- **Verified conclusion:** The interrupted harness exited with code 1 and wrote no raw JSONL or normalized CSV because result files are finalized only after the matrix completes.
- **Verified conclusion:** No Phase 4 throughput values, plots, or analysis JSON are available, and this report does not substitute baseline data or fabricate results.
- **Verified conclusion:** The benchmark infrastructure now provides flushed per-invocation progress, append/flush/fsync JSONL persistence, atomic incremental summaries, configuration-validated resume, selective failure retry, child termination on interruption, and incomplete-output safeguards. These capabilities have been tested with a fake executable only; the real Phase 4 matrix remains incomplete.

## Environment and experiment

- **Verified conclusion:** Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`; quantization: Q4_K_M.
- **Verified conclusion:** CPU: AMD Ryzen 7 5800H with Radeon Graphics; environment: WSL2 Linux.
- **Verified conclusion:** llama.cpp commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`; CPU-only, mmap enabled, batch 512, ubatch 512, and no GPU offload.
- **Verified conclusion:** Prompt scaling is p={32,128,512,1024,2048}, n=64, t=4. Generation scaling is p=512, n={16,64,128}, t=4. Thread scaling is p={128,1024}, n=64, t={1,2,4,8}.
- **Verified conclusion:** Deduplication produces 13 unique cases, 13 warm-ups, and 65 measured runs (78 invocations total).
- **Verified conclusion:** The focused union is preferable to the 60-case full Cartesian product (360 invocations including warm-ups) because each slice changes one independent variable and avoids unnecessary, harder-to-interpret combinations.
- **Verified conclusion:** Each subprocess has a 900-second timeout. The sequential matrix has no separate whole-run timeout.

## Concepts

- **Verified conclusion:** Prompt processing, or prefill, evaluates the supplied prompt and constructs the KV cache while allowing work across prompt tokens to be batched.
- **Verified conclusion:** Autoregressive decode generates tokens sequentially, with each next token depending on prior generated tokens while reusing the KV cache.
- **Hypothesis:** Prefill and decode may respond differently to thread count and sequence length because their available parallel work and sequential dependencies differ.
- **Verified conclusion:** Tokens-per-second measurements alone cannot establish compute-bound or memory-bound behavior.

## Results, scaling, variability, and failures

- **Measured observation:** No completed Phase 4 measurements were persisted.
- **Measured observation:** The background harness was interrupted and returned exit code 1; successful and failed subprocess counts and total subprocess duration are unavailable because the JSONL audit file was not finalized.
- **Verified conclusion:** Prompt-processing and generation throughput ranges, thread speedups, parallel efficiencies, coefficients of variation, and failed-case details cannot be reported without persisted measurements.

## Reproduction

Run the benchmark only when a new real run is explicitly authorized. If interrupted, resume the same durable run with `--resume`; add `--retry-failures` only when failed attempts should be repeated:

```console
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml --resume
.venv/bin/python benchmarks/run_llama_bench.py configs/prefill_decode_scaling.yaml --resume --retry-failures
```

Then pass the emitted CSV and JSONL paths to:

```console
.venv/bin/python benchmarks/analyze_prefill_decode.py CSV_PATH \
  --raw-jsonl JSONL_PATH \
  --output-dir benchmarks/results/prefill_decode_scaling \
  --report docs/results/prefill_decode_scaling.md \
  --model-filename qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Limitations

- **Verified conclusion:** This incomplete run provides no performance evidence and supports no optimization, scaling, compute-bound, or memory-bound claim.
- **Verified conclusion:** A completed run would still cover only one WSL2 host, one model and quantization, one llama.cpp commit, fixed batch/ubatch settings, and five repetitions. CPU affinity, frequency, temperature, power mode, and competing host load are not controlled or recorded.
