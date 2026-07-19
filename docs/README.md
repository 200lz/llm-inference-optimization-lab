# Documentation index

Start with the [final technical report](final_report.md) for the complete narrative or the repository [README](../README.md) for a one-minute overview.

## Setup and methodology

- [Pinned dependencies](dependencies.md)
- [Qwen GGUF model setup](model_setup.md)
- [Reproducible llama.cpp CPU Release build](llama_cpp_build.md)
- [Benchmark methodology](benchmarking.md)
- [CPU profiling environment](profiling_environment.md)

## Serving simulator design

- [Architecture](serving/architecture.md)
- [Development plan](serving/development_plan.md)
- [Metric definitions and evidence semantics](serving/metrics.md)
- [Phase S2 simulator](serving/simulator.md)
- [Phase S2 scheduler and FCFS policy](serving/scheduler.md)
- [Phase S3 deterministic continuous batching](serving/continuous_batching.md)

## Results by phase

- Phase 3: [Q4_K_M CPU baseline](results/cpu_baseline_q4.md)
- Phase 4: [prefill versus decode scaling](results/prefill_decode_scaling.md)
- Phase 5: [F16 / Q8_0 / Q4_K_M comparison](results/quantization_comparison.md)
- Phase 6: [KV-cache and context scaling](results/kv_cache_context_scaling.md)
- Phase 7: [CPU profiling and bottleneck attribution](results/cpu_profiling.md)
- Phase 8A: [rejected Q6_K/Q8_K accumulator candidate](results/q4_hotpath_optimization.md) and [disassembly inspection](results/q4_kernel_disassembly.md)
- Phase 8B: [Q8_0 target selection](results/phase8b_target_selection.md)
- Phase 8C: [extracted Q8_0 RN=2 optimization](results/phase8c_q8_rn2_scale.md) and [integration provenance](results/phase8c_integration_provenance.md)
- Phase 8D: [binary provenance](results/phase8d_binary_provenance.md) and [paired end-to-end A/B](results/phase8d_q8_end_to_end_ab.md)

The Phase 8 patch is exported at [`patches/phase8c-q8-rn2-scale-preparation.patch`](../patches/phase8c-q8-rn2-scale-preparation.patch). It is not applied to the pinned submodule.
