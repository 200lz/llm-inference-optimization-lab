# Educational serving-engine comparison

This is a conceptual survey, not a compatibility, performance, or feature-parity claim. External engines evolve; their official documentation is authoritative.

| Topic | This project | llama.cpp | vLLM | SGLang |
| --- | --- | --- | --- | --- |
| Target environment | Educational metadata simulation | Portable local model inference | Accelerator-oriented serving engine | LLM program/runtime serving system |
| Scheduler | Single FCFS; DecodeFirst; FcfsMixed | Runtime/server request handling outside this project | Production request scheduling concepts | Runtime scheduling and cache-aware execution concepts |
| Batching | Deterministic simulated iterations | Real model batching capabilities; not integrated here | Continuous batching concepts | Continuous/cache-aware batching concepts |
| KV allocation | Fixed block IDs and token occupancy only | Real backend KV storage | Block-pool/PagedAttention concepts | Runtime-managed KV with RadixAttention concepts |
| Prefix reuse | Exact verified, full-block hash chain | Not measured or integrated here | Automatic Prefix Caching concepts | Radix-tree-oriented prefix reuse concepts |
| Eviction | Deterministic metadata LRU | Not characterized here | Engine-managed block reuse/eviction | Runtime cache eviction |
| Preemption | Not implemented | Not modeled here | Production engine capability; details version-dependent | Production runtime capability; details version-dependent |
| Distributed | None | Outside this project | Supported configurations exist | Supported configurations exist |
| APIs | CLI files only | CLI/server APIs | Serving APIs | Serving/program APIs |
| Hardware assumptions | None; coefficients are synthetic | Actual configured backends | Accelerator/runtime dependent | Accelerator/runtime dependent |

The project is conceptually inspired by vLLM's documented full-block, parent-dependent Automatic Prefix Caching and block-pool ideas, and by SGLang's published RadixAttention goal of longest-prefix KV reuse. Its exact-collision verification, FNV encoding, two-phase transaction, and fixed-block hash chain are educational project choices. It does not implement PagedAttention, RadixAttention, a radix tree, real tensor storage, production APIs, or distributed execution.

See the official [vLLM prefix-caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/), [vLLM block-pool API](https://docs.vllm.ai/en/latest/api/vllm/v1/core/block_pool/), and [SGLang paper](https://arxiv.org/abs/2312.07104). No current implementation detail beyond those already verified concepts is asserted.
