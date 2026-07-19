# Educational prefix-cache comparison

This is an architectural comparison, not a compatibility or performance
claim. Implementations evolve; the linked official sources are the authority
for their current behavior.

## This project

S5 is a deterministic metadata simulator with fixed-size, full-block-only
reuse. It chains a project-local stable 64-bit hash through parent blocks,
verifies exact key material after bucket selection, maintains request-table
reference counts, retains zero-reference cached blocks, and evicts them by
deterministic LRU with a physical-ID tie-break. Compatibility material includes
a model namespace and salt. It stores no tensor data and deliberately delays
same-plan publication until the next iteration.

## vLLM concepts

vLLM's official Automatic Prefix Caching design describes hash-based lookup
whose components include a parent hash, exact block tokens, and extra
compatibility hashes such as LoRA, multimodal, or cache-salt information. It
also documents full-block-only caching and cache operations integrated with a
block pool and LRU-style reuse/eviction. See the
[vLLM Automatic Prefix Caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
and [vLLM block-pool API](https://docs.vllm.ai/en/latest/api/vllm/v1/core/block_pool/).

Those shared concepts motivate S5's block granularity, parent dependency, and
compatibility namespace. S5's exact collision verification, one-canonical-key
policy, FNV encoding, state machine, and next-iteration visibility are its own
educational choices. They should not be inferred as descriptions of current
vLLM internals. This project does not implement PagedAttention or vLLM's GPU
memory/runtime paths.

## SGLang concepts

The SGLang paper presents RadixAttention as a runtime optimization for KV-cache
reuse, and the associated design uses radix-tree-oriented prefix organization
to support longest-prefix matching and cache-aware execution. See
[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104).

S5 shares only the high-level goals of longest reusable prefix discovery and
coordinating cache state with request execution. It uses a linear chain of
fixed token blocks and a hash index—not a radix tree—and does not implement
RadixAttention.

## Explicit non-parity statement

This project is not a reimplementation of vLLM or SGLang. It has no GPU
kernels, PagedAttention, RadixAttention, real tensor storage, distributed
cache, distributed serving, preemption, swapping, or production security
claim. The comparison is architectural and educational.

The broader [serving-engine comparison](engine_comparison.md) also places this
project beside llama.cpp and summarizes scheduler, batching, KV allocation,
APIs, distributed support, and hardware assumptions without a parity claim.
