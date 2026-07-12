# Q6_K × Q8_K hot-path laboratory

This directory isolates the pinned llama.cpp `ggml_vec_dot_q6_K_q8_K` AVX2 path.
It uses the exact 256-value, 210-byte `block_q6_K` weight layout and 292-byte
`block_q8_K` activation layout. `reference/` decodes each value clearly;
`baseline/` extracts the pinned AVX2 arithmetic; `optimized/` contains the candidate;
`tests/` compares all three; and `benchmark/` reports repeated median timings.

Build with `cmake --preset release && cmake --build --preset release`, then run
`build/release/kernel_q4_hotpath_test` and `build/release/kernel_q4_hotpath_bench`.
The AVX2 targets require AVX2, FMA, and F16C; llama.cpp retains its existing AVX and
generic fallback implementations.
