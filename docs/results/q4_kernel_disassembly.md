# Phase 8 Q6_K × Q8_K disassembly inspection

Both lab variants were compiled by GCC 15.2 with `-O3 -mavx2 -mfma -mf16c`.
The pinned build uses `-O3 -march=native`. The generated kernels contain AVX2
`vpand`, `vpor`, `vpsrlw`, `vpsllw`, `vpmaddubsw`, `vpmaddwd`, and `vpaddd`, then
`vcvtdq2ps` and `vfmadd213ps` for float accumulation. Loads are unaligned
`vmovdqu`. Horizontal reduction happens once after all blocks.

The split-accumulator candidate adds an independent integer reduction chain and a
final `vpaddd`. It did not reduce the unpack, shuffle, multiply-add, conversion, or
float-reduction work. On this compiler/CPU it was not a reliable improvement and
was commonly slower at 8 and 16 blocks. No AVX-512 instruction was observed.

Raw object dumps are intentionally not tracked. Reproduce with:

```sh
objdump -d -C -M intel build/release/libq4_hotpath_kernels.a
```
