# Benchmark Results

All measurements from the [tinygrad-experiments](https://github.com/kimjune01/tinygrad-experiments) investigation. Baselines are upstream tinygrad heuristics; improvements are from speedygrad's merged PRs.

---

## Post-TC heuristic fix

UPCAST N + UNROLL K instead of the heuristic's axis-0 bias.

### Metal (M4 Max, fp32)

| Shape | Heuristic | Fixed | Speedup |
|---|---|---|---|
| 16x4096 x 4096x4096 | 3362us | 1912us | 1.76x |
| 8x2048 x 2048x2048 | 1281us | 528us | 2.43x |
| 256x256 x 256x256 | 313us | 154us | 2.03x |

*PRs: #16104, #16107, #16109*

### CUDA (RTX 4080, fp16)

| Shape | Master us | Master GFLOPS | Fixed us | Fixed GFLOPS | Speedup |
|---|---|---|---|---|---|
| 16x4096 x 4096x4096 | 223 | 2405 | 66.6 | 8066 | 3.35x |
| 256x256 x 256x256 | 42.9 | 781 | 45.1 | 745 | ~neutral |
| 8x2048 x 2048x2048 | 122.6 | 547 | 117.5 | 571 | ~neutral |

CUDA fp32 is trivially neutral — TC requires fp16/bf16, so the post-TC code path is never reached.

### gfx12 (RDNA4)

UNROLL(0,4) misaligns WMMA lane mapping on gfx1201. Safe path: UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,2). Validated via CI on both amd and amdllvm backends.

---

## Abduction loop vs heuristic

52-trial measurement loop: TC, UPCAST per axis, LOCAL per axis, GROUP, GROUPTOP, UNROLL, stride-based axis ordering.

### Metal (M4 Max)

| Workload | Heuristic | Abduction (52 trials) | Ratio |
|---|---|---|---|
| gemm_1024 | 307us | 153us | 0.50x |
| mul_sum | 343us | 223us | 0.65x |
| softmax | 15us | 4us | 0.24x |
| matvec | 103us | 112us | 1.10x |
| layernorm | 33us | 19us | 0.56x |

Geometric mean: 1.85x faster than heuristic. Sole gap: matvec (1.10x), where the heuristic's joint GROUP+LOCAL+UPCAST combo beats greedy search.

---

## Theory transfer

Semantic theory derived from ONE measurement (gemm_1024) applied to 7 matmul shapes with zero additional measurements.

### Metal (M4 Max)

| Shape | Heuristic | Adaptive theory | Ratio |
|---|---|---|---|
| 1024x1024 | 55068us | 9224us | 0.17x |
| 256x256 | 749us | 146us | 0.19x |
| 2048x2048 | 580501us | 89313us | 0.15x |
| 16x4096 x 4096x4096 | 21428us | 4230us | 0.20x |
| 8x2048 x 2048x2048 | 3715us | 420us | 0.11x |
| 4096x16 x 16x4096 | 16671us | 2955us | 0.18x |
| 512x2048 x 2048x256 | 15435us | 2229us | 0.14x |

---

## Stride-aware matvec

MV_ROWS_PER_THREAD 4 → 16. Verified no regressions on LLaMA, GPT-2, BERT, Whisper, Mixtral.

### Metal

| Layout | Before | After | Gain |
|---|---|---|---|
| contiguous (K,N) | 53 GB/s | 86 GB/s | 62% |
| transposed (N,K).T | 43 GB/s | 87 GB/s | 105% |

*PR: #16072*

---

## Quantized GGUF inference (contiguous + prune)

LLaMA 1B Q6_K. Contiguous weights break lazy dequant fusion; prune makes dequant one-time during JIT capture.

| Backend | Before | After | Speedup |
|---|---|---|---|
| Metal (M5 Max) | 10.5 tok/s | 147 tok/s | 14.0x |
| NV (RTX 5000 Ada) | 85.8 tok/s | 85.8 tok/s | 8.2x |

Bit-exact output on both backends.

*PR: #16094*

---

## BEAM baseline analysis

What BEAM actually does, measured. Platform: Metal, M4 Max. BEAM=2, IGNORE_BEAM_CACHE=1, CNT=4.

| Workload | Heuristic | BEAM | b/h | Rounds | Timed | Search time |
|---|---|---|---|---|---|---|
| gemm_1024 | 1.88ms | 1.71ms | 0.91x | 13 | 165 | 3.3s |
| gemm_256 | 1.23ms | 0.87ms | 0.70x | 12 | 132 | 2.2s |
| add_4096 | 1.30ms | 1.34ms | 1.03x | 13 | 105 | 2.5s |
| mul_sum | 2.36ms | 1.48ms | 0.63x | 16 | 165 | 3.1s |
| relu_4096 | 1.40ms | 0.99ms | 0.71x | 9 | 100 | 1.8s |
| exp_2048 | 1.06ms | 1.13ms | 1.07x | 9 | 97 | 2.0s |
| sum_4096 | 1.84ms | 1.46ms | 0.79x | 15 | 229 | 3.2s |
| permute | 1.02ms | 1.00ms | 0.97x | 9 | 148 | 1.9s |
| softmax | 1.41ms | 1.49ms | 1.05x | 15 | 235 | 3.3s |
| layernorm | 2.07ms | 1.61ms | 0.78x | 14 | 255 | 3.5s |
| matvec | 1.87ms | 1.41ms | 0.76x | 15 | 146 | 4.6s |

BEAM regresses on 3/11 workloads (add, exp, softmax) because it has no baseline candidate and uses proxy measurements at 1/16th scale.

---

## Warp-reduce for GROUPTOP

Replace scalar shared-memory reduction loop with simd_sum. 2.1-4.2x faster on the reduction step. HIP disabled pending AMD hardware testing.

*PR: #16070*

---

## UPat matcher: skip redundant root op check

| Workload | Speedup |
|---|---|
| softmax | 15% |
| conv | 14% |
| 4x conv | 10% |
| transformer | 9% |

Cold schedule time improvement. The compiled matcher already runs inside `pdict[uop.op]` dispatch.

*PR: #16096*

---

## Cython floor-lowering chain (2026-05-09)

ResNet50 schedule + rewrite (20 kernels). Metal, Python 3.14. `import monkeypatch` for Cython.

| Stage | Time | LOC | `rewrite` ms | vs Python |
|---|---|---|---|---|
| Pure Python | 3.457s | 0 | — | baseline |
| + Cython unified_rewrite | 2.282s | 95 .pyx | 341 | -34% |
| + bitmask early-reject | 2.216s | +8 .py | 307 | -36% |
| + int(op) descriptor fix | 2.145s | +3 .py | 283 | -38% |
| + list-indexed dispatch | 2.042s | +2 .py | 229 | -41% |
| + skip mega guard | 1.979s | +1 .py | 202 | -43% |
| + Cython rewrite + inline pm | 1.817s | +50 .pyx | (in C) | -47% |
| + Sou-ly dfs_match (#15491) | 1.752s | +30 .py | (in C) | -49% |

Pattern match frequency: 6.7% hit rate, 80% median winner concentration per op. 93.3% of attempts are wasted. Huffman ordering (under Cython native branches) is the next layer.

Attribution: Sou-ly (github.com/Sou-ly) for dfs_match, recursive_property fast path, op_in_backward_slice_with_self, fix_store_after_hazard DFS. From tinygrad PR #15491, closed as "AI SLOP."

---

## Warp-reduce activation (2026-05-09)

GROUPTOP 16→32 in heuristic.py + relaxed `fix_group_for_reduce_warp` to accept REDUCEs with additional reduce ranges.

### Metal (softmax 1024x1024, fp32)

| Kernel | Shared mem (GROUPTOP=16) | Warp-reduce (GROUPTOP=32) | Speedup |
|---|---|---|---|
| max reduction | 94.13us | 26.88us | **3.50x** |
| sum reduction | 25.75us | 25.25us | ~neutral |

Max 3.5x faster: eliminates shared memory writes, `threadgroup_barrier`, serialized 16-iteration final loop (thread 0 active, 15 idle). Sum neutral — `exp2` computation dominates.

---

## CPL scheduling — KILLED on Metal (2026-05-09)

10-line CPL priority in linearizer. Two findings:

1. **Matvec:** identical kernel source (topological constraints too tight after stride-aware fix)
2. **GEMM TC:** different kernel source (interleaved loads/WMMAs vs batched), same GPU time (~870-900us)

Kill condition: Metal's shader compiler normalizes instruction order. Source-level scheduling is advisory.

---

## Windows CUDA setup notes

From the RTX 5000 Ada bench session:

1. `DEV=NV` is hard-asserted off on Windows. Use `DEV=CUDA`.
2. tinygrad's loader searches for `cuda.dll` but the Windows driver is `nvcuda.dll`. Override with `CUDA_PATH=C:\Windows\System32\nvcuda.dll`.
3. Default Python thread stack (~1MB on Windows) blows `pretty_print` recursion. Workaround: run bench on a worker thread with 64MB stack.
