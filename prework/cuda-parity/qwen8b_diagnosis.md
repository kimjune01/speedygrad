# Qwen 3 8B Q4_K_M decode regression — diagnosis

**Status:** characterized, not fixed. Filed for future kernel work.

## Symptom

| Model | quant | decode_p50 | tok/s | sg/torch |
|---|---|---|---|---|
| Qwen 3 1.7B | Q4_K_M | 7.9 ms | 127 | 14.1× faster |
| **Qwen 3 8B** | **Q4_K_M** | **1051 ms** | **0.95** | **7× SLOWER** |

Linear scaling from 1.7B → 8B (4.7× model size) would predict ~37 ms / 27 tok/s.
Actual: 1051 ms / 0.95 tok/s. **28× worse than linear scaling would predict.**

## Smoke test (`probe_qwen8b_8s.py`)

- All 399 model-parameter buffers on CUDA (no CPU fallback).
- 8.19B params, ~4.1 GB Q4_K_M weights.
- Loaded in 7.1 s; VRAM headroom OK.
- TTFT (prefill + first decode + JIT capture): 40.4 s. Steady-state decode 900-1700 ms/token.
- `_apply_map_to_tensors` cost: 1638 us / call × ~1 call per token = **0.0% of wall**. Host overhead is NOT the bottleneck. The slowness is GPU-side.

## nsys trace (`prework/cuda-parity/nsys_qwen8b.nsys-rep`)

13 forwards (3 burn + 10 measured). Per-forward GPU time (mean × instances/forward):

| Kernel | mean (ms) | calls/fwd | per-fwd (ms) | role |
|---|---|---|---|---|
| r_4096_4_8_384 | 3.22 | 36 | **116** | matmul over hidden_dim, 1 per layer |
| r_12288_8_4_8_16 | 2.73 | 36 | **98** | FFN matmul (intermediate=12288), 1 per layer |
| r_3072_256_4_16 | 2.56 | 36 | **92** | matmul, 1 per layer |
| r_toks_4096_16_8_96 | 32.81 | ~3 | **91** | per-token op, very slow per call |
| r_151936_256_4_4 | 75.37 | ~1 | **70** | output projection (vocab=151936), per token |
| r_toks_12288_16_256n1 | 26.01 | ~3 | **72** | per-token FFN op |
| r_toks_12288_16_256 | 25.15 | ~3 | **70** | per-token FFN op |
| E_24576_4_2_2_16_8 | 18.30 | ~6 | **101** | element-wise (!), should be sub-ms |
| E_12288_2_2_2_2_16_16 | 29.29 | ~1.5 | **41** | element-wise |
| Other | — | — | ~110 | tail |
| **Total per forward** | | | **~860 ms** | matches measured 900-1700 ms |

CUDA API summary:
- cuCtxSynchronize: 4.4 s total, 20 calls, **avg 221 ms** — host waiting on GPU work
- cuLaunchKernel: 318 ms / 2553 calls — kernel launch overhead reasonable
- cuGraphLaunch: 222 ms / 55 calls — graph replay reasonable
- cuMemAlloc_v2: 2.8 s / 561 calls (one outlier at 841 ms — probably JIT capture)

## Diagnosis

GPU is genuinely busy (cuCtxSync waiting 221 ms on average — real work, not Python idle). The kernels themselves are slow.

**Comparison to Llama 1B's largest kernel** (which works at 140 tok/s):
- Llama 1B largest: r_512_16_512_512_4_4 at 134us median × 16 layers = **2.1 ms/forward**
- Qwen 8B largest matmul: r_4096_4_8_384 at 3.22ms × 36 layers = **116 ms/forward**
- Ratio: **55× longer** per kernel-type, vs only ~3× more compute (FFN scaling: 4096×12288 vs 2048×8192 = 3×)
- **Per-flop efficiency is ~18× worse on 8B than on 1B for similar matmul work.**

This is a **codegen inefficiency**, not a fundamental compute or bandwidth limit. Tinygrad's BEAM/SEARCH is finding kernel configurations (tile sizes, GROUPTOP, UPCAST counts) that work well for 1B-shape matmuls but fall over at 8B shapes. The Q4_K_M dequant adds compute per byte read, which should be GOOD for arithmetic intensity — but the resulting kernels may have bad cache reuse or wrong threadblock sizing.

## Hypotheses for future work

1. **Tile sizes not optimal for 8B shapes.** Re-run SEARCH with deeper budget (`SEARCH=5+`) on 8B-specific kernels. May find faster configs.
2. **L2 cache thrashing across consecutive layers.** 36 layers × ~5 kernels = 180+ kernels back-to-back, each reading 50-100MB of Q4 weights. L2 (40MB on 4080) cycles through every kernel. Could be helped by manually flushing or by reordering execution.
3. **Q4_K_M dequant not fused with matmul.** Each kernel may be dequantizing to fp16 intermediate then matmul'ing — 2× the memory traffic vs fused dequant-matmul. Llama.cpp has hand-fused Q4_K_M kernels for exactly this reason. Worth checking if tinygrad's PTX renderer is producing fused or split kernels.
4. **Element-wise kernels at 18-29 ms each are pathological.** E_24576_4_2_2_16_8 at 18 ms mean is an element-wise op that should take sub-ms. Likely a broken codegen path that's launching too many small blocks or doing redundant memory passes.

## Why this matters for v1.0

8B is the dominant local-inference model size for tinybox-class hardware. Losing 7× to torch+HF on this size undermines the "speedygrad as a serious inference framework" claim. The fix isn't a quick monkeypatch — it's actual kernel codegen work, possibly including Q4_K_M dequant hand-tuning. Estimated effort: weeks, not days.

## Reproduce

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python prework\cuda-parity\probe_qwen8b_8s.py 2>&1 | Select-Object -Last 35

$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
& $nsys profile --trace=cuda --cuda-graph-trace=node --output prework/cuda-parity/nsys_qwen8b --force-overwrite=true .venv\Scripts\python prework\cuda-parity\nsys_qwen8b.py
& $nsys stats --report cuda_gpu_kern_sum --format csv prework/cuda-parity/nsys_qwen8b.nsys-rep | Out-File -Encoding utf8 prework/cuda-parity/qwen8b_kern.csv
```

## Files

- `prework/cuda-parity/probe_qwen8b_8s.py` — smoke test confirming GPU-side bottleneck
- `prework/cuda-parity/nsys_qwen8b.py` — minimal nsys runner
- `prework/cuda-parity/qwen8b_diagnosis.md` — this document
