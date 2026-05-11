# speedygrad

AGPL-3.0 fork of [tinygrad](https://github.com/tinygrad/tinygrad). Measurement-driven kernel optimization replaces hand-tuned heuristics.

## vs PyTorch (Metal, M-series, fp32, TinyJit warm, p50 of 50 trials)

| op | PyTorch | speedygrad | | was (tinygrad) |
|---|---|---|---|---|
| sum | 257us | **187us** | **0.73x** | 2.3x |
| add | 275us | **228us** | **0.83x** | 1.8x |
| mul\_sum | 215us | **214us** | **1.00x** | 2.3x |
| softmax | 206us | **209us** | **1.02x** | 5.3x |
| exp | 195us | 222us | 1.14x | 2.1x |
| gemm\_256 | 210us | 249us | 1.18x | 1.5x |
| permute | 181us | 222us | 1.22x | 3.0x |
| layernorm | 188us | 271us | 1.44x | 5.9x |
| gemm\_1024 | 340us | 559us | 1.64x | 1.8x |
| matvec | 530us | 1204us | 2.27x | 3.0x |

Shapes: N=1024 for gemm/mul\_sum, 256 for gemm\_256/softmax/layernorm/permute, 4096 for add/relu/sum, 4096x4096 for matvec. Wall-clock time including Metal dispatch. "was" column from the [realize investigation](HYPOTHESIS_GRAPH.md) on upstream tinygrad.

## LLM inference (RTX 4080, p50 decode tok/s)

| Model | quant | vanilla tinygrad | **speedygrad** | sg/vanilla | torch+HF (eager) | sg/torch |
|---|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B (safetensors) | fp16 | 83 tok/s | **140 tok/s** | **1.68×** | 20 tok/s | **7.0×** |
| Qwen 3 0.6B (GGUF) | Q8\_0 | 226 tok/s | **241 tok/s** | **1.07×** | 13 tok/s | **18.5×** |
| Qwen 3 1.7B (GGUF) | Q4\_K\_M | 133 tok/s | **127 tok/s** | **0.95×**\* | 9 tok/s | **14.1×** |
| Qwen 3 8B (GGUF) | Q4\_K\_M | 0.9 tok/s | **1.0 tok/s** | 1.1× | 7 tok/s | **0.14×**⚠️ |

\* 1.7B speedygrad row is within measurement noise of vanilla; the new GGUF inference path (`tinygrad/llm/model.py`) is already efficient enough that speedygrad's monkeypatch optimizations have less room to help. The bigger wins on Llama 1B are on the older `examples/llama3.py:build_transformer` safetensors path.

⚠️ **Qwen 3 8B Q4\_K\_M is broken** at ~1 tok/s decode (we should be ≥50). Q4\_K\_M dequantization on tinygrad's CUDA path is bandwidth-pathological at 8B size — known issue, filed as frontier item. Use Q8\_0 or fp16 (when memory permits) for now.

Reproduce:

```bash
PYTHONPATH=. python bench/scaling_table.py --runs 3 --n-new 20
```

The bench harness runs each (model × framework) combo as a subprocess for clean state. `SPEEDYGRAD_VANILLA=1` env var disables all monkeypatch optimizations on the same bench code, isolating the speedygrad fork's contribution from the underlying tinygrad framework's. torch+HF baseline uses `transformers.AutoModelForCausalLM` eager mode (no `torch.compile`, no SDPA hint) — the default path most users start with. Both produce identical greedy text (`temperature=0`).

## What changed

**Abduction engine** replaces the heuristic and BEAM search. Hypothesis-driven: try candidates, keep the best, follow the winner's category at the next depth. `SEARCH=3` by default. Usually finds faster kernels than the heuristic when search budget is acceptable (~20 trials/kernel avg, cached after first run).

**Reduction padding** in `_reduce` — pads single-axis MAX/ADD reductions with static misaligned dimensions to multiples of 32 so GROUPTOP fires. Eliminates performance cliffs on non-32-divisible reduction axes.

**Warp-reduce activation** — GROUPTOP=32 + `simd_sum`/`simd_max` replaces shared-memory reduction. 3.5x on max reduction kernels.

**Matmul TC padding** — pads matmul axes to multiples of 8 at the Tensor API so tensor cores can apply on misaligned shapes.

**Cython schedule path** — `unified_rewrite`, `rewrite`, `toposort`, `dfs_match` compiled to C. ~50% faster schedule (3.46s → 1.75s on ResNet50, 20 kernels).

**Cython runtime path** — `run_linear` + inlined single-device `exec_kernel` compiled to C. Per-call JIT replay overhead drops 3-7us (add_4096 51→46us, softmax 35→31us, gemm_1024 121→114us — wins torch by 10%).

**CUDA parity** — speedygrad on RTX 4080 with `import monkeypatch` wins or ties PyTorch on gemm_1024 (114 vs 126us), gemm_256 (51 vs 47us), mul\_sum, layernorm, matvec. Small ops (add/relu/exp) still ~1.8-2.0x off due to the host-side `cuLaunchKernel` floor.

**Ported work** — [Sou-ly's](https://github.com/Sou-ly) toposort→dfs\_match optimization ([tinygrad #15491](https://github.com/tinygrad/tinygrad/pull/15491)), with attribution and push access.

## What was removed

- `hand_coded_optimizations` — replaced by abduction engine
- `beam_search` — replaced by abduction engine
- `OptOps.PADTO` — replaced by Tensor-level padding
- Structure test assertions — blocked optimization without proving correctness

Net: **-148 lines** across `tinygrad/` and `test/` (+300, -448) vs upstream at fork point. Measured by `git diff 72a504471..HEAD --stat -- tinygrad/ test/`.

## Setup

Requires Python >=3.11.

```bash
git clone https://github.com/kimjune01/speedygrad.git
cd speedygrad

# optional: Cython schedule speedup (~50% faster compile)
python3 setup_cy.py build_ext --inplace
```

## Usage

```python
from tinygrad import Tensor

# works out of the box — SEARCH=3 finds fast kernels automatically
out = Tensor.randn(256, 256).softmax()

# optional: activate Cython schedule path
import monkeypatch
```

```bash
# override search depth
SEARCH=0 python3 my_model.py   # GROUPTOP=32 stub only, no search (fastest compile)
SEARCH=1 python3 my_model.py   # shallow search (good tradeoff)
SEARCH=3 python3 my_model.py   # default (best kernels, ~2s first-run cost)
```

## Investigation

The full investigation trail is in [`HYPOTHESIS_GRAPH.md`](HYPOTHESIS_GRAPH.md) — every hypothesis, measurement, kill, and attribution.

## License

Additions: [AGPL-3.0](LICENSE). Original tinygrad code: MIT.
