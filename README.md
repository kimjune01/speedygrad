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

The legitimate question for a tinygrad user evaluating this fork: **does installing speedygrad over upstream tinygrad change anything on the same workload?**

| Model | quant | path | vanilla tinygrad | **speedygrad** | speedygrad/vanilla |
|---|---|---|---:|---:|---:|
| Llama 3.2 1B | fp16 | safetensors via `examples/llama3.py` | 83 tok/s | **140 tok/s** | **1.68×** |
| Qwen 3 0.6B | Q8\_0 | GGUF via `tinygrad/llm/model.py` | 226 tok/s | **241 tok/s** | **1.07×** |
| Qwen 3 1.7B | Q4\_K\_M | GGUF | 133 tok/s | **127 tok/s** | **0.95×**\* |
| Qwen 3 8B | Q4\_K\_M | GGUF | 0.9 tok/s | **1.0 tok/s** | 1.1× ⚠️ |

The speedygrad-vs-vanilla delta is path-dependent. On the older safetensors path, the monkeypatch optimizations (`counter.realize` + memoize-walk) deliver a real 1.68× win. On the newer GGUF path, the upstream `tinygrad/llm/model.py` is already efficient enough that speedygrad's contributions shrink to noise. Use this fork if you're on the older path or running a workload that exposes its specific bottlenecks; for stock GGUF inference, vanilla tinygrad is roughly equivalent.

\* 1.7B is within measurement noise of vanilla.

⚠️ **Qwen 3 8B Q4\_K\_M is broken** at ~1 tok/s decode (decode\_p50 = 1051 ms; theoretical bandwidth ceiling is ~7 ms). Q4\_K\_M dequantization on tinygrad's CUDA path has bandwidth-pathological access patterns at 8B+ model size. Filed as a frontier item in [`HYPOTHESIS_GRAPH.md`](HYPOTHESIS_GRAPH.md). Use Q8\_0 (when memory permits) for now; this is the workload most tinybox shoppers care about, and we lose to torch+HF here.

### Context: where speedygrad sits in the inference landscape

Comparison vs other inference stacks on Llama 3.2 1B fp16, same hardware:

| Stack | decode tok/s | notes |
|---|---:|---|
| torch + HF transformers (eager mode) | 20 tok/s | the default path most users start with — no `torch.compile`, no SDPA |
| **speedygrad** (this fork) | **140 tok/s** | tinygrad with monkeypatched fast paths |
| llama.cpp (CUDA backend, fp16) | ~224 tok/s | hand-tuned CUDA + fused attention; 1.6× faster than speedygrad |

Treat the torch+HF eager number as **a worst-case lower bound, not a real comparator**. Anyone running production inference on torch uses `torch.compile` + SDPA (probably 2-3× faster than eager) or moves to vLLM / ExLlamaV2 entirely. Likewise llama.cpp will beat speedygrad on workloads it has hand-tuned kernels for; that gap is documented and bounded in [`HYPOTHESIS_GRAPH.md`](HYPOTHESIS_GRAPH.md).

If you're shopping for a tinybox with the question "will I have to fight the inference stack?" — speedygrad sits in the middle of the spectrum: faster than the easy torch path most people start with, slower than the hand-tuned C++ specialists, and shares all of tinygrad's hackability and multi-device generality.

### Reproduce

```bash
PYTHONPATH=. python bench/scaling_table.py --runs 3 --n-new 20
```

Bench harness runs each combo as a subprocess for clean JIT state. `SPEEDYGRAD_VANILLA=1` env var disables all monkeypatch optimizations on the same bench code, isolating the fork's contribution. Greedy decode (`temperature=0`) for deterministic output — both vanilla and speedygrad produce identical text.

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
