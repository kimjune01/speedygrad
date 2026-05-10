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

## What changed

**Abduction engine** replaces the heuristic and BEAM search. Hypothesis-driven: try candidates, keep the best, follow the winner's category at the next depth. `SEARCH=3` by default. Usually finds faster kernels than the heuristic when search budget is acceptable (~20 trials/kernel avg, cached after first run).

**Reduction padding** in `_reduce` — pads single-axis MAX/ADD reductions with static misaligned dimensions to multiples of 32 so GROUPTOP fires. Eliminates performance cliffs on non-32-divisible reduction axes.

**Warp-reduce activation** — GROUPTOP=32 + `simd_sum`/`simd_max` replaces shared-memory reduction. 3.5x on max reduction kernels.

**Matmul TC padding** — pads matmul axes to multiples of 8 at the Tensor API so tensor cores can apply on misaligned shapes.

**Cython schedule path** — `unified_rewrite`, `rewrite`, `toposort`, `dfs_match` compiled to C. ~50% faster schedule (3.46s → 1.75s on ResNet50, 20 kernels).

**Ported work** — [Sou-ly's](https://github.com/Sou-ly) toposort→dfs\_match optimization ([tinygrad #15491](https://github.com/tinygrad/tinygrad/pull/15491)), with attribution and push access.

## What was removed

- `hand_coded_optimizations` — replaced by abduction engine
- `beam_search` — replaced by abduction engine
- `OptOps.PADTO` — replaced by Tensor-level padding
- Structure test assertions — blocked optimization without proving correctness

## How to use

```bash
# default: abduction search (SEARCH=3)
python3 my_model.py

# fast mode: GROUPTOP=32 stub only, no search
SEARCH=0 python3 my_model.py

# with Cython schedule speedup
python3 setup_cy.py build_ext --inplace
import monkeypatch  # add to your script
```

## Investigation

The full investigation trail is in [`HYPOTHESIS_GRAPH.md`](HYPOTHESIS_GRAPH.md) — every hypothesis, measurement, kill, and attribution.

## License

Additions: [AGPL-3.0](LICENSE). Original tinygrad code: MIT.
