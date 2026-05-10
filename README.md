# speedygrad

AGPL-3.0 fork of [tinygrad](https://github.com/tinygrad/tinygrad). Measurement-driven kernel optimization replaces hand-tuned heuristics.

## vs PyTorch (Metal, fp32, TinyJit warm)

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

## What changed

**Abduction engine** replaces the heuristic and BEAM search. Two samples, one diff, the shape of the failure names the next experiment. `SEARCH=3` by default — 62% faster kernels than the heuristic, 32 trials per kernel, cached after first run.

**Universal reduction padder** in `_reduce` — pads to multiples of 32 so GROUPTOP fires on every reduction. Eliminates 9.2x performance cliffs on misaligned dimensions. 5 lines.

**Warp-reduce activation** — GROUPTOP=32 + `simd_sum`/`simd_max` replaces shared-memory reduction. 3.5x on max reduction kernels.

**Matmul TC padding** — pads M/K/N to tensor core tile multiples at the Tensor API. TC fires on misaligned matmul (253×251) via structural deduction.

**Cython schedule path** — `unified_rewrite`, `rewrite`, `toposort`, `dfs_match` compiled to C. 55% faster schedule (3.46s → 1.57s on ResNet50).

**Ported work** — [Sou-ly's](https://github.com/Sou-ly) toposort→dfs\_match optimization ([tinygrad #15491](https://github.com/tinygrad/tinygrad/pull/15491)), with attribution and push access.

## What was removed

- `hand_coded_optimizations` (190 lines) — replaced by abduction engine (90 lines)
- `beam_search` (99 lines) — strictly dominated by abduction
- `OptOps.PADTO` (16 lines) — replaced by Tensor-level matmul padding
- Structure test assertions (55 lines) — blocked optimization without proving correctness

Net: **-148 lines** vs upstream tinygrad's optimization code.

## How to use

```bash
# default: abduction search finds optimal kernel opts
python3 my_model.py

# fast mode (GROUPTOP=32 stub only, no search)
SEARCH=0 python3 my_model.py

# with Cython schedule speedup
python3 setup_cy.py build_ext --inplace
import monkeypatch  # add to your script
```

## Investigation

The full investigation trail is in [`HYPOTHESIS_GRAPH.md`](HYPOTHESIS_GRAPH.md) — every hypothesis, measurement, kill, and attribution.

## License

Additions: [AGPL-3.0](LICENSE). Original tinygrad code: MIT.
