# Hypothesis Graph: speedygrad

AGPL-3.0 fork of [tinygrad](https://github.com/tinygrad/tinygrad). The thesis: tinygrad's heuristic-driven kernel optimization leaves measurable performance on the table, and a theory-guided system can close the gap.

## Root hypothesis

**H0: tinygrad's codegen heuristics are improvable by measurement-derived theories.**

The heuristic in `tinygrad/codegen/opt/heuristic.py` encodes 9 strategies in a priority cascade. 45% of the logic is measurement-derivable (parameters, thresholds). The rest is structural pattern matching (TC eligibility, kernel class detection). The heuristic is wrong often enough to matter: BEAM finds 44-59% improvements on shapes the heuristic handles poorly.

**Status:** CONFIRMED across Metal and CUDA.

---

## Confirmed findings

### Post-TC optimization targets wrong axes

The heuristic UPCASTs axis 0 (M dimension) after tensor core setup. For tall-skinny matmuls where M is small, this wastes register budget. UPCAST N + UNROLL K is universally better.

| Shape | Heuristic | Theory-derived | Speedup |
|---|---|---|---|
| 16x4096 x 4096x4096 | 3362us | 1912us | 1.76x |
| 8x2048 x 2048x2048 | 1281us | 528us | 2.43x |
| 256x256 x 256x256 | 313us | 154us | 2.03x |

CUDA fp16: 3.35x on the 16x4096 case. No regressions on square matmuls.

gfx12 (RDNA4) constraint: UNROLL(0,4) misaligns WMMA lane mapping. Safe path is UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,2) — both operand axes must be upcasted to preserve swizzle.

*PRs: #16104, #16107, #16109*

### MATVEC misclassification

`(a * b).sum()` triggers the MATVEC path because the subset check is non-strict — equal ranges pass. Fix: reject when `idx0.ranges == idx1.ranges`. The misclassification causes 4x regression on elementwise reductions.

*PRs: #16111, #16113, #16116, #16117*

### Theory transfer works

Semantic theories ("UPCAST N, UNROLL K, UPCAST M, LOCAL M — each by largest divisor that fits") transfer across 7/7 tested matmul shapes. Exact schedules fail on 3/6. Cache the theory, not the schedule.

| Shape | Heuristic | Adaptive theory | Ratio |
|---|---|---|---|
| 1024x1024 | 55068us | 9224us | 0.17x |
| 2048x2048 | 580501us | 89313us | 0.15x |
| 8x2048 x 2048x2048 | 3715us | 420us | 0.11x |

### Abduction loop beats heuristic

52-trial measurement loop (TC, UPCAST per axis, LOCAL per axis, GROUP, GROUPTOP, UNROLL, stride-based axis ordering) achieves 1.85x geometric mean over the heuristic on 5 workloads. Sole remaining gap: matvec (1.10x), where the heuristic's joint GROUP+LOCAL+UPCAST combo beats greedy search.

### PTX/CUDA renderer fallback

`is_dtype_supported` checks against the base renderer class, not the resolved renderer. PTXRenderer silently replaces CUDARenderer when NVRTC is missing, producing different (slower) kernels without warning.

*PR: #16108*

---

## Open edges

1. **Theory transfer to non-matmul classes** — reductions, elementwise, convolutions each need a seed measurement. Untested.
2. **Joint GROUP+LOCAL+UPCAST optimization** — the matvec gap requires evaluating combos, not greedy steps. ~20 lines for a 2-deep mini-beam.
3. **Amortized cost measurement** — 52 trials x compile+time. What's the wall-clock cost vs BEAM's 200+ trials?
4. **Theory transfer on CUDA** — Metal-confirmed only. CUDA verification started but incomplete.

---

## Adjacent investigations

Detailed hypothesis graphs live in [tinygrad-experiments](https://github.com/kimjune01/tinygrad-experiments):

| Investigation | Key finding |
|---|---|
| [beam/](https://github.com/kimjune01/tinygrad-experiments/tree/master/beam) | Abduction engine for BEAM — full H0-H5 tree, 52-trial loop, theory transfer |
| [matvec/](https://github.com/kimjune01/tinygrad-experiments/tree/master/matvec) | Full LLaMA inference gap decomposition |
| [linecount/](https://github.com/kimjune01/tinygrad-experiments/tree/master/linecount) | Line budget analysis, onnx proto dedup |
| [pareto-frontier/](https://github.com/kimjune01/tinygrad-experiments/tree/master/pareto-frontier) | graph_rewrite dispatch entropy |
| [realize/](https://github.com/kimjune01/tinygrad-experiments/tree/master/realize) | 11-op Metal benchmark, fusion gap analysis |
| [or/](https://github.com/kimjune01/tinygrad-experiments/tree/master/or) | Scheduling overhead, instruction scheduling, bank conflicts |

---

## Dependency graph

```
H0 (heuristics are improvable) — CONFIRMED
 ├─ post-TC axis selection — CONFIRMED (Metal + CUDA)
 │   └─ gfx12 WMMA constraint — CONFIRMED (RDNA4-specific safe path)
 ├─ MATVEC misclassification — CONFIRMED
 ├─ theory transfer — CONFIRMED (matmul class)
 │   └─ non-matmul transfer — OPEN
 ├─ abduction loop (52 trials, 1.85x geo mean) — CONFIRMED
 │   └─ joint optimization for matvec — OPEN
 └─ renderer fallback detection — CONFIRMED
```
