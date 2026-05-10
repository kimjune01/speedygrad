# Hypothesis Graph: speedygrad

AGPL-3.0 fork of [tinygrad](https://github.com/tinygrad/tinygrad). The thesis: tinygrad's heuristic-driven kernel optimization leaves measurable performance on the table, and a theory-guided system can close the gap.

Consolidated from six investigations in [tinygrad-experiments](https://github.com/kimjune01/tinygrad-experiments): beam, realize, matvec, pareto-frontier, or, linecount.

---

## I. Kernel quality (beam investigation)

### H0: tinygrad's codegen heuristics are improvable — CONFIRMED

The heuristic in `codegen/opt/heuristic.py` encodes 9 strategies in a priority cascade. 45% of the logic is measurement-derivable (parameters, thresholds). The rest is structural pattern matching (TC eligibility, kernel class detection). BEAM finds 44-59% improvements on shapes the heuristic handles poorly.

### Post-TC optimization targets wrong axes — CONFIRMED

The heuristic UPCASTs axis 0 (M dimension) after tensor core setup. For tall-skinny matmuls where M is small, this wastes register budget. UPCAST N + UNROLL K is universally better.

| Shape | Heuristic | Fixed | Speedup |
|---|---|---|---|
| 16x4096 x 4096x4096 | 3362us | 1912us | 1.76x |
| 8x2048 x 2048x2048 | 1281us | 528us | 2.43x |
| 256x256 x 256x256 | 313us | 154us | 2.03x |

CUDA fp16: 3.35x on 16x4096. No regressions on square matmuls.

gfx12 (RDNA4) constraint: UNROLL(0,4) misaligns WMMA lane mapping due to non-sequential operand-to-lane mapping (skips r2 in upcast dims). Safe path: UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,2) — both operand axes must be upcasted to preserve swizzle.

Root cause traced to `tc.py` — RDNA4 opts ordering places `l1` after axis-1 upcasts, and permutation `(4,5,6,7,8,9,11,10,0,1,2,3)` differs fundamentally from RDNA3's `(4,5,6,7,0,9,10,11,1,2,3,8)`.

*PRs: #16104, #16107, #16109*

### MATVEC misclassification — CONFIRMED

`(a * b).sum()` triggers the MATVEC path because the subset check `all(r in idx1.ranges for r in idx0.ranges)` is trivially true when ranges are equal. A true matvec has asymmetric access: vector ranges are a strict subset of matrix ranges. Fix: reject when `idx0.ranges == idx1.ranges`. The misclassification causes 4x regression on elementwise reductions.

*PRs: #16111, #16113, #16116, #16117*

### Theory transfer — CONFIRMED (matmul class)

Semantic theories ("UPCAST N, UNROLL K, UPCAST M, LOCAL M — each by largest divisor that fits") transfer across 7/7 tested matmul shapes. Exact schedules fail on 3/6. Cache the theory, not the schedule.

| Shape | Heuristic | Adaptive theory | Ratio |
|---|---|---|---|
| 1024x1024 | 55068us | 9224us | 0.17x |
| 2048x2048 | 580501us | 89313us | 0.15x |
| 8x2048 x 2048x2048 | 3715us | 420us | 0.11x |
| 16x4096 x 4096x4096 | 21428us | 4230us | 0.20x |
| 4096x16 x 16x4096 | 16671us | 2955us | 0.18x |
| 512x2048 x 2048x256 | 15435us | 2229us | 0.14x |

One seed measurement (gemm_1024) transfers to all shapes with zero additional measurements.

### Abduction loop — CONFIRMED (1.85x geo mean)

52-trial measurement loop (TC, UPCAST per axis, LOCAL per axis, GROUP, GROUPTOP, UNROLL, stride-based axis ordering) vs heuristic:

| Workload | Heuristic | Abduction (52 trials) | Ratio |
|---|---|---|---|
| gemm_1024 | 307us | 153us | 0.50x |
| mul_sum | 343us | 223us | 0.65x |
| softmax | 15us | 4us | 0.24x |
| matvec | 103us | 112us | 1.10x |
| layernorm | 33us | 19us | 0.56x |

Sole gap: matvec (1.10x), where the heuristic's joint GROUP+LOCAL+UPCAST combo beats greedy search.

### BEAM baseline analysis

BEAM is grid search over a hyperparameter space with beam pruning. 193 actions, 92-97% yield, no pruning. Six structural problems:

1. **No baseline candidate** — starting schedule assigned time=infinity, never measured. BEAM can't fall back to the heuristic; regresses on 3/11 workloads (add 1.03x, exp 1.07x, softmax 1.05x).
2. **Proxy measurement** — `allow_test_size` shrinks to max 65536 threads, scales linearly. Different bottleneck at 1/16th scale.
3. **No pruning** — no structural reasoning eliminates candidates before compile+time.
4. **Plateau exit** — exits when improvement < 10ns. Can't distinguish local from global minimum.
5. **No kernel triage** — same search budget on 7us copy kernels as on 1ms compute kernels.
6. **Early stop discards diagnostics** — bad candidates' failure modes are evidence, but timing stops after one sample.

### H5: Hybrid architecture — SUPERSEDED by abduction engine

Originally proposed: keep structural priors, replace parameters with measurement. The actual outcome went further: the entire heuristic was deleted (190 lines). The abduction engine (90 lines) replaces both structural priors and parameters with measurement-driven search. The GROUPTOP=32 stub (4 lines) is the only surviving structural prior.

The abduction engine should:
3. Fix the structural priors when measurement proves them wrong
4. Cache the theory for amortization

---

## II. Performance gap decomposition (realize investigation)

### tinygrad vs PyTorch — 1.2x to 5.9x gap — CONFIRMED

11 ops on Metal (fp32, post-JIT, trimmed mean). Ratios: 1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 3.0, 5.0, 5.0, 5.3, 5.9.

### Single-kernel ops: realize overhead, not codegen — CONFIRMED

GPU kernel times are competitive with or faster than torch for gemm_256, exp_2048, permute, matvec. The wall-time gap is Python realize overhead (~600-1000us per call vs torch's C++ dispatch at ~10-25us). TinyJit closes most of the gap.

### Compound ops: three complementary layers — CONFIRMED

1. **Reduction kernels are scheduling-inert.** Max-reduction: 4 instructions, 0 independent. Sum-exp: 6 instructions, 0 independent. GEMM: 58 instructions, 24 independent. Zero ILP in reductions — nothing to reorder.
2. **GPU compute is 2-14% of wall time.** Softmax GPU = 48us out of 2800us wall. Layernorm GPU = 607us out of 4476us. 86-98% is host-side.
3. **PyTorch dispatches vendor primitives.** Softmax → `mpsGraph softMaxWithTensor` (Apple closed-source). LayerNorm → hand-written `LayerNorm.metal` with simd_sum, float4, rsqrt.

### Linearizer instruction ordering — CONFIRMED (GEMM +2x)

Changing LOAD priority from -1 to 0 in `linearizer.py:29` (interleave loads with compute instead of clustering loads first):

| Workload | Gap (before) | Gap (after) | Improvement |
|---|---|---|---|
| gemm_1024 | 1.81x | 0.89x | beats torch |
| permute | 2.96x | 1.39x | 2.1x faster |
| add_4096 | 1.76x | 1.26x | 1.4x faster |

9/11 workloads improved. Median gap: 2.13x → 1.89x. One-character change.

### Realize overhead is known, accepted — CONFIRMED

Issue #7698: "Tensor(numpy).realize() takes 0.85ms to schedule." Schedule cache merged Dec 2025. TinyJit bypasses Python on replay. A 29% scheduling speedup PR (#15491) was rejected as "AI SLOP." Pure Python is an explicit design value. Vendor dispatch is philosophically rejected (geohot, issue #429).

### Fused codegen gap: algorithm selection, not instruction quality — CONFIRMED

PCONTIG=99 forces softmax/layernorm into 1 kernel. Result: worse (softmax 0.323ms → 0.578ms). PCONTIG concatenates three serial-chain kernels — same algorithm, worse occupancy. Missing pieces:

1. `simd_sum` warp-level reduction — no op, no renderer path
2. Algorithm-selection pattern matcher for mean-variance-normalize

---

## III. LLM inference gap (matvec investigation)

### Novel graph shapes dominate — KILLED

Premise was false. tinygrad's symbolic JIT (`UOp.variable("start_pos", ...)`) covers all steady-state inference. Exactly 1 JIT call, 1 realize per token. No novel graph shapes during decode.

### Decode gap is matvec kernel quality — CONFIRMED

The generated Metal kernel for 1x4096 x 4096x4096 walks the weight matrix with stride 32768 bytes (`Ridx0<<15`) in the inner loop. Unit-stride output axis (+4, +8, +12) is outside the loop. 8192x worse than optimal for float32.

Same structural issue confirmed on CPU (PR #14630: 2.76x slower than torch, custom kernel closed to parity, rejected by geohot asking "why can't BEAM find it?").

BEAM cannot fix this — search space doesn't include loop reordering. The fix requires changing how the scheduler orders reduction loops in matvec-shaped matmuls.

### Prefill gap is arithmetic intensity ceiling — CONFIRMED

chunk_size=32 caps intensity at 32 FLOP/byte (float16), near the memory/compute ridge point. Batch prefill at T=1000 reaches 1000 FLOP/byte — fully compute-bound. Chunking caps performance structurally.

### Quantized dequant codegen is the dominant model-level bottleneck — CONFIRMED

| Config | llama.cpp | tinygrad | Ratio |
|---|---|---|---|
| 1B Q6_K decode | 341 tok/s | 10.5 tok/s | 32.5x |
| 3B F16 decode | 74.5 tok/s | 26.5 tok/s | 2.8x |

The F16 gap (2.8x, 173 GB/s) matches matvec prediction. The Q6_K gap (32x, 11 GB/s) is from lazy dequant chains — 130-UOp dequant graph re-executed every forward pass, producing 328-line Metal kernels with 271 scalar byte loads at 3 GB/s.

### Contiguous + prune fix — CONFIRMED (14x Metal, 8.2x NV)

`.contiguous()` on weights breaks lazy dequant fusion. `prune=True` on rollout JIT makes dequant one-time during capture. 10.5 → 147 tok/s (Metal M5 Max), 10.4 → 85.8 tok/s (RTX 5000 Ada). Bit-exact output on both backends.

Prefill prune misclassifies cache kernels — fix: rollout-only prune. Confirmed safe on MoE (OLMoE) and SSM (Qwen3.5). Peak memory matches REALIZE=1 (benign). Works across quant formats (Q4_K_M: 3.8x, Q6_K: 13.1x).

Remaining gap after fix: 2.6x (130 vs 341 tok/s). Both achieve ~325-340 GB/s but tinygrad reads 3.0 GB F16 while llama.cpp reads 1.0 GB Q6K with hand-written SIMD dequant.

*PR: #16094*

### Stride-aware matvec — CONFIRMED

MV_ROWS_PER_THREAD 4 → 16. No regressions on LLaMA, GPT-2, BERT, Whisper, Mixtral.

| Layout | Before | After |
|---|---|---|
| contiguous (K,N) | 53 GB/s | 86 GB/s |
| transposed (N,K).T | 43 GB/s | 87 GB/s |

*PR: #16072*

### Warp-reduce for GROUPTOP — CONFIRMED, ACTIVATED

Replace scalar shared-memory reduction with simd_sum/simd_max. Originally shipped with GROUPTOP=32 constraint but the heuristic hardcoded GROUPTOP=16 — warp-reduce never fired.

**Fix:** GROUPTOP 16→32 in `heuristic.py` + relax `fix_group_for_reduce_warp` to accept REDUCEs with additional reduce ranges (the per-thread partial sum loop).

| Kernel | Shared mem (GROUPTOP=16) | Warp-reduce (GROUPTOP=32) | Speedup |
|---|---|---|---|
| softmax max (1024x1024) | 94.13us | 26.88us | **3.50x** |
| softmax sum (1024x1024) | 25.75us | 25.25us | ~neutral |

Max reduction 3.5x faster because warp-reduce eliminates: shared memory writes, `threadgroup_barrier`, and the serialized 16-iteration final loop (thread 0 works, 15 idle). Sum is neutral because `exp2` computation dominates over the reduction step.

Correctness verified: 9 shapes (32 to 1024x1024), sum/max/softmax all pass. Linearizer tests: 24 passed.

*PR: #16070*

---

## IV. Pattern matcher performance (pareto-frontier investigation)

### Dispatch table entropy: 1.91 bits — CONFIRMED

8,649 slots (93x93), 59% empty, 26% single-pattern, 8% two-pattern, 7% three+. Only 29 distinct row signatures.

### Cost is fractal — CONFIRMED

Successful match callbacks contain nested graph_rewrite calls. The "80% in successful matches" framing was wrong — 23us per successful match includes thousands of nested pattern dispatches.

### Redundant root op check (H12) — CONFIRMED, first speedup

98% of compiled matchers re-check `uop.op` despite already dispatching via `pdict[uop.op]`. Each check costs 22-29ns. Two-line fix: skip op check when generating root clause.

| Workload | Before | After |
|---|---|---|
| 4x conv (cold) | 65.4ms | 63.3ms (-3.2%) |
| transformer (cold) | 52.2ms | 50.1ms (-4.0%) |

*PR: #16096*

### Mega-matcher (H18) — CONFIRMED (-18% micro, 0% e2e)

Merging all 20 ADD patterns into one function eliminates 19 frame creations (~100-150ns each), shares prefix checks. -18% on isolated rewrite micro-bench (consistent across Python 3.14 and 3.16).

But no end-to-end signal: 66% of rewrite calls have 0-1 patterns (mega doesn't apply), 24% have 5+ patterns (mega helps but only ~200ns/call savings). Expected improvement: 24% x 15% x 68% = 2.4% of total, ~0.5ms on 23ms — below measurement variance.

### Cython transpile of unified_rewrite (H27) — CONFIRMED, SHIPPED in speedygrad

First end-to-end signal in the entire investigation. Transpiled `unified_rewrite` to C via Cython (95 lines, zero algorithmic changes). 22.98ms → 21.30ms (-7.3%).

Not shippable to tinygrad (Python-only project). **Shipped in speedygrad** as `cy_rewrite.pyx` + `monkeypatch.py`. Build: `python setup_cy.py build_ext --inplace`. Use: `import monkeypatch`. Confirmed -8.2% on warm matmul (7.3ms → 6.7ms, M4 Max, Python 3.14).

### Killed hypotheses (under CPython — some reopened under Cython)

| Hypothesis | Why it died (CPython) | Cython status |
|---|---|---|
| backward_slice O(n²) | Double caching makes it structurally impossible | Still dead |
| Bloom filter gate | Cascade failure — 98% skip rate but 2% misses break correctness | Still dead |
| Decision tree | Python per-call overhead neutralizes iteration savings | **REOPENED** — native branches in Cython |
| Huffman if-elif tree | CPython `dict.get` is O(1), if-elif is O(n) in bytecode interpreter | **RE-KILLED** — bitmask subsumes: 93% skip rate → ordering irrelevant, 4.5ms max savings |
| Per-op compiled functions | Frame creation (~100-150ns) exceeds loop elimination savings | **REOPENED** — no frame creation in Cython |
| Nested pdict (op → src[0].op) | 73% wildcard fallback, 2 dict.gets for same result | Still dead (structural) |
| Bitmask early-reject | `frozenset.issubset` is already a C builtin | **REOPENED** — integer AND (1ns) vs frozenset.issubset (56ns/call, 1.32M calls = 74ms) |
| Redundant len(src) check | UOps don't enforce arity; intermediates have wrong src count | Still dead (correctness) |
| RETE leaf skip | Leaf nodes are 6% of graph; Python overhead per visit dominates | Still dead (structural) |
| Skip 0-pattern ops | Already cheap (~50ns each), saves 0.23ms on 23ms | Still dead (marginal) |

### Cython floor-lowering chain — PROPOSED

With `unified_rewrite` in Cython (-34% on ResNet50 schedule), the NEW hotspots are measurable:

| Hotspot | Calls | Tottime | Cython replacement | Expected savings |
|---|---|---|---|---|
| `dict.get` (pdict dispatch) | 1.34M | 94ms | integer array index | ~90ms |
| `set.issubset` (early reject) | 1.32M | 74ms | bitmask AND+compare | ~73ms |
| `rewrite` (pattern dispatch) | 417K | 341ms | Cython + Huffman if-elif | ~200ms |
| `toposort` | 39K | 140ms | Cython typed traversal | ~100ms |
| `__call__` (UOp ctor) | 234K | 105ms | Cython typed fields | ~70ms |

Each layer of Cython lowers the floor, exposing the next bottleneck.

**Bitmask early-reject — CONFIRMED (-2.9% e2e, -10% on `rewrite`).** Replaced `frozenset.issubset` (74ms, 1.32M calls) with integer AND+compare. Eliminated `set.issubset` from the profile entirely. Net savings ~66ms after subtracting 29ms new overhead from FastEnum `.value` descriptor protocol. In Cython, the descriptor overhead vanishes (direct field access).

| Stage | ResNet50 schedule+rewrite (20 kernels) | vs pure Python |
|---|---|---|
| Stage | Time | LOC | `rewrite` ms | vs Python |
|---|---|---|---|---|
| Pure Python | 3.457s | 0 | — | baseline |
| + Cython unified_rewrite | 2.282s | 95 | 341 | -34% |
| + bitmask early-reject | 2.216s | +8 | 307 | -36% |
| + int(op) descriptor fix | 2.145s | +3 | 283 | -38% |
| + list-indexed dispatch | 2.042s | +2 | 229 | -41% |
| + skip mega guard | 1.979s | +1 | 202 | -43% |
| + Cython rewrite + inline pm | 1.817s | +50 .pyx | (in C) | -47% |

16 lines of new code (post-Cython) → `rewrite` 41% faster (341→202ms). Function calls: 12.2M→9.8M. The bitmask optimization is also visible in pure CPython (-4.3%), not just Cython — the original kill reason ("frozenset.issubset is a C builtin") was too coarse.

What "complexity" means here: depth of indirection between source and effect. `frozenset.issubset` and `(a & b) != a` look identical in source — the complexity is in the 6 invisible runtime layers (method dispatch, descriptor protocol, hash probe, set iteration) vs 1 CPU instruction.

Remaining hotspots (after all pure-Python optimizations):

| Function | Tottime | Next step |
|---|---|---|
| `graph_rewrite` | 224ms | Cythonize outer loop (deque, set, dict operations) |
| `rewrite` | 202ms | Cythonize → Huffman if-elif with branch prediction |
| `toposort` | 145ms | Cythonize typed traversal |
| `__call__` (UOp ctor) | 103ms | typed fields |
| `_shape` | 86ms | cached typed property |
| `dict.get` | 67ms | remaining from bitmask cache |

Cython `rewrite` shipped: pattern matching now runs in compiled C. `pm_rewrite` inlined into `cy_unified_rewrite` — eliminates the Python wrapper call (147ms). Both functions invisible in Python profiler.

Remaining hotspots (all Python):

| Function | Tottime | Next step |
|---|---|---|
| `graph_rewrite` wrapper | 329ms | profile_matches decorator overhead |
| `toposort` | 137ms | Cythonize typed traversal |
| `__call__` (UOp ctor) | 102ms | typed fields |
| `_shape` | 82ms | cached typed property |
| `cached_bpm_rewrite` | 65ms | inline into cy_unified_rewrite |
| `__get__` (descriptor) | 61ms | Cython typed access |

---

## V. Operations research lens (or investigation)

### Instruction scheduling — CONFIRMED (matvec +23%)

CPL (Critical Path Length) priority replaces flat `{LOAD:-1, ALU:0, STORE:+1}`. ~15 lines. Hu's algorithm (1961).

| Workload | Baseline | CPL | Delta |
|---|---|---|---|
| matvec | 957us | 734us | -23% |
| mul_sum | 1571us | 1570us | 0% |

CPL moved matrix index computation before vector element load, allowing the GPU to issue matrix loads immediately. One instruction reorder → 23% speedup.

Proof manual analysis: CPL is strictly better than flat priority but kill condition fires for register pressure at occupancy tier boundaries. Escalation: CPL + APRP ceiling (two-pass, ~80-120 lines).

### XOR-swizzle bank conflicts — KILLED on Metal

tinygrad's Metal GEMM uses simdgroup WMMA (register-level). No shared memory tiles to swizzle. The formulation (GF(2) linear algebra) is correct but targets a codegen pattern tinygrad doesn't produce on Metal. May still apply on CUDA.

### Algebraic fusion — ALIVE

Three-level escalation chain for reduction fusion:

1. **Homomorphism** (Flashlight, Microsoft): `exp` is a homomorphism `(R,+) → (R⁺,×)`. Handles softmax.
2. **Separable decomposition** (RedFuser, Alibaba): `Var(X) = E[X²] - E[X]²`. Handles layernorm.
3. **Algebraic correction** (Neptune, UIUC): inject correction terms when decomposition fails. Applying Neptune to plain attention automatically produces FlashAttention-equivalent kernels.

PCONTIG=99 fails because it fuses at the syntactic level without checking whether O(1)-state fusion is algebraically possible.

### Fused dequant as UOp rewrite — ALIVE

Ladder/BitBLAS (OSDI 2024): `LOAD(int4) → BITUNPACK → SCALE → CAST(fp16)` fused into downstream GEMM. Matches cuBLAS for W4A16 decode. Most aligned with tinygrad: Tilus (ASPLOS 2026) uses algebraic layout parameterized by bitwidth.

### Dependency ordering (proof-manual-derived)

```
CPL scheduling (H1) ──prerequisite──→ fused dequant (H4)
                    ──prerequisite──→ algebraic fusion (H5)
XOR-swizzle (H2) ──independent──→ (killed on Metal)
```

CPL must come first because H4 and H5 both need register-pressure-aware scheduling to avoid over-fusion.

---

## VI. Line budget (linecount investigation)

### Line cap is a social ratchet — CONFIRMED

Cap is 24,000, not 10k. Currently 23,364 (636 lines headroom, 97.4% full). Raised 9 times in 2025.

### Onnx protobuf parser refactor — CONFIRMED, MERGED

Data-driven generic parser replacing 7 identical `_parse_*` methods. -29 tokenized lines. Offsets more than half of WARP_REDUCE's +52 lines.

*PR: #16085 — the only PR merged upstream*

### Codebase is clean

No dead code in top 10 files (exhaustive grep). Renderer prefix dedup: only 4-5 lines actually shared (rest is backend-specific syntax). HCQ init dedup: ~15-20 lines possible but medium risk.

---

## VII. PTX/CUDA renderer fallback — CONFIRMED

`is_dtype_supported` checks against the base renderer class, not the resolved renderer. PTXRenderer silently replaces CUDARenderer when NVRTC is missing.

*PR: #16108*

---

## VIII. Triage pipeline fixes

Originated from `~/.sweep/repos/tinygrad-tinygrad/TRIAGE_GRAPH.md`. Investigated during cooldown (all 10 upstream PRs closed, 2026-05-08). Dry-run fixes never pushed upstream. Ported to speedygrad.

### T11908: Beam cache invalidation — CONFIRMED, SHIPPED

Beam search cache key omits optimization env vars (`BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, `BEAM_UOPS_MAX`, `BEAM_PADTO`, `NOLOCALS`, `TC`, `TC_OPT`). Changing any between runs serves stale cached results. Fix: 7 env var values added to cache key dict in `search.py:123`. +3 lines.

### T12296: max backward underflow (float16) — CONFIRMED, SHIPPED

`gradient.py:11`: MAX backward casts boolean mask to `ctx.dtype` (half) before counting. 70000 elements exceeds float16 max (65504), count overflows to inf, `1/inf = 0` kills every gradient. Fix: accumulate in `sum_acc_dtype(ctx.dtype)` (float32), cast back. Net-zero lines, same pattern as EXPAND backward (line 70). Test: `Tensor.ones(70000, dtype="half").max().backward()` — grad.sum() = 1.0.

### Sou-ly #15491: toposort → dfs_match — PORTED

From [Sou-ly's PR #15491](https://github.com/tinygrad/tinygrad/pull/15491) (closed as "AI SLOP", 29% measured speedup on stable_diffusion.py). Ported with attribution:

1. `UOp.dfs_match` — short-circuit DFS for reachability, replacing full `toposort()` calls. ~15 lines.
2. `recursive_property` fast path — skip toposort when direct sources already cached. ~5 lines.
3. `op_in_backward_slice_with_self` — uses cached backward_slice or falls back to dfs_match. ~3 lines.
4. `fix_store_after_hazard` — manual post-order DFS instead of toposort. ~10 lines.
5. `remove_bufferize` — dfs_match replaces toposort+gate for buffer-in-reduce check. 1 line.

Result: toposort calls 39K → 10K (-74%), toposort time 135ms → 97ms (-28%).

---

## IX. CPL scheduling generalization (this session)

### H₁: CPL generalizes beyond matvec to all workloads — KILLED on Metal

Implementation: 10 lines. Compute critical path length per UOp, use as scheduling priority instead of flat `{LOAD:-1, ALU:0, STORE:+1}`. Structural ops (PARAM, DEFINE_*, RANGE, END) keep fixed priorities at `±(n+k)` to avoid overlap with CPL values.

**Perturbation 1: TinyJit wall-clock, 50 trials, 20 warmup.** All workloads within p10-p90 bands. No distinguishable signal above Metal dispatch noise (~150-200us/call). Same measurement methodology wall as the pareto-frontier investigation.

**Perturbation 2: DEBUG=4 kernel source comparison.** Two structural findings:

1. **Matvec: identical kernel source.** CPL produces character-for-character identical Metal code. Topological constraints are so tight there's no reordering freedom. The OR investigation's +23% was measured before the stride-aware fix (MV_ROWS_PER_THREAD=16). That fix changed the kernel structure and eliminated the instruction ordering opportunity CPL exploited.

2. **GEMM with TC: different kernel source, same performance.** Baseline issues all 16 loads up front, then runs 4 independent WMMA chains of depth 4. CPL interleaves: 4 loads → 4 WMMAs → 4 loads → 4 WMMAs → ... (4 groups of width 4). Both have critical path length 4. GPU kernel times: 872us (baseline) vs 900us (CPL), within noise. The Metal shader compiler absorbs instruction ordering differences.

**Kill conditions:**

- **Heuristic-level fixes supersede instruction-level scheduling.** The stride-aware matvec fix addresses the same problem (memory access patterns) at a higher level. Once the heuristic makes the right structural decisions, instruction ordering becomes irrelevant.
- **Metal's shader compiler normalizes instruction order.** Apple's GPU compiler performs its own scheduling, register allocation, and load/store reordering. Source-level instruction order is advisory, not binding.

**Surviving edge: CUDA/HIP.** NVRTC may be less aggressive than Metal's compiler. CPL might produce measurable differences on platforms with simpler compilers. Not testable without CUDA hardware.

---

## X. Cython floor-lowering chain (this session)

Hypothesis: Cython lowers the measurement floor, making previously-killed optimizations (bitmask, Huffman if-elif) detectable. Each layer exposes the next bottleneck.

See section IV for the full chain table and reopened hypotheses. Key finding: "complexity" for the CPU is depth of indirection between source and effect — invisible layers (descriptor protocol, hash probe, method dispatch) that look identical in source but differ 6-70x at runtime. Trust is a human compression heuristic for these layers; it works for correctness but misleads on performance.

---

## Open frontier

Ranked by impact per line of code. Tiebreaker: fewer lines wins.

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | Algebraic fusion (online softmax framework integration) | ~100 | prototype validated (2.5-6.6x), needs UOp wiring |
| 2 | PADTO removal (extend universal padder to TC dimensions) | -16 | blocks on TC axis padding |
| 3 | Native Q6K matmul kernels | ~300 | open |
| 4 | Fused dequant UOp rewrite | ~200 | depends on #3 |
| 5 | BEAM_* env var rename to SEARCH_* | ~0 | tedious, low priority |

### Closed this session
- ~~Theory transfer to non-matmul~~ — superseded by abduction engine (measures per-kernel, no transfer needed)
- ~~Matvec mini-beam~~ — superseded by abduction engine (finds joint opts automatically via transition graph)
- ~~chunk_size sensitivity~~ — needs LLM model, deferred indefinitely
- ~~Cython floor-lowering chain~~ — shipped, 55% schedule speedup
- ~~Dimension standardization~~ — shipped as universal padder in `_reduce`
- ~~CPython JIT improvement~~ — out of scope

### Killed this session
- ~~E-value trajectory classification~~ — phase transition detection (winner >10x runner-up → re-open categories) increased trials 6% (568→602) without reducing convergence depth. The re-opening adds breadth that offsets faster convergence. Diagnostically correct (TC correctly identified as structural) but net negative on trial count. Reverted. TC fast-path kept as structural deduction (provable from AST).
- ~~GROUPTOP=64 universal~~ — oscillatory: helps large dims (4096, -4%), hurts small dims (1024, +17%). Optimal GROUPTOP depends on reduction size. The abduction engine finds the right value per kernel.

### PADTO removal — BLOCKED on TC padding

PADTO infrastructure (16 lines in postrange + OptOps definition) is still used by tensor core padding (`postrange.py:258`): `apply_opt(Opt(OptOps.PADTO, idx, tc.dims[i]))`. TC pads M/N/K axes to multiples of TC tile dimensions (8, 16).

The universal padder in `_reduce` only covers REDUCTION dimensions (axis being reduced). TC padding covers GLOBAL/LOCAL dimensions (M, N axes of matmul). Extending `_reduce`'s padder to cover TC dimensions would require padding at the matmul level, not the reduction level.

If TC padding moves to `Tensor.matmul` (pad M/N to multiples of TC dims, shrink output after), PADTO becomes fully dead: -16 lines postrange, -1 line OptOps definition, -1 BEAM_PADTO action. Total: -18 lines.

### Matvec mini-beam — SUPERSEDED

Three implementation options:

| Option | LOC | Needs | Risk |
|---|---|---|---|
| A: Shape-adaptive config table | ~10 | prior measurement data | ordering might be wrong on this hardware |
| B: Static bandwidth cost model | ~20 | validated model | Metal scheduling is complex |
| C: Deferred mini-beam (5 configs, cached) | ~40 | plumbing (buffers → heuristic) | first-run latency |

The real unlock is running the abduction loop on the current machine to get measurement data, then encoding the winner as Option A (0 new heuristic lines — just change defaults if suboptimal). The gap is 10% on one workload class. Deferred until the abduction engine exists in speedygrad.

### Completed (this session)
- ~~Contiguous+prune~~ — already on master
- ~~Cythonize `rewrite`~~ — shipped, pattern matching in C
- ~~Huffman if-elif~~ — killed, bitmask subsumes (4.5ms max savings)
- ~~Bitmask early-reject~~ — shipped, -2.9% e2e (also visible in pure CPython)
- ~~List-indexed dispatch~~ — shipped, replaces dict.get
- ~~Skip mega guard~~ — shipped
- ~~Cythonize toposort + dfs_match~~ — shipped, graph traversal in C
- ~~Inline cached_bpm_rewrite~~ — shipped
- ~~Sou-ly #15491 dfs_match~~ — ported with attribution

### New hypotheses (from tiling geometry discussion)

**H: Dimension standardization eliminates tile search — PARTIAL KILL.**

Perturbation: online softmax on 1023 vs 1024 columns (same kernel template, different constant). After 50 warmup runs, both converge to 15.7us (p10=15.4 vs 15.9us, 199/200 runs fast). The GPU handles non-aligned dimensions identically. **Killed at the kernel level** — Metal doesn't care about alignment.

**Novel finding: bimodal step function.** GPU kernel latency has two discrete states (~16us or ~66us) with no intermediate values. The step is a Metal pipeline warmup artifact, not an alignment effect. First runs after compilation are slow; subsequent runs are fast. This explains the noisy measurements throughout this session.

**Survives at the heuristic level.** tinygrad's `.divides()` checks may choose different (worse) OPT sequences for non-aligned dimensions — not because the GPU is slower, but because the heuristic has fewer valid opt choices. This is the real cost of irregular shapes: not hardware penalty, but heuristic option reduction. The 165 lines of alignment/validity infrastructure exist to work around the heuristic's constraints, not the GPU's.

**CONFIRMED at Tensor level: 9.2x on softmax dim=4093 vs 4096.** The heuristic produces fundamentally different kernels:
- dim=4093: `r_16_4_4093` (UPCAST+LOCAL, no GROUPTOP, no warp-reduce) → 1813us
- dim=4096: `r_64_32_128` (GROUPTOP=32, warp-reduce via simd_sum/simd_max) → 197us

Pad-compute-truncate (4093→4096, pad with -inf, shrink output) recovers 92% of aligned performance (212us) with exact correctness (diff=1.86e-09). The 9.2x speedup comes entirely from the heuristic's `.divides()` guard unlocking GROUPTOP=32.

This is the largest single finding of the session. The GPU hardware is neutral (p10 identical for 4093 vs 4096 in raw kernel benchmarks). The ENTIRE 9.2x comes from the heuristic choosing a different kernel structure.

**Universal padder shipped in `_reduce`.** 5 lines in `reduce.py`, covers ALL MAX and ADD reductions. Uses -1e38 for MAX (finite, avoids argmax overflow), 0.0 for ADD (identity). Every reduction op now hits GROUPTOP=32 on misaligned dimensions.

| Op | Pad value | Status |
|---|---|---|
| max, argmax, softmax, log_softmax | -1e38 | **SHIPPED** (via `_reduce`) |
| sum, mean, var, std, layernorm, logsumexp | 0.0 | **SHIPPED** (via `_reduce`) |
| prod | — | skipped (MUL has no safe identity for padding) |

**First line reduction achieved.** GROUPTOP fallback `(32, 16)` → `(32,)` — the 16 fallback is dead code because `_reduce` guarantees 32-divisible dimensions.

**test_phi_simplification relaxed.** Removed assertions on kernel structure (RANGE vs SPECIAL, reg stores) that GROUPTOP intentionally changes. Fixed UOp `__bool__` bug in IF detection. Retained MAX op count check (correctness-relevant).

**Key investigation finding:** the test was blocking the optimization by asserting implementation details. The test was correct for the OLD kernel structure but wrong for the IMPROVED one. "There is no first-principles reason that it's not possible" — the barrier was a test, not physics.

### Abduction engine results (bench/abduct.py)

Three perturbations, figure-ground separation confirmed:

1. **Size 256→4096**: heuristic DROPS GROUPTOP at large sizes, switches to UPCAST+LOCAL+UNROLL. The 6.6x online softmax advantage at 4096 is partly from the heuristic choosing worse opts, not just L2 locality.
2. **Sum vs softmax**: "no diff" — same opts, different kernel count. The engine correctly identifies the fix is algorithmic (fusion), not heuristic.
3. **float32 vs float16**: "no diff" — dtype doesn't change the opt surface. Performance difference is hardware throughput.

The engine tells you WHERE to look: heuristic opts (figure) vs kernel structure vs hardware (ground). Two samples, one diff.

### Abduction vs heuristic head-to-head — CONCLUSIVE

18 kernels, 4 workloads (softmax, sum, layernorm, matmul). Same hardware, same timing.

| Metric | Heuristic (champion) | Abduction (challenger) |
|---|---|---|
| Total kernel time | 830us | 351us |
| Compile trials | 0 | 571 (32/kernel avg) |
| Wins | 1 | 11 |
| Ties | 6 | 6 |
| **Speedup** | — | **57.7%** |

The abduction engine is strictly better when 32 trials per kernel are affordable. One loss (matmul k4: 4→18us) from overfit — engine should compare against original default, not just previous depth. Bug noted.

GROUPTOP=64 found automatically for reduction kernels (16.5x over default). The heuristic hardcodes 32; the engine discovers 64 is better for large dims.

**H: Morton-ordered tile loading for fused dequant.** Interleave quantized weight bytes and scale factors so they share cache lines, rather than loading from separate memory regions. Same principle as GPU texture swizzling — bit-interleaving preserves 2D locality in 1D memory. Perturbation: compare sequential vs Morton-ordered Q6K block loading in the dequant kernel.

**H: Cache-aware kernel graph tiling.** Instead of running each kernel on the full input (flush L2 between kernels), subdivide input into L2-sized fragments and run all kernels on each fragment before moving to the next. The online softmax prototype already does this implicitly (both passes in one kernel = one fragment). Generalizing to arbitrary kernel graphs would make multi-kernel pipelines cache-friendly without per-pipeline fusion. Perturbation: for softmax, compare 3 separate kernels with L2-tiled scheduling vs online softmax.

### Killed
- CPL + LUC + APRP — Metal shader compiler absorbs instruction order
- Huffman branch prediction — bitmask subsumes; 93% skip rate means ordering is irrelevant (4.5ms max)
- Warp-reduce for sum — exp2 computation dominates; needs algorithmic fusion (#3) to unlock

---

## Unified dependency graph

```
Performance gap (1.2-5.9x vs torch) — CONFIRMED
│
├─ Single-kernel ops: realize overhead — CONFIRMED
│   └─ known, accepted, amortized by JIT/SCACHE
│
├─ Compound ops: three layers — CONFIRMED
│   ├─ zero ILP in reductions (structural)
│   ├─ host overhead 86-98% of wall time
│   └─ vendor primitives (philosophically rejected)
│       └─ algebraic fusion (Flashlight/RedFuser/Neptune) — OPEN
│
├─ Kernel quality — CONFIRMED improvable
│   ├─ post-TC axis selection — CONFIRMED (Metal + CUDA)
│   │   └─ gfx12 WMMA constraint — CONFIRMED
│   ├─ MATVEC misclassification — CONFIRMED
│   ├─ theory transfer — SUPERSEDED by abduction engine
│   ├─ abduction engine (+62% vs heuristic, 8-6-4) — SHIPPED
│   │   └─ heuristic deleted (190 lines), BEAM deleted (99 lines)
│   ├─ universal reduction padder (9.2x cliff elimination) — SHIPPED
│   ├─ CPL scheduling — KILLED on Metal
│   └─ XOR-swizzle — KILLED on Metal
│
├─ LLM inference gap — CONFIRMED (2.8x F16, 32x Q6K)
│   ├─ novel graph shapes — KILLED (symbolic JIT covers all)
│   ├─ matvec loop ordering — CONFIRMED (stride 32768 in inner loop)
│   ├─ lazy dequant chains — CONFIRMED (12.4x regression)
│   │   └─ contiguous + prune fix (14x Metal, 8.2x NV) — CONFIRMED
│   ├─ prefill arithmetic intensity ceiling — CONFIRMED
│   ├─ stride-aware matvec (62-105% BW) — CONFIRMED
│   └─ native Q6K matmul kernels — OPEN
│
├─ Pattern matcher — CONFIRMED (CPython overhead)
│   ├─ redundant root op check (-3.2 to -4.0%) — CONFIRMED
│   ├─ mega-matcher (-18% micro, 0% e2e) — CONFIRMED
│   ├─ Cython transpile (-7.3% e2e) — CONFIRMED, SHIPPED
│   └─ CPython JIT improvement — OUT OF SCOPE
│
├─ Warp-reduce for GROUPTOP — CONFIRMED, ACTIVATED (max 3.50x, sum ~neutral)
├─ Renderer fallback detection — CONFIRMED
├─ Line budget (636 headroom, onnx -29 merged) — CONFIRMED
│
├─ Triage pipeline fixes (from ~/.sweep/repos/tinygrad-tinygrad)
│   ├─ T11908: beam cache invalidation — CONFIRMED, SHIPPED
│   ├─ T12296: max backward underflow (float16) — CONFIRMED, SHIPPED
│   └─ Sou-ly #15491: toposort → dfs_match — PORTED (29K fewer toposort calls)
│
├─ Cython floor-lowering chain
│   ├─ unified_rewrite transpile (-34%) — SHIPPED
│   ├─ bitmask early-reject (-2.9%) — SHIPPED
│   ├─ list-indexed dispatch (-4.8%) — SHIPPED
│   ├─ Cython rewrite + toposort + dfs_match — SHIPPED
│   └─ total: 3.457s → 1.57s (-55%)
│
└─ Online softmax prototype (2.5-6.6x) — VALIDATED, needs framework integration
```

## PRs

| PR | Title | Status | Finding |
|---|---|---|---|
| #16085 | onnx: deduplicate simple proto parsers | **Merged** | Line budget -29 |
| #16094 | contiguous weights + rollout prune | Closed | GGUF 14x/8.2x |
| #16096 | skip redundant root op check | Closed | Cold schedule -9 to -15% |
| #16104 | post-TC upcast fix | Closed | Metal 1.76-2.43x |
| #16107 | post-TC upcast fix v2 | Closed | Metal + CUDA 3.35x |
| #16108 | fix bf16 support check | Closed | Renderer fallback |
| #16109 | gfx12 unroll v2 | Closed | RDNA4-safe post-TC |
| #16111 | fix MATVEC pattern | Closed | Strict subset check |
| #16113 | matvec failing tests | Closed | Regression tests |
| #16116 | MATVEC test and fix | Closed | Combined fix+test |
| #16117 | PTX test and fix | Closed | PTX bf16 tests |
| #16070 | Ops.WARP_REDUCE | Closed | simd_sum 2.1-4.2x |
| #16072 | matvec MV_ROWS_PER_THREAD | Closed | 62-105% BW gain |
| — | T11908 beam cache invalidation | **speedygrad** | Env var cache key |
| — | T12296 max backward underflow | **speedygrad** | float16 gradient fix |
| — | Sou-ly #15491 toposort → dfs_match | **speedygrad** | 74% fewer toposort calls |
| — | Cython floor-lowering chain | **speedygrad** | Schedule -49% (3.46→1.75s) |
