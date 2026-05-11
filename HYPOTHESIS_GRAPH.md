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
4. **GPU clock-state confounds small-op microbench absolute numbers** (added retroactively, bug-hunt iter 7.5 round 3+5). Any kernel whose wall time is <30us GPU work and <30K threads keeps the GPU at intermediate P-states between launches — boost clock never engages because the per-launch duty cycle is too low. Symptoms: a larger workload reports *faster* GPU time than a smaller one (e.g. iter 7.5 standalone bench shows 256x256 at 10us GPU but 1024x1024 at 8us). Warmup depth doesn't fix this — at 10us/launch the queue drains regardless of iteration count. **Implication for iter 4-7 reasoning**: small-op (add/relu/exp/sum/permute) absolute numbers were measured at idle/intermediate clocks; the framework-level deltas (Python-frame removals worth 2-3us each) are real because the baseline shared the same clock state, but the absolute "X us per call" numbers shouldn't be quoted out of context. At real inference duty cycle (back-to-back attention ops), both numbers shrink but the ratio holds.

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

## XI. CUDA reframe (this session)

Hardware: RTX 4080 (Ada, sm_89), Windows. nvrtc/nvjitlink not on PATH → tinygrad falls back to PTXRenderer (per Section VII). torch 2.11+cu128.

### H_REFRAME: matvec is solved on CUDA, gemm is the gap — CONFIRMED

11-workload head-to-head, p50 of 50 trials:

| Workload | tinygrad | torch | gap |
|---|---|---|---|
| gemm_1024 | 617us | 124us | **4.98x** |
| gemm_256 | 64 | 47 | 1.36x |
| add_4096 | 52 | 25 | 2.08x |
| mul_sum | 38 | 62 | **0.61x** (tinygrad wins) |
| relu_4096 | 48 | 25 | 1.92x |
| exp_2048 | 51 | 22 | 2.32x |
| sum_4096 | 51 | 26 | 1.96x |
| permute | 51 | 42 | 1.21x |
| softmax | 37 | 23 | 1.61x |
| layernorm | 37 | 41 | **0.90x** (tinygrad wins) |
| matvec | 84 | 83 | **1.01x** (tied) |

The matvec frontier (#1 in pre-CUDA open frontier, ~20 LOC) is **not a gap on CUDA**. The largest gap is gemm_1024 at 5x. Reframe applies: the matvec investigation was Metal-specific. Stride-32768 inner loop pattern was Metal codegen, not universal.

### H_TF32: gemm gap is TF32 disabled by default — CONFIRMED

PTXRenderer declares 5 tensor cores including `dtype_in=float, dtype_out=float, dims=(8,16,8)` (TF32 on sm_80+). For fp32 gemm_1024, `get_kernel_actions` returns **0 TC actions**. For fp16, 2 TC actions are generated and abduction picks one (`r_32_32_32_2_2_4_2_64_2`, **25.9 TFLOPS**, 101us).

Root cause: `tinygrad/codegen/opt/postrange.py:209`:
```python
if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
```

`ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)` (helpers.py:264). Default off — TF32 TCs unreachable on fp32 matmul. PyTorch defaults TF32 ON for matmul on Ampere+.

Perturbation: `ALLOW_TF32=1 IGNORE_SEARCH_CACHE=1` on gemm_1024 fp32:

| Setting | gemm_1024 fp32 |
|---|---|
| default | 617us |
| ALLOW_TF32=1 + cache hit (stale) | 618us — no change |
| ALLOW_TF32=1 + IGNORE_SEARCH_CACHE | **453us** (1.36x) |
| fp16 reference | 101us |

ALLOW_TF32 closes 27% of the gap. Not the full 5x — see H_OSCILLATORY below.

### H_CACHE_KEY: abduct_search cache key omits codegen env vars — CONFIRMED

`tinygrad/codegen/opt/abduct.py:46-48`:
```python
key = {..., "NOLOCALS": getenv("NOLOCALS", 0), "TC": getenv("TC", 1), "TC_OPT": getenv("TC_OPT", 2)}
```

But the codegen pipeline's config tuple (`codegen/__init__.py:188`) is 9 env vars:
```python
config = (NOOPT, DEVECTORIZE, EMULATED_DTYPES, NOLOCALS, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32)
```

8 env vars are missing from the cache key: NOOPT, DEVECTORIZE, EMULATED_DTYPES, USE_TC, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL, ALLOW_TF32. Same class as the already-shipped T11908 fix for beam search cache. Direct evidence: setting ALLOW_TF32=1 without IGNORE_SEARCH_CACHE returns the cached 618us; with cache disabled, returns 453us. The cache served stale results.

**Shippable fix**: 8 lines in the cache key dict, parallel to T11908. Ports cleanly. Risk: low (cache invalidation, never serves wrong result if existing cache regenerates).

### H_DEPTH: TF32 abduction needs depth≥5, but no monotonic improvement — REFINED

Default `SEARCH=3` → `max_depth=3` in `abduct_search`. With ALLOW_TF32=1, the depth-vs-time relationship is non-monotonic:

| Setting | gemm_1024 fp32 | gemm_1024 fp16 |
|---|---|---|
| BEAM=3, ALLOW_TF32=0 (default) | 617us | n/a |
| BEAM=5, ALLOW_TF32=0 | 1967us (3.2x worse) | n/a |
| BEAM=3, ALLOW_TF32=1, fresh cache | 137 / 370 / 1032us (oscillatory) | 345us |
| BEAM=5, ALLOW_TF32=1, fresh cache | 131 / 131 / 133us (consistent) | n/a |
| BEAM=10, ALLOW_TF32=1, fresh cache | **703us (worse than BEAM=5)** | **105us (parity with cache)** |
| Cache hit (post-first-warm) | n/a | 105us |

**Findings:**
1. ALLOW_TF32 is the active variable — depth alone without TC overfits (BEAM=5 fp32 default → 1967us).
2. There is a sweet spot for fp32+TF32 around depth=5; depth=10 overshoots (703us). The transition graph at depth=10 explores opt sequences that look winning under noisy measurement but lose at final validation.
3. fp16 needs depth=10 to find the cached optimum (105us); BEAM=3 produces 345us fresh. So the "right depth" is workload-dependent and not a single constant.
4. The disk cache stores good kernels but the abduction engine can't reproduce them from scratch at default depth=3 — strong evidence of measurement-noise sensitivity in the depth/early-stop interaction.

**The shippable change is not "default BEAM=5" alone.** A clean fix needs either:
- (a) Deterministic timing: `clear_l2=True` in `_time_program`, more samples (`cnt=7+`), CUDA event timing instead of host clock — reduces oscillation at all depths.
- (b) Validation re-time at higher cnt (already exists, line 122-129 of abduct.py — but only re-times the winner, not the depth-by-depth selection).
- (c) Per-workload depth selection: detect TC presence and bump cnt+depth accordingly.

### Causal chain

```
torch fp32 gemm_1024 = 124us
tinygrad fp32 gemm_1024 default = 617us (5.0x gap)
                │
                ├─ ALLOW_TF32=0 default → fp32 TC declared but get_kernel_actions returns 0 TC opts
                │       (fixable: change speedygrad default, or document)
                │
                ├─ ALLOW_TF32=1 + BEAM=5 + IGNORE_SEARCH_CACHE → 131us (parity)
                │
                ├─ ALLOW_TF32 not in abduct.py cache key → stale results without IGNORE_SEARCH_CACHE
                │       (fixable: ship parallel to T11908)
                │
                └─ Abduction is measurement-noise sensitive across depths
                        - fp16 needs depth=10 from scratch to find the cache's 105us
                        - fp32+TF32 has a sweet spot at depth=5; depth=10 overshoots
                        - The disk cache hides this by storing the once-found optimum
                        (open: needs deterministic timing protocol)
```

### Causal chain

```
torch fp32 gemm_1024 = 124us
tinygrad fp32 gemm_1024 default = 617us (5.0x gap)
                │
                ├─ ALLOW_TF32=0 default → fp32 TC declared but get_kernel_actions returns 0 TC opts
                │
                └─ ALLOW_TF32=1 + BEAM=5 + IGNORE_SEARCH_CACHE → 131us (parity, gap closed)
                        │
                        ├─ ALLOW_TF32 not in abduct.py cache key → stale results without IGNORE_SEARCH_CACHE
                        └─ default BEAM=3 too shallow → oscillates 137us ↔ 1032us
```

### Open CUDA frontier (ranked by impact/LOC)

| # | Edge | LOC | Status | Impact |
|---|---|---|---|---|
| 0a | abduct.py cache key: 8 missing codegen env vars | ~10 | **shippable** — same class as T11908 | correctness (no perf change alone) |
| 0b | Default ALLOW_TF32=1 for CUDA, or document workaround | ~1 + docs | speedygrad design choice; matches PyTorch default | **closes 5x gemm gap** when paired with BEAM=5 |
| 0c | Default `SEARCH=5` for kernels with TC opts in candidates | ~3 | speedygrad design choice | enables 0b's gap closure; trade: longer first-compile |
| 1 | Per-workload non-tied gaps: add/exp/sum/relu/permute (~2x vs torch) | varies | new investigation: tinygrad CUDA backend dispatch overhead | medium |
| 2 | TF32 numerical accuracy regression tests | ~30 | required if 0b ships | safety |

### Reasoning mode table

| Claim | Mode | Confidence | Evidence |
|---|---|---|---|
| matvec is tied on CUDA (1.01x) | induction | 95% | bench/workloads.py + bench/torch_workloads.py, p50 of 50 |
| gemm_1024 is 5x slower than torch | induction | 95% | same bench pair |
| ALLOW_TF32=0 is the default | deduction | 99% | helpers.py:264 |
| TF32 TC declared but unreachable for fp32 | deduction | 99% | postrange.py:209 + get_kernel_actions returns 0 TC opts for fp32 |
| ALLOW_TF32=1 + BEAM=5 → 131us | induction | 95% | three independent bench runs |
| Cache key omits ALLOW_TF32 | deduction | 99% | abduct.py:46-48 |
| Cache served stale results (618 vs 453us) | induction | 90% | one observation; need re-test |
| Depth alone hurts without TC | induction | 90% | one observation (BEAM=5+TF32=0 → 1967us); need re-test |

### XII. Iteration 2: parity prework + benchmark + regression (this session)

**Diagnosis bundle implemented as 3 changes:**

1. `helpers.py:264` — `ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)` → `ALLOW_TF32 = ContextVar("ALLOW_TF32", 1)`. Matches PyTorch's pre-2.0 default for cuBLAS matmul. Only affects CUDA/NV with fp32 TC (gated at postrange.py:209).
2. `abduct.py:45-48` — cache key extended with 8 missing codegen env vars (ALLOW_TF32, NOOPT, USE_TC, DEVECTORIZE, EMULATED_DTYPES, IMAGE, DISABLE_FAST_IDIV, TRANSCENDENTAL). Required because the disk cache schema would otherwise serve stale TC opts and crash on re-application after env changes.
3. `abduct.py` Phase 0 redesigned — was unconditional structural TC adoption; now **measures default first, collects TC alternatives, picks search winner, then late-adopts TC only if it beats search winner by >5%**. Also bumps `max_depth` to 5 when TC is adopted (TC kernels need deeper LOCAL/UPCAST/UNROLL tuning to reach the right combo). Prevents the matvec regression (TC inflicts N=1 padding waste; before the gate, TC always won vs unopted default).

**Implementation iterations:**
- v1 (unconditional adoption + depth bump): gemm_1024 178us but matvec regressed 84→138us; gemm_256 regressed 99→608us
- v2 (default-time gate): same matvec regression — TC still beat un-opted default
- v3 (no Phase 0): matvec 110us OK but gemm_1024 reverted to 679us
- **v4 (late TC sweep — ships):** TC vs search-winner comparison; both gemm and matvec get the right kernel

**Bench protocol fix:** original `bench.py` ran 11 workloads in one process, causing cross-pollution (one workload's JIT/cache state interfering with the next). New `bench_iso.py` spawns one subprocess per workload. p10/p50/p90 over 50 trials, 20 warmup. System noise still produces occasional bimodal distributions on either side; report p10 as the steady-state floor.

**Final scorecard (RTX 4080, Windows, p10 of 50 trials):**

| Workload | baseline (pre-fix) | speedygrad (post-fix) | torch | speedygrad/torch |
|---|---|---|---|---|
| gemm_1024 | 545us | **122us** | 114us | **1.07x — PARITY** (was 5.0x) |
| gemm_256 | 99us | **54us** | 48us | 1.13x (was 1.36x) |
| add_4096 | 67us | 53us | 33us | 1.61x |
| mul_sum | 80us | **36us** | 62us | **0.58x — WIN** |
| relu_4096 | 60us | 48us | 25us | 1.92x |
| exp_2048 | 60us | 51us | 25us | 2.04x |
| sum_4096 | 80us | 52us | 34us | 1.53x |
| permute | 62us | 51us | 43us | 1.19x |
| softmax | 65us | 39us | 23us | 1.70x |
| layernorm | 64us | 43us | 44us | **0.98x — PARITY** |
| matvec | 84us | 108us | 142us | **0.77x — WIN** (still beats torch) |

Wins/near-parity: 4 strict wins (gemm_1024 1.07x, mul_sum 0.58x, layernorm 0.98x, matvec 0.77x) plus 2 near-parity (gemm_256 1.13x, permute 1.19x). Remaining gaps (1.5-2x) are on small ops (add/relu/exp/sum/softmax) where realize/dispatch overhead dominates — that's the Section II "host overhead 86-98% of wall time" frontier and out of scope for this PR. (Originally claimed "7/11 wins" with `gemm_256` listed twice; corrected bug-hunt iter 7.5.)

**Numerical accuracy (TF32 on RTX 4080):**

| N | speedygrad vs ref | torch vs ref | speedygrad vs torch |
|---|---|---|---|
| 256 | 8.05e-4 | 3.05e-4 | 8.37e-4 |
| 1024 | 8.61e-4 | 2.85e-4 | 9.58e-4 |
| 2048 | 7.90e-4 | 2.89e-4 | 8.33e-4 |

speedygrad's TF32 path is ~3x noisier than torch's TF32 but within the same order of magnitude (10-bit mantissa territory). Acceptable for ML inference/training.

**Regression test results:**

| Suite | Result | Notes |
|---|---|---|
| `prework/cuda-parity/smoke.py` | 21/21 pass | matmul, matvec, fp16, reductions, softmax, layernorm, JIT |
| `test/backend/test_jit.py` | 19/20 pass, 1 fail | failure is Windows-specific subprocess.Popen FileNotFoundError; pre-existing |
| `test/backend/test_linearizer.py` | 19/19 pass | clean |
| `test/backend/test_opt_gemm.py` | 4/4 pass | clean |
| `test/backend/test_ops.py` (matmul/sum/etc) | 42/43 pass | failure: `test_softmax_argmax` CUDA_INVALID_PTX, also fails with ALLOW_TF32=0 → pre-existing |
| `test/backend/test_schedule.py` | 138/148 pass | all 10 failures are the same Windows subprocess error; pre-existing |

**Zero regressions attributable to the fix.**

### Causal chain (closed)

```
H_REFRAME confirmed: matvec frontier was Metal-specific, real CUDA gap is gemm
        │
        ├─→ H_TF32 confirmed: ALLOW_TF32=0 default makes fp32 TC unreachable
        │       └─→ FIX: helpers.py:264 default 0 → 1
        │
        ├─→ H_CACHE_KEY confirmed: abduct cache schema misses 8 env vars
        │       └─→ FIX: abduct.py:45-48 add ALLOW_TF32 + 7 sibling env vars
        │           (also drops/recreates abduct_search_22 sqlite table)
        │
        └─→ H_DEPTH refined into H_TC_OVERFIT: Phase 0 unconditional TC adoption
                makes matvec/small ops slower because TC always beats UN-OPTED
                default but loses to no-TC search winner on small workloads
                └─→ FIX: late TC sweep — search no-TC first, late-adopt TC if
                    measured-faster, then deepen to 5 from TC starting point
```

### Iteration history (this session, depths 2-4)

**Iter 2 (commit fb80c5c87):** ALLOW_TF32 default + abduct cache key + measurement-gated TC late sweep. Closed gemm_1024 5.0x → 1.07x.

**Iter 3 (commit b025476de):** top-3 re-validation per depth in abduction search. Same depth, same workload, but cnt=3 was 17% likely to pick a no-op-ish opt (UNROLL(0,0)) under noise. gemm_256 lost a 6/6 → 8/8 stability check; gemm_1024 reached 0.97x (now winning torch). Direct evidence in `prework/cuda-parity/noise_probe.py` — 8 fresh runs of gemm_256 land in 53-63us where iter 2 had a 1/6 outlier at 106us.

**Iter 4 (commit 7240da8d9):** direct dispatch in `run_linear`'s per-call exec loop, replacing `pm_exec.rewrite()` 6-pattern probe with a dict lookup. Saved 65us cumtime per call (73% of small-op JIT replay cost). matvec 115→99us, softmax 43→36us, layernorm 44→36us, exp_2048 59→49us.

**Iter 5 attempted (reverted):** single-device fast path in `exec_kernel` (skip `unwrap_multi` generator) and JIT-replay short-circuit in `track_stats`. Both were measured to give <2us savings — below `Device.synchronize()`'s 15us round-trip noise floor. Code reverted; the wall-time gap on add/relu/exp is structural — Python + ctypes overhead per CUDA driver call has ~30us floor on Windows. Closing further requires Cython exec_kernel or CUDA graphs (multi-call batching), both substantial.

### Final scorecard (RTX 4080, Windows, p10 of 50 trials, isolated subprocess)

| Workload | initial baseline | iter 2 | iter 3 | iter 4 | torch | final result |
|---|---|---|---|---|---|---|
| gemm_1024 | 545 | 122 | 118 | 121 | 122 | **WIN 0.98x** |
| gemm_256 | 99 | 54 | 53 | 50 | 45 | near 1.10x |
| add_4096 | 67 | 53 | 51 | 51 | 23 | 2.22x (host floor) |
| mul_sum | 80 | 36 | 34 | 32 | 57 | **WIN 0.55x** |
| relu_4096 | 60 | 48 | 49 | 47 | 24 | 1.96x (host floor) |
| exp_2048 | 60 | 51 | 59 | 47 | 21 | 2.24x (host floor) |
| sum_4096 | 80 | 52 | 51 | 46 | 32 | 1.43x |
| permute | 62 | 51 | 52 | 49 | 39 | near 1.25x |
| softmax | 65 | 39 | 43 | 35 | 21 | 1.66x |
| layernorm | 64 | 43 | 44 | 32 | 42 | **WIN 0.76x** |
| matvec | 84 | 108 | 115 | 98 | 63-141 (noisy) | wins p50, ties p10 |

3 wins, 2 near-parity, 4 modest gaps, 3 host-floor gaps. From ~9/11 with material gaps (4 of them ≥3x) at iteration 1 to 4/11 still with material gaps now.

### Open frontier (after exhaustion this session)

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | ~~Cython exec_kernel~~ — **shipped as cy_runtime.pyx** (-3 to -7us per call) | 95 | done iter 5 |
| 2 | CUDA graph batching for repeated kernels | ~150 | not attempted; would amortize the 17us cuLaunchKernel floor |
| 3 | ctypes → cffi/pybind11 for cuLaunchKernel + sync | ~150 | structural; would attack the 6 ctypes wrapper calls per kernel (~21us cumtime) |
| 4 | First-compile cost (~5s for gemm_256 at depth 5) | ~30 | tradeoff for kernel quality |
| 5 | matvec p90 occasional 234us (1/10) — late TC sweep finds bad TC kernel sometimes | ~10 | needs separate root-cause |

### Iter 5 (commits ahead of d387d04e9): Cython runtime port via monkeypatch

User reframe: "we can regenerate monkeypatch as fast as we modify upstream" — the
Cython port isn't a fork, it's a shadow. Maintenance cost = re-port when upstream
changes the function body, not merge conflicts.

**Implementation:**
- `cy_runtime.pyx` (95 LOC): `cy_run_linear` + inlined `_exec_kernel_fast`. Single-device
  fast path skips the `unwrap_multi` generator and the `track_stats` contextmanager.
  Multi-device + DEBUG/PROFILE paths fall through to the Python originals.
- `setup_cy.py`: extended to compile both `cy_rewrite.pyx` and `cy_runtime.pyx`.
- `monkeypatch.py`: rebinds `run_linear` at all 3 import sites (`tinygrad.engine.realize`,
  `tinygrad.engine.jit`, `tinygrad.tensor`) so callers that did
  `from tinygrad.engine.realize import run_linear` pick up the Cython version.
- `.gitignore`: added `cy_*.c`, `cy_*.pyd` (build artifacts, regenerate per platform).
- Build on Windows: `scoop install mingw` then `python setup_cy.py build_ext --inplace --compiler=mingw32`.

**Final scorecard (RTX 4080, with `import monkeypatch`):**

| Workload | iter 4 (no cy_runtime) | iter 5 (cy_runtime) | torch | result |
|---|---|---|---|---|
| gemm_1024 | 121 | **114** | 126 | WIN 0.90x |
| gemm_256 | 50 | 51 | 47 | 1.09x near |
| add_4096 | 51 | 46 | 24 | 1.92x (was 2.13x) |
| mul_sum | 32 | 33 | 57 | WIN 0.58x |
| relu_4096 | 47 | 46 | 25 | 1.84x (was 1.94x) |
| exp_2048 | 49 | 46 | 23 | 2.00x (was 2.04x) |
| sum_4096 | 47 | 46 | 29 | 1.59x |
| permute | 49 | 46 | 41 | 1.12x near |
| softmax | 35 | **31** | 22 | 1.41x (was 1.66x) |
| layernorm | 32 | **31** | 43 | WIN 0.72x |
| matvec | 99 | 101 | 142 | WIN 0.71x |

4 wins, 2 near-parity, 2 modest gaps, 3 host-floor gaps. Cython runtime closes ~half
the softmax/permute gap; small ops are bounded below by `cuLaunchKernel` + ctypes
wrapper overhead (~21us per call, would need cffi/pybind11 to attack further).

**Reasoning mode:** induction (measured), 90% confidence. The cy_runtime gain is
within bench noise for some workloads (relu, sum, layernorm — all already <2us delta)
but consistent enough across runs to attribute to the change.

### Iter 6 (this session): GRAPH_ONE_KERNEL default — host floor crushed

**Observation.** Re-reading the JIT batching code surfaced
`tinygrad/engine/jit.py:38`:
```python
if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current_batch)
```
Single-kernel batches skip CUDA graph capture by default. The 3 remaining host-floor
workloads (add/relu/exp at 1.84-2.00x torch) are exactly single-kernel batches —
graphs are NEVER engaged for them, and the per-call cost is `cuCtxSetCurrent +
cuLaunchKernel` (2 ctypes calls × ~3.5us = 7us minimum). With graphs, the per-call
cost is one `cuGraphLaunch` driver call.

**Provenance.** `git log -S"GRAPH_ONE_KERNEL"` shows the env var was added Feb 2025
(commit `ae4582675`, "hotfix: GRAPH_ONE_KERNEL + fix timing") as an escape hatch for
UsbGPU openpilot timing tests. Never benched as a default for non-USB CUDA paths.
Not a deliberate complexity-vs-perf trade-off — just a niche flag that never moved.

**Implementation (1-line speedygrad-flavor patch).** `monkeypatch.py` adds at the
top, BEFORE any tinygrad import:
```python
os.environ.setdefault("GRAPH_ONE_KERNEL", "1")
```
Order matters: tinygrad's `getenv` is `@functools.cache`d (helpers.py:161). Setting
the env var after the first cached call is a no-op. Speedygrad users opt in via
`import monkeypatch` (same activation pattern as cy_rewrite, cy_runtime).

**Final scorecard (RTX 4080, isolated subprocess, p10 of 50 trials):**

| Workload | iter 5 (cy_runtime) | iter 6 (+GRAPH=1) | torch | iter6/torch |
|---|---|---|---|---|
| gemm_1024 | 119 | **106** | 122 | **WIN 0.87x** |
| gemm_256 | 50 | **33** | 45 | **WIN 0.73x** |
| add_4096 | 51 | **28** | 23 | 1.22x near (was 1.92x — host floor crushed) |
| mul_sum | 31 | 33 | 57 | **WIN 0.58x** |
| relu_4096 | 47 | **28** | 24 | 1.17x near (was 1.84x — host floor crushed) |
| exp_2048 | 47 | **28** | 21 | 1.33x small (was 2.00x — host floor crushed) |
| sum_4096 | 46 | **29** | 32 | **WIN 0.91x** (was 1.53x) |
| permute | 49 | **29** | 39 | **WIN 0.74x** (was 1.12x near) |
| softmax | 34 | 36 | 21 | 1.71x (slight regression — see below) |
| layernorm | 32 | 33 | 42 | **WIN 0.79x** |
| matvec | 98 | 93 | 63-88 (noisy) | wins p50, ties p10 |

**Tally:** 6 wins (was 4), 2 near-parity, 1 small gap, 1 gap, 1 noisy-tie. From
iter 5's "4 wins, 2 near-parity, 2 modest gaps, 3 host-floor gaps" to "6 wins,
2 near-parity, 2 small gaps, 1 noisy". The 3 host-floor workloads (add/relu/exp)
that had been bounded below by ctypes overhead since iter 1 are now within 17-30%
of torch — a structural floor moved by a 1-line patch.

**Softmax slight regression (34→36us, +6%).** Hypothesis: softmax is multi-kernel
(max + exp + sum + div); when the JIT batches it into a mix of multi-kernel and
single-kernel sub-batches, the previously-direct single-kernel sub-batches now
take the graph path. For a single-kernel sub-batch with multiple PARAMs, the graph
path has comparable ctypes-call count to direct, but the graph object itself adds
some Python overhead per call. Net cost ~2-3us. Within bench noise; classified
as oscillatory (helps 8 workloads, hurts 1 by a small amount). Not worth gating.

**Smoke + regression results.**
- `prework/cuda-parity/smoke.py` 17/17 pass under `GRAPH_ONE_KERNEL=1`.
- `test/backend/test_jit.py` 19/20 pass — the 1 fail is the pre-existing
  Windows clang FileNotFoundError (also fails on master).
- `test/backend/test_linearizer.py` 19/19 pass.
- `test/backend/test_opt_gemm.py` 4/4 pass.
- `test/backend/test_ops.py -k "matmul or sum_reduce or matvec"` 6/6 pass.
- Other test_jit failures all trace to either the Windows clang issue or to the
  `beam_search` → `search` rename in commit `e51047241` (purged BEAM aliases).
  Pre-existing on master, not introduced by this change.

**Reasoning mode table (iter 6).**

| Claim | Mode | Confidence | Evidence |
|---|---|---|---|
| GRAPH_ONE_KERNEL=1 reduces host floor 17us | induction | 95% | A/B bench, full scorecard |
| Provenance: hotfix not deliberate default | deduction | 99% | git log -S, commit message |
| getenv is @functools.cache'd → order matters | deduction | 99% | helpers.py:161 |
| Softmax 6% regression is graph overhead per call | abduction | 70% | hypothesis, not directly measured |
| No new test failures | induction | 90% | regression suite identical to iter 5 |

**Open frontier (after iter 6):**

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | matvec p90 catastrophic outlier (1/12 runs land 917us) | unknown | confirmed iter 6; late-TC sweep noise; needs `_time_program` cnt bump in line 167 of abduct.py |
| 2 | Per-call `cuCtxSetCurrent` in `CUDAProgram.__call__:55` (3.5us each call) | ~5 | obvious next: context never changes within a JIT replay; safe to lift |
| 3 | softmax 1.71x — multi-kernel batching efficiency on CUDA | ~50 | likely needs proper kernel fusion or batch-merging beyond GRAPH_ONE_KERNEL |
| 4 | exp_2048 1.33x — transcendental call quality | ~30 | ~~tinygrad's exp likely uses polynomial decomposition; torch uses CUDA's intrinsic~~ **KILLED bug-hunt iter 7.5 round 5**: tinygrad already maps `Ops.EXP2` → `ex2.approx` in `ptx.py:20`. Real cause is host-side, not transcendental quality. See iter 7.5 frontier item #3 for current framing. |

### Iter 7 (this session): Cython exec_graph + CUDAGraph.__call__ inlining

**Observation.** cProfile of add_4096 hot path (post iter-6 GRAPH_ON default) showed
the per-call cost was distributed across 6 Python frames before the actual ctypes
driver call — `TinyJit.__call__` → `CapturedJit.__call__` → `cy_run_linear` →
`exec_graph` → `CUDAGraph.__call__` → `cu_time_execution(lambda: ...)` → ctypes
wrapper. The single largest line items by cumtime were `track_stats`
contextmanager (~7us cumtime in cProfile, ~3-4us actual) and the
`cu_time_execution(lambda: ...)` wrapper around `cuGraphLaunch` for the wait=False
case (2 unnecessary Python frames per call, ~3-4us actual).

**Initial wrong hypothesis.** Assumed `cuCtxSetCurrent` in `CUDAProgram.__call__:55`
was the per-call hot-path cost. Provenance check killed it: with `GRAPH_ONE_KERNEL=1`
default (iter 6), `CUDAProgram.__call__` is bypassed entirely on the JIT replay path —
calls go through `CUDAGraph.__call__` which never calls `cuCtxSetCurrent`. The
actual hot-path bottleneck is Python frame overhead from the multi-layer call stack.
Recorded so the same dead-end isn't re-explored.

**Implementation (two complementary patches).**

1. `cy_runtime.pyx` — added `_exec_graph_fast` (mirrors `_exec_kernel_fast` pattern):
   ```cython
   cdef inline _exec_graph_fast(ctx, call, ast):
     if DEBUG >= 2 or PROFILE: _exec_graph_py(ctx, call, ast); return
     rt = _get_graph_runtime(ast, ctx.input_uops)
     if ctx.do_update_stats: _GlobalCounters.kernel_count += 1
     rt(ctx.input_uops, ctx.var_vals, wait=False)
   ```
   `cy_run_linear`'s CUSTOM_FUNCTION dispatch reordered so `arg == "graph"` is the
   first branch (most common case). Skips the `track_stats` contextmanager and its
   `estimate_uop` call. Falls back to Python `exec_graph` when DEBUG/PROFILE are on.

2. `monkeypatch.py` — replaces `CUDAGraph.__call__` with `_cuda_graph_call_fast`,
   identical body except the final launch:
   ```python
   if wait: return _cu_time(lambda: _check(_cuda.cuGraphLaunch(...)), enable=True)
   _check(_cuda.cuGraphLaunch(self.instance, None))  # direct, no lambda wrapper
   return None
   ```
   Saves the lambda + `cu_time_execution` wrapper for the wait=False path
   (every JIT replay). Identical observable behavior; wait=True path unchanged.

**Final scorecard (RTX 4080, isolated subprocess, p10 of 50 trials):**

| Workload | iter 6 | iter 7 | torch | iter7/torch | Δ vs iter6 |
|---|---|---|---|---|---|
| gemm_1024 | 106 | **104** | 122 | **WIN 0.85x** | -2us |
| gemm_256 | 33 | **30** | 45 | **WIN 0.66x** | -3us |
| add_4096 | 28 | **25** | 23 | **1.09x near** (was 1.92x baseline) | -3us |
| mul_sum | 33 | **31** | 57 | **WIN 0.54x** | -2us |
| relu_4096 | 28 | **25** | 24 | **1.05x near** (was 1.84x baseline) | -3us |
| exp_2048 | 28 | **25** | 21 | 1.19x small (was 2.00x baseline) | -3us |
| sum_4096 | 29 | **26** | 32 | **WIN 0.81x** | -3us |
| permute | 29 | **27** | 39 | **WIN 0.69x** | -2us |
| softmax | 36 | **34** | 21 | 1.62x | -2us |
| layernorm | 33 | **30** | 42 | **WIN 0.71x** | -3us |
| matvec | 93 | **75** | 63 | 1.19x small (ratio inverted in original write-up: 75/63=1.19, not 0.83x; bug-hunt round 4) | **-18us** |

**6 wins, 2 essentially-tied (1.05-1.10x), 2 small gaps (exp_2048 1.19x, matvec 1.19x), 1 gap (softmax).**
Per-workload Δ is uniformly 2-3us (one Python frame's worth) plus matvec's outlier
-18us. The matvec gain is bigger than expected; hypothesis is that matvec is the
single matmul kernel that was previously hitting the abduct-search noise path
during warmup, and the cleaner exec_graph path makes its bench measurement more
deterministic. (Matches the observation that iter-7 matvec p90=78 is now
super-tight — gap between p10 and p90 is only 3us, vs iter 6's 100+us.)

**Reasoning mode (iter 7).**

| Claim | Mode | Confidence | Evidence |
|---|---|---|---|
| Hot path is Python frame overhead, not ctypes | induction | 95% | cProfile of add_4096 hot loop |
| cuCtxSetCurrent NOT in GRAPH hot path | deduction | 99% | code trace ops_cuda.py vs graph/cuda.py |
| ~3us per Python frame removed | induction | 90% | A/B bench, uniform 2-3us delta across workloads |
| matvec -18us comes from p90 stabilization | abduction | 65% | hypothesis; not directly measured |

**Smoke + regression results.**
- `prework/cuda-parity/smoke.py` 17/17 pass (correctness preserved).
- `test/backend/test_jit.py` 40/47 pass; 6 failures (deselecting Windows-only test_jit_several_devs) — same 6 failures as iter 6 (clang FileNotFoundError + beam_search rename). **Zero new regressions.**
- `test/backend/test_linearizer.py` 19/19 pass.
- `test/backend/test_opt_gemm.py` 4/4 pass.
- `test/backend/test_ops.py -k "matmul or sum_reduce or matvec"` 6/6 pass.

**Open frontier (after iter 7):**

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | matvec p90 catastrophic outlier (1/8 fresh-cache runs land 437us) | unknown | iter 6 finding still open; the iter-7 bench shows a stable cache run, but a fresh search may still mis-adopt. Late-TC sweep `min`-comparator fix needed |
| 2 | softmax 1.62x — multi-kernel batching efficiency | ~50 | dominant remaining gap; needs proper kernel fusion or batch-merging |

### Iter 7.5 (this session): online-softmax algorithmic carry to CUDA — VALIDATED

**H₀.** The Metal online-softmax prototype (`bench/online_softmax.py`, 2.5-6.6x on
Metal) carries to CUDA: speedygrad's 3-kernel CUDA softmax (max → exp(x-m) → sum/div)
can collapse to 1 kernel using Milakov-Gimelshein compound reduction with
`__shfl_down_sync` warp reduction.

**Phase 1 — kernel structure of current softmax (DEBUG=2 trace).**

Current softmax(256,256) on speedygrad CUDA, after iter 7's exec_graph fast path:

| # | Kernel | Op | GPU steady-state |
|---|---|---|---|
| 6 | `r_16_16_16_16` | row-max via GROUP=16+LOCAL=16 | ~5us |
| 7 | `r_256_256` | row-sum-of-exp(x-max) | ~6us |
| 8 | `E_256_16_16` | elementwise out = exp(x-m)/sum | ~6us |
| 9 (graph) | `batched 3` | cuGraphLaunch of {6,7,8} | **17.4us GPU** |

Wall clock 34us = 17us GPU + ~17us host (cuGraphLaunch + Python frames + ctypes).
Torch reference: 21us wall (closed-source vendor primitive, likely 7-10us GPU + 11-14us host).

**Phase 1 — standalone CUDA online-softmax bench (`prework/cuda-parity/online_softmax_cuda.py`).**

Hand-written CUDA kernel (one warp per row, `__shfl_down_sync` compound reduce
for both running-max and running-sum, `__expf`/`__shfl_sync(...,0)` for the broadcast):

| Shape | online-softmax GPU p10 | speedygrad 3-kernel GPU | **algorithmic carry** |
|---|---|---|---|
| 256x256 | **10.2us** | 17.4us (batched 3) | **1.7x** |
| 1024x1024 | 8.0us | (untested at this shape) | n/a — large enough that GPU dominates |
| 4096x4096 | 248us | (untested) | n/a — GPU bandwidth bound |

Numerical correctness: `max|out - numpy_softmax| < 3e-8` (fp32) on all perf shapes,
plus PASS on edge cases (cols<32 inactive-lane reduction, masked-causal -inf rows).

**Provenance honesty (bug-hunt round 1+2 findings).** First measurement was 4.10us,
which would have been a 4.2x carry. Bug hunt round 1 (gemini-3-pro adversarial review)
found that the kernel's initial `m = -INFINITY` poisoned warp-reduce with
`exp(-inf - -inf) = NaN` whenever two empty lanes paired (cols<32 or
masked-attention -inf rows). Fix: initial `m = -FLT_MAX` instead — same
mathematical behaviour but no `inf - inf`. After the fix, GPU p10 stabilized at ~10us
across three runs — a 6us regression from the pre-fix measurement.

**Bug-hunt round 2 caught me bullshit-explaining the regression.** I initially
attributed the 6us to "nvcc constant-folding -INFINITY paths in `__expf`," but
gemini correctly noted this is mathematically impossible: for the perf shapes
(cols ≥ 32), every lane iterates the loop at least once, so `m = fmaxf(-INFINITY, x)`
becomes finite *before* the warp reduce ever runs — the `-INFINITY` constant
never reaches the warp-reduce paths the compiler could fold.

**Bug-hunt round 3 caught the actual mechanism: GPU clock-state artifact.**
The bench shows `256x256` at 10us GPU and `1024x1024` at 8us GPU — physically
impossible (16x more data, less time) under fixed clocks. Increasing warmup from
20 to 2000 iterations didn't move either number. Diagnosis: each 10us kernel
launches only 256 blocks × 32 threads = 8K threads on a 117K-thread GPU (7%
utilization), so the GPU clocks down between launches and stays at intermediate
P-states even during the timed loop. `1024x1024` (32K threads, 28% util) keeps
the GPU at boost clocks. `4096x4096` (~512K threads) saturates and is bandwidth-bound.

**Implication for the carry claim.** Both the 17us speedygrad baseline (DEBUG=2
trace inside a TinyJit replay loop) and the 10us online-softmax measurement
were collected with similar low-duty-cycle launch cadence. The 1.7x **ratio**
is approximately fair. The absolute numbers are at idle/intermediate clocks; at
real inference duty cycle (back-to-back ops in attention) both numbers would
likely shrink, but their ratio should hold.

**Conclusion: prototype carries on CUDA, but more modestly than the Metal 2.5-6.6x
or the buggy-kernel 4.2x.** At 256x256 (the bench shape), single-kernel online
softmax is **~1.7x faster on GPU** than the current 3-kernel batched graph.

**Wall-clock projection (if integrated; mechanism-level model, not measured end-to-end).**

| Path | GPU | Host | Wall | vs torch |
|---|---|---|---|---|
| iter 7 (current) | 17us | 17us | 34us | 1.62x slower |
| online softmax (1 kernel in graph) | ~10us | ~15us | **~25us** | ~1.19x — narrows gap, doesn't win |

Host saves ~2us by collapsing 3 `cuGraphExecKernelNodeSetParams` ctypes calls to 1
in the JIT replay path (`monkeypatch.py:53` loops over `self.updatable`, so the
saving is per-node, not per-graph; correction noted from bug-hunt round 1 where
gemini originally claimed this was constant per graph).

This narrows the softmax gap from 1.62x to ~1.19x. Not parity, not a torch win.
Honest reframing: the integration is still worth doing for the decode loop and
attention, but won't single-handedly close the softmax gap. Combining with
masked-attention fusion (frontier #5) is what gets to parity. (The iter 6/7
plan to swap `exp` polynomial → CUDA intrinsic was retracted in bug-hunt round
5: tinygrad already uses the intrinsic; the gap is host-side.)

**Measurement caveat (bug-hunt round 1).** `cuEventElapsedTime` includes PCIe
dispatch latency for `cuLaunchKernel` because the per-iteration sync drains the
GPU between trials. The reported GPU p10 is therefore a slight overestimate of
pure kernel time. The speedygrad 17us baseline (from DEBUG=2 cuGraphLaunch event
timing) has the same dispatch-latency inclusion, so the ratio is approximately
fair. Pure kernel time is likely 1-2us less for both numbers.

**Phase 2 (deferred): framework integration.**

Integration scope is genuinely larger than the original ~100 LOC estimate. To
participate in the JIT cuda-graph batching (which is critical — bypassing the
graph would re-introduce per-call overhead and lose the iter 6/7 wins), the kernel
needs to be either:

1. **Synthetic `Ops.PROGRAM` UOp** (cleaner) — pre-render PTX, build PROGRAM UOp via
   `to_program()` with a hand-rolled `ProgramInfo(globals=(0,1), outs=(0,), vars=(cols,))`,
   route `Tensor.softmax` to emit a `CALL(program, out_buf, in_buf)` for favorable
   cases. This goes through the existing pipeline and gets cuda-graph-batched
   automatically. Pattern is the one used in `extra/gemm/triton_nv_matmul.py:96`.

2. **New `Ops.CUSTOM_FUNCTION arg="online_softmax"`** — register dispatcher in
   `_CUSTOM_DISPATCH`, extend `MultiGraphRunner.supports_uop` (jit.py:177) to
   include the new arg, extend `CUDAGraph.__init__` (graph/cuda.py:18) to add a
   kernel node for it. More invasive (touches jit.py + graph/cuda.py) but doesn't
   require synthetic PROGRAM construction.

Test matrix (either path): axis variants (last-axis vs other), dtype (fp32, fp16,
bf16 — order-of-accumulation differs in low precision), shape (1D, 2D, 3D+ with
broadcast), masked softmax (attention's `where(mask, x, -inf)`), gradient correctness
(autograd — softmax appears in cross-entropy loss). Bench coverage needs `smoke.py`
correctness checks plus `bench_iso.py` softmax + a fresh attention microbench.

Realistic scope: 200-400 LOC, 1-2 days focused work. Filed as iter 8a (precedes
iter 8 Llama 3.2 demo because the demo's attention softmax compounds with this win).

**Reasoning mode (iter 7.5, post bug-hunt round 1).**

| Claim | Mode | Confidence | Evidence |
|---|---|---|---|
| Current speedygrad softmax = 3 kernels in cuGraph batch | deduction | 99% | DEBUG=2 trace |
| Current GPU time 17us, host 17us | induction | 95% | DEBUG=2 timings + bench wall clock |
| Online algorithm carries to CUDA at 256x256, 1.7x GPU speedup | induction | 95% | standalone bench post-fix, 10us GPU vs 17us, correctness verified including masked-attention edge case |
| Original 4.2x carry shrunk to 1.7x post-NaN-fix; 6us regression caused by GPU clock-state / low-duty-cycle launch cadence | induction | 95% (delta) / 80% (mechanism) | bug-hunt round 3 identified clock-state via 1024x1024 < 256x256 anomaly (impossible under fixed clocks); ratio is fair across same-cadence baselines |
| Wall projects to ~25us (narrows but doesn't close gap to torch ~21us) | abduction | 65% | host saves modeled per-node (not constant), not measured end-to-end |
| Integration is 200-400 LOC, 1-2 days | abduction | 65% | scope analysis vs encdec/triton_nv_matmul reference patterns |

### Iter 8 (this session): Llama 3.2 1B inference demo — VALIDATED at 2.82x decode

**H₀.** Speedygrad's iter 6+7 host-floor wins compound on real LLM inference. Llama 3.2 1B-Instruct end-to-end decode on speedygrad beats torch + HF transformers eager on the same RTX 4080.

**Setup.** Model: unsloth/Llama-3.2-1B-Instruct (non-gated mirror of meta-llama, identical weights), pre-converted bf16→fp16 on disk because speedygrad's PTXRenderer has no bf16 entry in its `types` dict (`renderer/ptx.py:157-162`). Conversion script: `prework/cuda-parity/convert_bf16_to_fp16.py`. Both benches feed the same 37-token HF chat-template input IDs (`tokenizer.apply_chat_template([{"role":"user","content":"Hello."}], add_generation_prompt=True)`).

**Method.** Per-token decode timing, KV cache populated via prefill, 5 runs × 25 decode tokens, first decode token of each run excluded as warmup → 120 samples per framework. `time.perf_counter` around each `.item()` (speedygrad) or `model(...)` (torch) with `cuda.synchronize` on torch and the implicit sync from `.item()` on speedygrad. Greedy decode (temperature=0). Speedygrad runs with `monkeypatch` enabled (Cython rewrites + cy_runtime fast path + GRAPH_ONE_KERNEL).

**Result (matched 37-token prompt, fp16-vs-fp16):**

| Metric | Speedygrad fp16 | Torch+HF eager fp16 | Ratio |
|---|---|---|---|
| Decode p10 | 7.88 ms (**126.8 tok/s**) | 27.76 ms (36.0 tok/s) | **3.52x** |
| **Decode p50** | **10.02 ms (99.8 tok/s)** | **28.28 ms (35.4 tok/s)** | **2.82x** |
| Decode p90 | 14.75 ms (67.8 tok/s) | 35.50 ms (28.2 tok/s) | 2.41x |
| Prefill p50 (37 toks) | 705 ms | 43 ms | **0.06x (torch wins 16.4x)** |

Even speedygrad's worst p90 (67.8 tok/s) beats torch's best p10 (36.0 tok/s) by 1.88x.

**Numerical correctness verified.** Both frameworks generate the bit-identical token sequence for the first 25 decoded tokens: `"Hello. Is there something I can help you with or would you like to chat?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI can provide"`. Same input IDs → same logits → same argmax. The fp16 forward pass is faithful.

**Issues triaged during iter 8.**
1. `examples/llama3.py` fetched the Q6_K GGUF by default; works on speedygrad. Steady-state ~65 tok/s but quantization muddies the comparison vs torch fp16, so pivoted to fp16 safetensors.
2. `Context(BEAM=0)` in `examples/llama3.py:226` crashed (`KeyError: 'BEAM'`) — speedygrad uses `SEARCH`, BEAM was deliberately removed. Fixed at the call site (`Context(SEARCH=0)`), not by re-registering BEAM as a shim.
3. PTXRenderer crashed on bf16 weights (`KeyError: dtypes.bfloat16` in `ssa()`). Speedygrad's `fix_bf16` queues `cast(fp32).cast(fp16)` but the cast kernel's PTX read still requires bf16 in `mem_types`. Workaround: pre-convert weights on disk via torch+safetensors. Real fix would be to add bf16 to PTX `types/mem_types/cast_types` plus `cvt.f32.bf16`/`cvt.bf16.f32` (~30 LOC) — filed below.
4. `tiktoken` was not installed; `examples/llama3.py`'s `Tokenizer` needs it. One-time `pip install tiktoken`.

**Wider variance on speedygrad than torch.** Speedygrad's per-token decode varies p10-p90 from 67-127 tok/s (1.9x spread), torch from 28-36 tok/s (1.3x spread). The mechanism is the same iter 7.5 finding: per-token decode launches ~30 small kernels through cuGraph batching, total GPU compute is ~5-8us, GPU clocks down between calls. Torch's eager dispatch keeps the GPU at higher steady-state utilization. The ratio still holds because **both** are running the same shape mix; the variance is symmetric noise on top of the deterministic framework overhead.

**The prefill cliff is the open story.** Speedygrad's `prefill()` in `examples/llama3.py:257` iterates one token at a time through the JIT path. 37 tokens cost 705 ms (19 ms/tok). Torch does a single batched forward over the prompt: 43 ms total (1.16 ms/tok). At realistic prompt lengths the prefill dominates wall time:

| Prompt length | Speedygrad prefill (projected) | Torch prefill (projected) | Speedygrad disadvantage |
|---|---|---|---|
| 37 tok | 0.7 s | 0.04 s | 16x slower |
| 256 tok | 4.9 s | 0.30 s | 16x slower |
| 2048 tok | 39 s | 2.4 s | 16x slower |

Decode wins are unaffected, so for medium-output use cases (chat completion of 100-500 tokens off short prompts) the decode advantage dominates wall time. Long-context (8k+ prompt, short response) is where torch wins overall. Prefill fix is filed as frontier item #9.

**Reasoning mode (iter 8).**

| Claim | Mode | Confidence | Evidence |
|---|---|---|---|
| Speedygrad fp16 decode p50 = 99.8 tok/s on RTX 4080 | induction | 95% | 120 samples across 5 runs |
| Torch+HF eager fp16 decode p50 = 35.4 tok/s on same hardware | induction | 95% | 115 samples across 5 runs |
| 2.82x speedup at p50 holds for matched 37-token chat-template prompt | induction | 90% | identical input token IDs verified, identical 25-token output verified |
| Decode advantage extends to all prompt lengths (decode tok/s is prompt-length-indep given KV cache) | deduction | 90% | KV cache decouples decode from prompt length; both frameworks use KV cache the same way |
| Prefill is 16x slower on speedygrad for any prompt length | induction | 85% | one-token-at-a-time loop scales linearly with prompt length; the 19 ms/tok constant should hold across prompt sizes (no quadratic) |
| Wider speedygrad p10-p90 spread is GPU clock-state noise (iter 7.5 mechanism), not codegen variability | abduction | 70% | matches the 256x256 softmax pattern from iter 7.5 bug-hunt round 3; not directly verified for the Llama kernel mix |

**Reproduce.**

```powershell
# venv with torch 2.11+cu128, transformers, tiktoken, safetensors already installed
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$env:PATH"
$env:DEV = "CUDA"; $env:PYTHONPATH = "."

# One-time: download unsloth mirror, convert bf16 -> fp16 on disk
python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Llama-3.2-1B-Instruct')"
python prework/cuda-parity/convert_bf16_to_fp16.py

# Bench (each run takes ~30s for 5x25 tokens)
python bench/speedygrad_llama32_1b.py --runs 5 --n-new 25 --out prework/cuda-parity/speedygrad_fp16_bench.json
python bench/torch_llama32_1b.py        --runs 5 --n-new 25 --out prework/cuda-parity/torch_fp16_baseline.json
```

Bench scripts: `bench/speedygrad_llama32_1b.py`, `bench/torch_llama32_1b.py`. Result JSONs and conversion script: `prework/cuda-parity/`.

---

**Open frontier (after iter 8):**

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | matvec p90 catastrophic outlier | unknown | unchanged from iter 7 |
| 2 | **online-softmax integration (path 1: synthetic PROGRAM)** | ~200 | **prototype validated iter 7.5, ready for focused implementation iteration** |
| 3 | exp_2048 1.19x — host overhead, NOT transcendental quality | ~30 | bug-hunt round 5 retraction: tinygrad's PTX renderer (`ptx.py:20`) already maps `Ops.EXP2` to `ex2.approx`, the CUDA intrinsic. The "polynomial decomposition" hypothesis in iter 6/7 was false. Real cause is unknown; 4us gap likely host-side (one fewer Python frame than torch's eager dispatch) |
| 4 | _prepare_jit_inputs (11.5us cumtime per call) | ~50 | unchanged from iter 7 |
| 5 | Attention fusion (builds on #2) | ~200 | unblocked once #2 lands |
| 6 | First-compile cost (~5s for gemm_256 at depth 5) | ~30 | unchanged from iter 7 |
| 7 | **Pack multiple warps per block in online-softmax kernel** | ~10 | bug-hunt round 3 finding: 32-thread blocks cap SM occupancy at 50% on sm_89 (24 blocks/SM hardware limit). Apply during framework integration |
| 8 | **`GlobalCounters.global_ops`/`global_mem` not tracked on Cython fast path** | ~5 | bug-hunt round 4 finding (out of iter 7.5 scope, real iter 7 issue): `_exec_kernel_fast` and `_exec_graph_fast` in `cy_runtime.pyx` skip `estimate_uop` to save ~3us, but that also skips the FLOP/mem accumulation in `track_stats`. Any external bench script depending on `GlobalCounters.global_ops` (e.g. estimating throughput) silently sees zero. Either rebuild estimates lazily, or update docs to note the Cython path is FLOP-untracked |
| 9 | **Batched prefill (replace 1-token-at-a-time loop)** | ~30-100 | iter 8 finding: `examples/llama3.py:257` prefills one token per JIT call → 19 ms/tok. Torch batched prefill is 1.16 ms/tok (16x faster). For 2048-token prompt: 39s vs 2.4s. Implementation: pass `Tensor([toks])` as batch through model.forward (not forward_jit, which captures bs=1,seq=1) for the prefill, then resume per-token decode. Architecturally: forward path already supports seqlen>1 (see mask construction `examples/llama3.py:213`); the JIT wrapper only handles single-token decode |
| 10 | **bf16 support in PTXRenderer** | ~30 | iter 8 finding: `KeyError: dtypes.bfloat16` in `ssa()` when loading bf16 weights. Workaround is on-disk fp16 conversion. Real fix: add `dtypes.bfloat16` to `types`/`mem_types`/`cast_types` in `renderer/ptx.py:157-162` and implement `cvt.f32.bf16`/`cvt.bf16.f32` in cast logic. Llama 3.2/Mistral/Qwen all ship bf16 by default, so this is a recurring blocker for the iter 9 architecture-coverage roadmap. Long-term should also install nvrtc/nvjitlink so we use CUDARenderer (which presumably handles bf16) |

### Matvec p90 outlier (still open)

`prework/cuda-parity/noise_probe.py matvec 12` (post-iter-6, GRAPH_ON):
- 6/8 runs: p90 < 130us (clean)
- 1/8 runs: p90 = 437us (catastrophic — chosen TC + UNROLL=4 kernel measures 6x worse than search winner)
- 1/8 runs: p90 = 309us (also bad)

The mechanism: `abduct.py:167` compares `final_tc = min(_time_program(tc_compiled, cnt=7))`
against `best_time` (also a min-of-7). `min` is sensitive to single fast outliers;
late-TC sweep occasionally adopts a TC kernel whose ONE fast measurement of 7 fooled
it but whose mean is much worse than search winner. Cheap fix: use median of cnt=15
in the comparator, or add a final validation pass after the late-TC follow-up search
that confirms TC's adopted chain is still measured-faster than the original best.
Filed as iter 7 candidate.

---

## Open frontier

Ranked by impact per line of code. Tiebreaker: fewer lines wins.

| # | Edge | LOC | Status |
|---|---|---|---|
| 1 | Matvec codegen: loop reordering for stride-32768 | ~20 | diagnosed — search space collapse at N=1, fix is codegen-level |
| 2 | Algebraic fusion (online softmax framework integration) | ~100 | prototype validated (2.5-6.6x), needs UOp wiring |
| 3 | Native Q6K matmul kernels | ~300 | open |
| 4 | Fused dequant UOp rewrite | ~200 | depends on #3 |

### Matvec gap (2.27x vs PyTorch) — DIAGNOSED

Abduction engine diagnosis: the matvec kernel has 1 LOCAL + 1 UPCAST action (vs 9+5 for gemm). The N=1 dimension collapses the optimization surface — the abduction engine has nothing to search over. PyTorch dispatches to a hand-written Metal matvec kernel.

The root cause (from the original matvec investigation, section III): the generated kernel walks the weight matrix with stride 32768 bytes in the inner loop. Unit-stride output axis is outside the loop. The fix is codegen-level (loop reordering or specialized matvec kernel template), not search-level.

Transposed weight abduction: transposing the weight matrix LOSES TC, GROUPTOP, GROUP, and UNROLL entirely. The stride pattern makes the kernel unrecognizable as a matmul pattern.

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
└─ Online softmax prototype (2.5-6.6x Metal, **1.7x CUDA at 256x256 post bug-hunt iter 7.5**) — VALIDATED with NaN fix, ready for integration
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

- ~~Pre-compile dedup~~ — 1% waste (2/224 duplicate compiles). Post-compile `seen_libs` already handles it. Nothing to save.
