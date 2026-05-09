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

### H5: Hybrid architecture — PROPOSED

The heuristic is a two-layer system: structural priors (pattern matching on the AST) and parameterized transformations (upcast amounts, local sizes, thresholds). The abduction engine should:

1. Keep the structural priors (TC eligibility, kernel class detection, stride analysis)
2. Replace the parameters with measurement
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

### Warp-reduce for GROUPTOP — CONFIRMED

Replace scalar shared-memory reduction with simd_sum. 2.1-4.2x on the reduction step. HIP disabled pending AMD testing.

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

### Cython transpile of unified_rewrite (H27) — CONFIRMED (-7.3% e2e)

First end-to-end signal in the entire investigation. Transpiled `unified_rewrite` to C via Cython (95 lines, zero algorithmic changes). 22.98ms → 21.30ms.

7.3% of tinygrad's compilation cost is pure CPython bytecode dispatch overhead in one function. Not shippable to tinygrad (Python-only project). Quantifies the CPython JIT opportunity.

### Killed hypotheses

| Hypothesis | Why it died |
|---|---|
| backward_slice O(n²) | Double caching makes it structurally impossible |
| Bloom filter gate | Cascade failure — 98% skip rate but 2% misses break correctness |
| Decision tree | Python per-call overhead neutralizes iteration savings |
| Huffman if-elif tree | CPython `dict.get` is O(1), if-elif is O(n) in bytecode interpreter |
| Per-op compiled functions | Frame creation (~100-150ns) exceeds loop elimination savings |
| Nested pdict (op → src[0].op) | 73% wildcard fallback, 2 dict.gets for same result |
| Bitmask early-reject | `frozenset.issubset` is already a C builtin |
| Redundant len(src) check | UOps don't enforce arity; intermediates have wrong src count |
| RETE leaf skip | Leaf nodes are 6% of graph; Python overhead per visit dominates |
| Skip 0-pattern ops | Already cheap (~50ns each), saves 0.23ms on 23ms |

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

## PTX/CUDA renderer fallback — CONFIRMED

`is_dtype_supported` checks against the base renderer class, not the resolved renderer. PTXRenderer silently replaces CUDARenderer when NVRTC is missing.

*PR: #16108*

---

## Open frontier

### Kernel quality
1. Theory transfer to non-matmul classes (reductions, elementwise, convolutions)
2. Joint GROUP+LOCAL+UPCAST optimization for matvec gap (~20 lines for 2-deep mini-beam)
3. Amortized cost measurement (52 trials vs BEAM's 200+)
4. Theory transfer on CUDA

### LLM inference
5. Native Q6K matmul kernels — close the 2.6x gap without 2x memory penalty
6. Matvec loop ordering fix at the scheduler level
7. chunk_size sensitivity for prefill (32 vs 128 vs 256)

### Codegen
8. CPL + LUC + APRP scheduling (~80-120 lines)
9. Algebraic fusion (Flashlight for softmax, RedFuser for layernorm)
10. Fused dequant UOp rewrite rule

### Infrastructure
11. CPython JIT for branchy dict/deque loops (contribution to CPython, not tinygrad)

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
│   ├─ theory transfer — CONFIRMED (matmul class)
│   │   └─ non-matmul transfer — OPEN
│   ├─ abduction loop (52 trials, 1.85x) — CONFIRMED
│   │   └─ joint optimization for matvec — OPEN
│   ├─ CPL scheduling (matvec +23%) — CONFIRMED
│   │   └─ APRP register pressure ceiling — OPEN
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
│   ├─ Cython transpile (-7.3% e2e) — CONFIRMED (not shippable)
│   └─ CPython JIT improvement — OPEN
│
├─ Warp-reduce for GROUPTOP (2.1-4.2x) — CONFIRMED
├─ Renderer fallback detection — CONFIRMED
└─ Line budget (636 headroom, onnx -29 merged) — CONFIRMED
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
