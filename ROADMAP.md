# Roadmap

Working document. Not a contract. Updated as priorities clarify.

## Two North Stars

Everything below is in service of these. When in doubt, check work against both.

1. **Copyleft or pay up.** AGPL public, commercial license per-buyer. Use copyright as judo
   on copyright. Every adoption is a node in the copyleft commons or a paying enterprise
   customer. No third option (no MIT relicense, no "let proprietary world consume the
   wins for free").
2. **Model serves one master.** Local training and inference on owned hardware, with the
   operator picking the policies. The runtime exists so users can escape vendor RLHF
   compromises, vendor refusals, vendor pricing, vendor surveillance. Performance is
   what makes the local choice *practical* rather than principled-but-painful.

These reinforce each other for almost every decision. They diverge in a few places (see
*Out of scope* below) — when they conflict, name it explicitly.

## Priority order

```
inference  >  fine-tune  >  training  >  dev
```

Tinygrad's implicit ordering is the inverse — `dev > training > fine-tune > inference` —
optimizing for the audience that doesn't pay. Speedygrad inverts because:

- **Inference first**: largest user volume (~100x fine-tune, ~10000x training), lowest
  validation friction, simplest workflow, broadest audience. Decode-phase generation is
  also where iter 6+7's host-floor wins compound the most (small-tensor, single-token,
  per-call overhead × N tokens). Best technical showcase, biggest market wedge.
- **Fine-tune second**: smaller audience but higher per-user value, strongest
  privacy/sovereignty argument, builds on the inference work (same architectures,
  weight loaders, tokenizers).
- **Training third**: tiny audience, highest complexity (full optimizer state,
  distributed, checkpointing). Iter 6+7 wins still apply but proportionally smaller.
- **Dev last**: devs aren't buyers, they're contributors. Dev-friendliness is a
  *byproduct* of small clean code, not a goal.

## The "no PyTorch dependency" thesis

Open weights are files of floats with a known architecture. PyTorch is one possible
runtime — not a technical requirement. Running open weights through PyTorch is *historical
accident*, not necessity. If the weights are open, requiring PyTorch to use them is a
form of capture: open weights locked to a specific framework's bloat is not really open.

Speedygrad's positioning: **demonstrate that open weights can run on a minimal AGPL
framework**. The smallness IS the argument. The competitive claim is not "PyTorch but
faster" — it is "open weights deserve an open framework, here is what running them
looks like with 500 LOC instead of 500K, and it happens to also be faster."

Practical implication: do **not** rebuild HF-equivalents (peft, accelerate, Trainer,
transformers). Those exist because PyTorch needs them; the abstractions aren't intrinsic
to running models. Build the *minimum* needed — model architecture, optimizer, weight
loader, training loop — and demonstrate that's enough.

## Audience model

| Population | Role | Revenue | Action |
|---|---|---|---|
| Tinybox-owners | Validation lighthouse | ~zero | Make adoption frictionless. Their visible benchmarks are the marketing. |
| AI hobbyists / prosumers | Audience | ~zero | Same as above. AGPL is fine for them. |
| Small AI startups | Mixed | Some commercial license sales | Free AGPL until they ship products downstream. |
| Privacy-bound enterprise | Revenue | Real ARR | Dual license. They pay because they can't take AGPL. |
| Sovereign / government | Revenue | Lumpy but large | Long sales cycle. Not near-term focus. |

Tinybox-owners are the **lighthouse, not the cargo**. Their function is making perf
claims credible to populations that pay. Optimize for their adoption friction (install
ease, reproducible benches) — not for extracting revenue from them.

## Versioning

Major version is a forcing function, not a marketing decision. **Each major version is
earned by beating the benches of every relevant toolset at the corresponding stage.**
No competitive bench wins, no major version bump.

| Version | Earned by | Competitive set |
|---|---|---|
| v0.x | (current) building toward stage-1 capability | — |
| v1.0 | Beating inference benches across the stage | PyTorch + HF transformers, vLLM, llama.cpp / Ollama, ExLlamaV2 |
| v1.x | Inference refinement (more architectures, shapes, contexts) | — |
| v2.0 | Beating fine-tune benches across the stage | HF Trainer + peft, Axolotl, Unsloth, LLaMA-Factory |
| v2.x | Fine-tune refinement | — |
| v3.0 | Beating training benches across the stage | PyTorch native, Lightning, Composer |
| v3.x | Training refinement | — |

**"Beating" means**: on at least one realistic workload per competitor, on tinybox-class
hardware, with reproducible bench scripts published. Not a single cherry-picked
microbenchmark. Not "competitive on average." Genuine wins.

**What this prevents**: shipping v1.0 because we're tired of v0.x, or because the
perf is "good enough." A bench-driven version policy means the version number
*carries information* — anyone who sees "speedygrad v2.0" knows the fine-tune stage
is genuinely cleared, because the policy doesn't permit otherwise.

**What this means for the validation play**: each major version is a complete
artifact. v1.0 is "speedygrad now runs your inference workload faster than
[whatever you're running today]." v2.0 is "speedygrad now fine-tunes faster than
your current fine-tune stack." Tinybox-owners can validate the major-version claim
directly because the policy *defines* what the major version means.

**What this means for stages we can't clear**: if we genuinely can't beat llama.cpp
on local inference (which is plausible — their CUDA kernels are hand-tuned and GGUF
is efficient), then v1.0 doesn't ship. We either close that specific gap, or we
revise the competitive set with explicit reasoning ("v1.0 = beating PyTorch HF and
vLLM; llama.cpp is a separate competitor we engage with at vN.0"). The version
policy is allowed to be revised — but only with public reasoning, not silently.

## Iterations

### Iter 8 — Llama 3.2 1B inference demo — DECODE VALIDATED (2.82x), prefill open

**Decode result.** Llama 3.2 1B-Instruct fp16 on RTX 4080:

| | Speedygrad | Torch+HF eager | Ratio |
|---|---|---|---|
| Decode p50 | 99.8 tok/s | 35.4 tok/s | **2.82x** |

Both frameworks fed identical 37-token HF chat-template input IDs, generate bit-identical 25-token output. Full writeup: HYPOTHESIS_GRAPH.md "Iter 8" section. Bench scripts: `bench/speedygrad_llama32_1b.py`, `bench/torch_llama32_1b.py`.

**Open before v1.0:**
- **Prefill is 16x slower** (`examples/llama3.py:257` does one-token-at-a-time prefill). For 2048-tok prompts this dominates wall time. Filed as frontier item #9 (~30-100 LOC). Required before claiming v1.0 against any long-context workload.
- **bf16 not supported in PTXRenderer** (frontier #10). Workaround: pre-convert weights on disk via `prework/cuda-parity/convert_bf16_to_fp16.py`. Recurring blocker for iter 9 architecture coverage (Mistral, Qwen also bf16).
- Single-script repro (`python infer_llama.py "Once upon a time"`) — not yet packaged. Bench scripts work but the demo-grade script is missing.
- Sampling utilities (top-k, top-p, temperature) — already in `examples/llama3.py:144`, not separately verified.
- KV cache correctness — verified by bit-identical 25-token output; long-output regression suite missing.

**Success criteria status**: tinybox-owner reproducibility partial. Bench scripts work in 10 minutes given the env setup; `pip install`-style install path not done.

### Iter 9 — Inference scale-out (~2-3 weeks)

- Llama 3.2 3B inference (still fits 4080 in fp16)
- Llama 3.1 8B inference with quantization (GGUF Q4 or Q5 — the format people actually run)
- Long-context handling: RoPE scaling for 32K+ tokens
- Streaming generation
- Bench across all three model sizes vs PyTorch + HF + (where applicable) llama.cpp

**Success criteria**: speedygrad covers the dominant local-inference model sizes
(1B/3B/8B) and demonstrates the perf claim on each.

### Iter 10 — LoRA fine-tune (~2-3 weeks)

- LoRA wrapper over `nn.Linear` (~50 LOC, not a peft port)
- AdamW optimizer with proper bf16 / fp32 mixed precision state
- Gradient accumulation
- Cross-entropy loss + label smoothing
- Minimal training loop (~100 LOC)
- HF dataset loading via `datasets` library directly
- Demo: fine-tune Llama 3.2 1B on a small dataset, verify it learns
- Bench: fine-tune tokens/sec vs HF Trainer + peft on same hardware

**Success criteria**: tinybox-owner can fine-tune a small Llama on their own data
without bringing the HF ecosystem.

### Iter 11 — Cross-platform + ecosystem (~3-4 weeks)

- Linux + CUDA target (currently Windows + CUDA only)
- Validate the install path on a tinybox (Ubuntu + NVIDIA stack)
- AMD support if economically justified (gated on tinybox sales mix)
- Add Qwen 2.5 architecture (~200-300 LOC of model definition)
- Add Mistral architecture (~200-300 LOC)
- Documentation: porting guide for "I have a PyTorch fine-tune script, here's how to
  port it to speedygrad with LLM help"

**Success criteria**: speedygrad is usable on the platform tinybox-owners actually run.

### Iter 12+ — Open frontier

- More architectures (Gemma, Phi)
- GGUF native loading (currently partial via Q6K)
- Longer-context inference (paged attention if memory pressure justifies)
- Full fine-tune (not just LoRA)
- Inference perf to actually compete with llama.cpp on local hardware (longer-term, harder)
- Eventually: training-from-scratch infrastructure if real demand emerges

## Perf engineering interactions

The iter 6 + 7 perf wins map onto the priority order favorably:

| Priority | Where iter 6+7 wins land |
|---|---|
| Inference (decode) | **Strongest** — single-token small-tensor work × N tokens. 17us per call × 200 tokens = 3.4ms/sequence saved. |
| Inference (prefill) | Modest — matmul-bound, GPU does real work, host overhead matters less |
| Fine-tune step | Strong — optimizer step is small-tensor heavy, host floor matters |
| Full training | Modest — large-tensor work dominates, host overhead proportionally smaller |

This is another reason to lead with inference: the showcase workload is also the one
where the existing perf wins look most impressive in benchmarks.

### Outstanding perf items (from HYPOTHESIS_GRAPH.md frontier)

In rough priority for shipping under this roadmap:

1. **Softmax 1.62x → ~1.19x (narrows, doesn't close)** — softmax is in the decode inner
   loop AND in attention. **CUDA algorithmic carry verified iter 7.5 post bug-hunt: 1.7x GPU
   speedup at 256x256** (10us vs 17us batched 3-kernel). Wall projects to ~25us if integrated
   as one cuGraph node — narrows the gap to torch ~21us but doesn't beat it. Original 4.2x
   pre-bug-hunt was a kernel-bug artifact (NaN-poisoning init); 6us delta is a GPU
   clock-state / low-duty-cycle artifact (256x256 too sparse to keep GPU at boost). Framework
   integration is honestly 200-400 LOC (synthetic Ops.PROGRAM via `to_program` pattern from
   `extra/gemm/triton_nv_matmul.py`, plus axis/dtype/mask test matrix), 1-2 days focused work.
   To actually close the softmax gap, combine with masked-attention fusion (the
   `exp_2048` polynomial→intrinsic theory was retracted in bug-hunt round 5: tinygrad
   already uses `ex2.approx`). Standalone artifact at `prework/cuda-parity/online_softmax_cuda.py`.
2. **Matvec p90 catastrophic outlier** — late-TC sweep `min`-comparator noise.
   Correctness/quality fix, not perf. Days when convenient.
3. **Attention fusion** — closes the largest remaining gap on transformer inference
   workloads. Builds on online-softmax. ~200 LOC of UOp work.
4. **`_prepare_jit_inputs` Cythonize** — 12us cumtime per JIT call. Messy. Defer
   unless inference bench surfaces it as a bottleneck.

## Out of scope (explicitly)

Listed because saying no is part of the roadmap.

- **Conv-heavy vision workloads (CNN training)**. Without cuDNN-equivalent we lose 2-5x
  to PyTorch. Structurally bad fit. Cede this market segment to torch eager.
- **`torch.compile` backend**. Lets users get perf without porting, dilutes contagion
  leverage (their downstream code never AGPLs), and is downstream of the validation play
  anyway. Not on the near-term roadmap. Reconsider only if a specific enterprise buyer
  requires it.
- **Hosted speedygrad service**. Betrays goal 2 (we'd be the master users wanted to
  escape). Not building.
- **Selling tinycorp a relicense**. Permanent loss of contagion leverage. Not selling.
- **Rebuilding HF-equivalent ecosystem (peft, accelerate, Trainer, transformers)**. The
  abstractions are PyTorch's complexity tax. Build the minimal pieces, demonstrate that
  the abstractions weren't necessary.
- **Distributed multi-node training**. Tinybox is single-node; this is enterprise-DGX
  territory. Defer indefinitely.
- **Speculative decoding, continuous batching, paged attention** — the vLLM moat. Not
  near-term. Reconsider if speedygrad becomes a serving runtime.
- **Apple Silicon support (MLX territory)**. Different platform, different community,
  different runtime norms. Out unless a specific buyer drives it.
- **Production-grade observability, telemetry, monitoring**. Users own the runtime;
  observability is their responsibility. Don't build a phone-home layer.

## Open strategic questions

Things to revisit when more data is in:

- **At what speedup does enterprise procurement bite?** 30%? 2x? Need a real customer
  conversation to calibrate. Affects pricing, prioritization.
- **Is GGUF support a near-term must-have?** Most local-AI users run GGUF via llama.cpp.
  Speedygrad supporting GGUF native load means we can demo on the same files they
  already have. Maybe iter 9 or iter 10.
- **Does AMD support move the needle on tinybox-owner adoption?** Need data on the
  red-vs-green tinybox sales split. If 50/50, AMD is required. If 80/20 NVIDIA, defer.
- **Should there be a paid hosted bench dashboard?** Tinybox-owners run benchmarks; a
  central place to see them all would amplify the validation signal. But: hosted service
  brushes against goal 2. Probably not. Could be GitHub Discussions instead.
- **When (if ever) does it make sense to talk to tinycorp about a tinybox-tuned
  distribution?** Not until speedygrad has independent traction and they have a reason
  to want the conversation. Don't initiate.

## Working notes

- HYPOTHESIS_GRAPH.md is the perf record (iter 1-7 fully documented).
- This file is the strategy record. Updated when priorities shift, not when individual
  iterations ship.
- Iteration timing estimates are working estimates, not commitments.
- "Soon" means weeks, not days. "Later" means quarters. "Eventually" means if-ever.
