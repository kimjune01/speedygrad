"""Scan all_tensors for stale-state Tensors with deep UOp chains beyond
the RNG counter. Then A/B: realize all stale candidates vs baseline.

iter 10c-cont v2 found one stale tensor (the RNG counter) accounting for
73% of _apply_map_to_tensors cost. The pattern is: a Tensor in all_tensors
held by some long-lived dict/list, with a deep AFTER/STORE chain that
accumulated during model construction and was never realized. This probe
asks: are there OTHERS we missed by filtering on shape=(2,) uint?

Method:
  1. Load model, do full prefill + decode_burn to reach steady state.
  2. Snapshot all_tensors, sort by n_uops descending.
  3. Print top 30 with op, shape, dtype, n_uops, BUFFER leaf summary,
     category (model_param / captured_ret / other / rng_counter).
  4. A/B: realize every "other deep" tensor (n_uops > 50, not model_param,
     not captured_ret), measure decode delta vs baseline.
"""
import os, sys, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops
from tinygrad.nn.state import get_state_dict

APPLY_TOTAL_NS = [0]
APPLY_CALLS = [0]
RECORDING = [False]

_orig_apply = _tensor_mod._apply_map_to_tensors
def _instrumented_apply(applied_map, name, walk=False):
    if not RECORDING[0]:
        return _orig_apply(applied_map, name, walk)
    t0 = time.perf_counter_ns()
    try:
        return _orig_apply(applied_map, name, walk)
    finally:
        APPLY_TOTAL_NS[0] += time.perf_counter_ns() - t0
        APPLY_CALLS[0] += 1
_tensor_mod._apply_map_to_tensors = _instrumented_apply

from examples.llama3 import build_transformer
from transformers import AutoTokenizer

model_path = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"
device = Device.DEFAULT

print("loading...", file=sys.stderr)
model = build_transformer(model_path, model_size="1B", quantize=None, device=device)

# build category lookup
sd = get_state_dict(model)
PARAM_UOP_IDS = {id(t.uop) for _, t in sd.items() if isinstance(t, Tensor)}
RNG_COUNTER_IDS = {id(t) for t in Tensor._device_rng_counters.values()}
RNG_SEED_IDS = {id(t) for t in Tensor._device_seeds.values()}

tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# update RET id (now that JIT is captured)
RET_ID = None
if model.forward_jit is not None and model.forward_jit.captured is not None:
    cap_ret = model.forward_jit.captured.ret
    if isinstance(cap_ret, Tensor): RET_ID = id(cap_ret)

# === scan all_tensors ===
print("\n=== scanning all_tensors for deep DAGs ===")
candidates = []
for tref in list(_tensor_mod.all_tensors):
    t = tref()
    if t is None: continue
    n_uops = len(t.uop.toposort())
    candidates.append((n_uops, t))

candidates.sort(key=lambda c: -c[0])
print(f"total live tensors: {len(candidates)}")
print(f"top 30 by UOp DAG size:")
print(f"  {'n_uops':>7} {'op':<10} {'shape':<25} {'dtype':<14} {'category':<14} {'id':>16}")
def categorize(t):
    if id(t) in RNG_COUNTER_IDS: return "rng_counter"
    if id(t) in RNG_SEED_IDS: return "rng_seed"
    if RET_ID is not None and id(t) == RET_ID: return "captured_ret"
    if id(t.uop) in PARAM_UOP_IDS: return "model_param"
    return "other"

for n_uops, t in candidates[:30]:
    try: shape = str(tuple(t.uop.shape))
    except Exception: shape = "?"
    dtype = str(t.uop.dtype).replace("dtypes.", "")
    op = t.uop.op.name
    cat = categorize(t)
    print(f"  {n_uops:>7} {op:<10} {shape:<25} {dtype:<14} {cat:<14} {id(t):>16}")

# === BUFFER-leaf summary for the top "other" deep tensors ===
print(f"\n=== BUFFER leaves of top 'other' deep tensors (n_uops > 50, non-rng, non-param) ===")
deep_other = [(n, t) for n, t in candidates if n > 50 and categorize(t) == "other"]
print(f"found {len(deep_other)} 'other' tensors with n_uops > 50")
for n_uops, t in deep_other[:10]:
    bufs = [u for u in t.uop.toposort() if u.op is Ops.BUFFER]
    buf_summary = Counter()
    for b in bufs:
        try: bsh = tuple(b.shape)
        except Exception: bsh = None
        buf_summary[(bsh, str(b.dtype).replace("dtypes.", ""), str(b.device) if hasattr(b, 'device') else '?')] += 1
    print(f"\n  tensor id={id(t)} op={t.uop.op.name} shape={tuple(t.uop.shape)} n_uops={n_uops}")
    print(f"    {len(bufs)} BUFFER leaves; top shapes:")
    for (bsh, bdt, bdv), cnt in sorted(buf_summary.items(), key=lambda x: -x[1])[:5]:
        print(f"      {cnt:>4} x  shape={bsh} dtype={bdt} dev={bdv}")
    op_hist = Counter(u.op.name for u in t.uop.toposort())
    print(f"    top 5 op-types in DAG:")
    for op, cnt in sorted(op_hist.items(), key=lambda x: -x[1])[:5]:
        print(f"      {op:<14} {cnt}")

# === A/B: realize all "other" deep tensors and re-measure ===
def measure_decode(label, n_decode=50):
    global last, start_pos
    APPLY_TOTAL_NS[0] = 0
    APPLY_CALLS[0] = 0
    # short re-burn
    for _ in range(3):
        last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
        start_pos += 1
    RECORDING[0] = True
    times_us = []
    for _ in range(n_decode):
        t0 = time.perf_counter()
        last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
        start_pos += 1
    RECORDING[0] = False
    times_us.sort()
    p50 = times_us[len(times_us)//2]
    apply_per_call = APPLY_TOTAL_NS[0] / APPLY_CALLS[0] / 1000.0 if APPLY_CALLS[0] else 0
    print(f"  [{label}] decode_p50={p50:.0f}us  apply={apply_per_call:.0f}us "
          f"({100*apply_per_call/p50:.1f}% of wall)")
    return p50, apply_per_call

print(f"\n=== A/B: progressive realize ===")
print("  baseline (nothing realized this run yet — chain still 1615):")
p50_base, apply_base = measure_decode("baseline")

# realize JUST the rng counter (sanity check)
print("\n  after Tensor._device_rng_counters['CUDA'].realize():")
for c in Tensor._device_rng_counters.values():
    c.realize()
p50_rng, apply_rng = measure_decode("rng-realized")

# realize everything in deep_other (excluding the rng counter, which is already realized)
print("\n  after .realize() on every 'other' tensor with n_uops > 50:")
realized_count = 0
for n_uops, t in deep_other:
    try:
        t.realize()
        realized_count += 1
    except Exception as e:
        print(f"    failed to realize tensor id={id(t)} ({type(e).__name__}: {e})", file=sys.stderr)
print(f"    realized {realized_count} tensors")
p50_all, apply_all = measure_decode("all-realized")

print(f"\n=== A/B SUMMARY ===")
print(f"  baseline (chain=1615):          decode_p50={p50_base:.0f}us  apply={apply_base:.0f}us")
print(f"  +rng-counter realized:           decode_p50={p50_rng:.0f}us  apply={apply_rng:.0f}us  (delta {p50_rng-p50_base:+.0f}us / {apply_rng-apply_base:+.0f}us)")
print(f"  +all 'other' deep realized:      decode_p50={p50_all:.0f}us  apply={apply_all:.0f}us  (delta {p50_all-p50_rng:+.0f}us / {apply_all-apply_rng:+.0f}us)")
