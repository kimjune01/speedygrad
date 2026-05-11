"""Cleaner A/B for the rng-counter-realize hypothesis.

v1 had a bug: it loaded TWO models in the same process, doubling all_tensors
and inflating Run B's _apply_map_to_tensors cost.

v2: single model. Measure decode A (baseline). Then realize the rng counter.
Then measure decode B. Same model, same all_tensors footprint — only the
counter's UOp chain has changed.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops

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
tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

# initial counter chain check
counter = Tensor._device_rng_counters.get(str(device))
print(f"rng counter chain depth at construction: "
      f"{len(counter.uop.toposort()) if counter is not None else 'N/A'}", file=sys.stderr)

# prefill once
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]

def burn_and_measure(n_burn=5, n_decode=50, label=""):
    global last, start_pos
    APPLY_TOTAL_NS[0] = 0
    APPLY_CALLS[0] = 0
    for _ in range(n_burn):
        last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
        start_pos += 1
    counter = Tensor._device_rng_counters.get(str(device))
    chain_size = len(counter.uop.toposort()) if counter is not None else 'N/A'
    n_live = sum(1 for tref in list(_tensor_mod.all_tensors) if tref() is not None)
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
    share = 100 * apply_per_call / p50 if p50 > 0 else 0
    print(f"  [{label}] live_tensors={n_live} counter_chain={chain_size}")
    print(f"  [{label}] decode_p50={p50:.0f}us  apply={apply_per_call:.0f}us  ({share:.1f}% of wall)")
    return p50, apply_per_call

# === Measurement A: baseline ===
print("MEASUREMENT A: baseline (counter chain unrealized)", file=sys.stderr)
p50_a, apply_a = burn_and_measure(n_burn=5, n_decode=50, label="A baseline")

# === Realize the counter ===
print()
print("--- realizing rng counter ---", file=sys.stderr)
counter = Tensor._device_rng_counters.get(str(device))
if counter is not None:
    pre = len(counter.uop.toposort())
    counter.realize()
    post = len(counter.uop.toposort())
    print(f"  counter chain: {pre} -> {post} after realize")

# === Measurement B: post-realize ===
print()
print("MEASUREMENT B: counter realized", file=sys.stderr)
p50_b, apply_b = burn_and_measure(n_burn=5, n_decode=50, label="B realized")

print()
print(f"=== SUMMARY (single model, A→realize→B) ===")
print(f"  A (chain=1615):   decode_p50={p50_a:.0f}us  apply={apply_a:.0f}us")
print(f"  B (chain=3):      decode_p50={p50_b:.0f}us  apply={apply_b:.0f}us")
print(f"  delta:            decode={p50_b - p50_a:+.0f}us ({100*(p50_b-p50_a)/p50_a:+.1f}%)  "
      f"apply={apply_b - apply_a:+.0f}us ({100*(apply_b-apply_a)/apply_a:+.1f}%)")
