"""Test the iter 10c-cont hypothesis: Tensor._device_rng_counters['CUDA']
is the dominant tensor in _apply_map_to_tensors's per-decode walk. It has
a 1615-UOp deep AFTER chain accumulated from 114 weight-init `.assign()`s
during build_transformer, never realized.

If we call .realize() on it once after model load, the AFTER chain
collapses to a single BUFFER and the dominant cost should drop ~73%.

Compares two runs side-by-side:
  RUN A: baseline, no rng counter realize after load
  RUN B: rng counter .realize() called once after load + after burn

Reports decode wall-clock and _apply_map_to_tensors total per token for both.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops, TracingKey
from tinygrad.helpers import cpu_profile

# instrumented apply that just totals the cost
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
tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

def run_decode(model, n_burn=5, n_decode=50, label=""):
    APPLY_TOTAL_NS[0] = 0
    APPLY_CALLS[0] = 0
    start_pos = 0
    for t in toks[:-1]:
        model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
        start_pos += 1
    last = toks[-1]
    for _ in range(n_burn):
        last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
        start_pos += 1
    # report counter chain depth
    counter = Tensor._device_rng_counters.get(str(device))
    if counter is not None:
        n = len(counter.uop.toposort())
        print(f"  [{label}] rng counter UOp DAG size: {n}", file=sys.stderr)
    else:
        print(f"  [{label}] rng counter not present", file=sys.stderr)
    # start recording
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
    apply_per_tok_us = APPLY_TOTAL_NS[0] / APPLY_CALLS[0] / 1000.0 if APPLY_CALLS[0] else 0
    apply_share = 100 * apply_per_tok_us / p50 if p50 > 0 else 0
    print(f"  [{label}] decode_p50={p50:.0f}us  _apply_map_to_tensors per call={apply_per_tok_us:.0f}us "
          f"({apply_share:.1f}% of wall, {APPLY_CALLS[0]} calls / {n_decode} tokens)")
    return p50, apply_per_tok_us

# --- RUN A: baseline ---
print("RUN A: baseline (no rng counter realize)", file=sys.stderr)
print("  loading model A...", file=sys.stderr)
model_a = build_transformer(model_path, model_size="1B", quantize=None, device=device)
counter_a = Tensor._device_rng_counters.get(str(device))
if counter_a is not None:
    print(f"  [A pre-decode] rng counter UOp DAG size at construction time: {len(counter_a.uop.toposort())}", file=sys.stderr)
p50_a, apply_a = run_decode(model_a, label="A baseline")

# --- RUN B: realize the rng counter after load ---
print()
print("RUN B: realize rng counter after load", file=sys.stderr)
print("  loading model B...", file=sys.stderr)
# Reset the rng state so we get a fresh counter for this run
Tensor._device_seeds = {}
Tensor._device_rng_counters = {}
model_b = build_transformer(model_path, model_size="1B", quantize=None, device=device)
counter_b = Tensor._device_rng_counters.get(str(device))
if counter_b is not None:
    pre = len(counter_b.uop.toposort())
    counter_b.realize()
    post = len(counter_b.uop.toposort())
    print(f"  [B] rng counter UOp DAG size before realize: {pre}, after realize: {post}", file=sys.stderr)
else:
    print(f"  [B] rng counter not present", file=sys.stderr)
p50_b, apply_b = run_decode(model_b, label="B with .realize()")

# --- summary ---
print()
print(f"=== SUMMARY ===")
print(f"  A (baseline):           decode_p50={p50_a:.0f}us  apply={apply_a:.0f}us")
print(f"  B (realize counter):    decode_p50={p50_b:.0f}us  apply={apply_b:.0f}us")
print(f"  delta:                  decode={p50_b - p50_a:+.0f}us ({100*(p50_b-p50_a)/p50_a:+.1f}%)  "
      f"apply={apply_b - apply_a:+.0f}us ({100*(apply_b-apply_a)/apply_a:+.1f}%)")
