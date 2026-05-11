"""A/B test: does counter.realize() also help prefill?

iter 10c-cont (commit 114e2276d) measured prefill at 11.75 ms/token of
_apply_map_to_tensors cost. The same global RNG counter is walked there
too — but with one twist: prefill happens IMMEDIATELY after model
construction, so the counter chain is at its fullest depth during
prefill (vs decode which sees it after .item() syncs may have done
incidental work).

This probe runs the same prefill twice in one process:
  A: counter chain unrealized (baseline)
  B: counter.realize() called BEFORE prefill

(Resets last_seen_toks state between runs by re-instantiating prompt.)
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod

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
prefill_toks = toks[:-1]
n_prefill = len(prefill_toks)
print(f"prefill length: {n_prefill}", file=sys.stderr)

counter = Tensor._device_rng_counters.get(str(device))
print(f"counter chain at construction: "
      f"{len(counter.uop.toposort()) if counter is not None else 'N/A'}", file=sys.stderr)

def measure_prefill(label, start_pos):
    APPLY_TOTAL_NS[0] = 0
    APPLY_CALLS[0] = 0
    RECORDING[0] = True
    t_total_start = time.perf_counter()
    for tk in prefill_toks:
        model(Tensor([[tk]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
        start_pos += 1
    t_total = time.perf_counter() - t_total_start
    RECORDING[0] = False
    apply_per_token = APPLY_TOTAL_NS[0] / n_prefill / 1000.0
    apply_per_call = APPLY_TOTAL_NS[0] / APPLY_CALLS[0] / 1000.0 if APPLY_CALLS[0] else 0
    wall_per_tok_us = t_total * 1e6 / n_prefill
    print(f"  [{label}] prefill_wall={t_total*1000:.1f}ms over {n_prefill} toks "
          f"({wall_per_tok_us:.0f}us/tok)")
    print(f"  [{label}] _apply_map_to_tensors: {APPLY_CALLS[0]} calls, "
          f"{apply_per_token:.0f}us/tok ({apply_per_call:.1f}us/call), "
          f"{100*apply_per_token/wall_per_tok_us:.1f}% of wall")
    return start_pos, wall_per_tok_us, apply_per_token

# === A: baseline ===
print("\nMEASUREMENT A: baseline prefill (counter chain=1615)")
start_pos = 0
start_pos, wall_a, apply_a = measure_prefill("A baseline", start_pos)

# Now realize the counter
print("\n--- realizing rng counter ---")
counter.realize()
print(f"  counter chain after realize: {len(counter.uop.toposort())}")

# === B: post-realize prefill (start fresh — re-prefill from start_pos=current) ===
# To make this comparable we need the same workload pattern. Easiest: just
# do another prefill of the same length from current start_pos. The work
# is similar (each call is a single-token JIT replay — start_pos affects
# only the symbolic var binding, not the kernel structure).
print("\nMEASUREMENT B: prefill after counter.realize() (counter chain=3)")
start_pos, wall_b, apply_b = measure_prefill("B realized", start_pos)

print(f"\n=== SUMMARY ===")
print(f"  A baseline:     wall={wall_a:.0f}us/tok  apply={apply_a:.0f}us/tok")
print(f"  B realized:     wall={wall_b:.0f}us/tok  apply={apply_b:.0f}us/tok")
print(f"  delta:          wall={wall_b-wall_a:+.0f}us ({100*(wall_b-wall_a)/wall_a:+.1f}%)  "
      f"apply={apply_b-apply_a:+.0f}us ({100*(apply_b-apply_a)/apply_a:+.1f}%)")
