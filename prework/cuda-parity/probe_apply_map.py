"""iter 10c re-measurement: instrument _apply_map_to_tensors with perf_counter_ns
WITHOUT cProfile, to bound the actual host cost per decode token.

Per gemini's review of iter 10c gate design (2026-05-11): the cProfile-based
1.5-2 ms/tok estimate is not load-bearing — cProfile inflates tight Python
lambda loops 5-15x, not the assumed 3x. Use raw perf_counter_ns instead.

Reports:
- per-phase total time (load / prefill / decode)
- per-phase call count
- per-call mean and per-decode-token mean for the steady-state decode phase
- decode wall-clock comparison (with vs without instrumentation overhead)
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod

PHASE = ["load"]  # mutable, so closure can read latest
counts = {"load": 0, "prefill": 0, "decode_burn": 0, "decode": 0}
totals_ns = {"load": 0, "prefill": 0, "decode_burn": 0, "decode": 0}

_orig_apply = _tensor_mod._apply_map_to_tensors

def _instrumented_apply(applied_map, name, walk=False):
    t0 = time.perf_counter_ns()
    try:
        return _orig_apply(applied_map, name, walk)
    finally:
        dt = time.perf_counter_ns() - t0
        ph = PHASE[0]
        counts[ph] += 1
        totals_ns[ph] += dt

_tensor_mod._apply_map_to_tensors = _instrumented_apply

# also instrument the import-site rebind in tinygrad.tensor for callify/realize call paths
# (they import _apply_map_to_tensors as a module-level name and lookup-by-attribute, so
# rebinding the module attribute is sufficient — they will pick up the new ref on call.)

from examples.llama3 import build_transformer
from transformers import AutoTokenizer

model_path = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"
device = Device.DEFAULT

print("loading...", file=sys.stderr)
PHASE[0] = "load"
model = build_transformer(model_path, model_size="1B", quantize=None, device=device)
tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

PHASE[0] = "prefill"
print(f"prefill {len(toks)-1} tokens...", file=sys.stderr)
start_pos = 0
prefill_t0 = time.perf_counter()
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
prefill_wall_ms = (time.perf_counter() - prefill_t0) * 1000
last = toks[-1]

PHASE[0] = "decode_burn"
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

PHASE[0] = "decode"
N = 50
times_us = []
print(f"timing {N} steady-state decode tokens (instrumented)...", file=sys.stderr)
for _ in range(N):
    t0 = time.perf_counter()
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    t1 = time.perf_counter()
    times_us.append((t1 - t0) * 1e6)
    start_pos += 1

times_us.sort()
p10, p50, p90 = times_us[N//10], times_us[N//2], times_us[9*N//10]

print()
print(f"=== _apply_map_to_tensors timing (raw perf_counter_ns, no cProfile) ===")
print(f"{'phase':<14} {'calls':>8} {'total_us':>12} {'per_call_us':>14} {'per_token_us':>14}")
def per_token(ph, ntok):
    if ntok == 0: return ""
    return f"{totals_ns[ph]/1000.0/ntok:.1f}"
print(f"{'load':<14} {counts['load']:>8} {totals_ns['load']/1000.0:>12.0f} "
      f"{(totals_ns['load']/counts['load']/1000.0 if counts['load'] else 0):>14.1f}")
prefill_n = len(toks) - 1
print(f"{'prefill':<14} {counts['prefill']:>8} {totals_ns['prefill']/1000.0:>12.0f} "
      f"{(totals_ns['prefill']/counts['prefill']/1000.0 if counts['prefill'] else 0):>14.1f} "
      f"{per_token('prefill', prefill_n):>14}")
print(f"{'decode_burn':<14} {counts['decode_burn']:>8} {totals_ns['decode_burn']/1000.0:>12.0f} "
      f"{(totals_ns['decode_burn']/counts['decode_burn']/1000.0 if counts['decode_burn'] else 0):>14.1f}")
print(f"{'decode (steady)':<14} {counts['decode']:>8} {totals_ns['decode']/1000.0:>12.0f} "
      f"{(totals_ns['decode']/counts['decode']/1000.0 if counts['decode'] else 0):>14.1f} "
      f"{per_token('decode', N):>14}")

print()
print(f"=== decode wall-clock (instrumented) ===")
print(f"p10={p10:.0f}us  p50={p50:.0f}us  p90={p90:.0f}us  mean={sum(times_us)/N:.0f}us")
print()
print(f"=== prefill wall-clock (instrumented) ===")
print(f"prefill_wall={prefill_wall_ms:.1f}ms over {prefill_n} tokens "
      f"({prefill_wall_ms*1000/prefill_n:.0f}us/tok)")
print()
decode_apply_us = totals_ns['decode'] / N / 1000.0
decode_wall_us = sum(times_us) / N
share = 100.0 * decode_apply_us / decode_wall_us if decode_wall_us > 0 else 0
print(f"=== summary: _apply_map_to_tensors share of decode wall ===")
print(f"per-token: apply_map={decode_apply_us:.0f}us / wall={decode_wall_us:.0f}us = {share:.1f}%")
