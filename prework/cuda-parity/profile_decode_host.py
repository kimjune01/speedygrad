"""iter 10c: cProfile a tight decode-only loop to find where the ~1.9ms/tok host time goes.

Skips prefill (uses .item() at end of prefill to drain pending_copyin and warm caches).
Profiles 50 steady-state decode tokens, then prints top cumtime/tottime by function.
"""
import os, sys, cProfile, pstats, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
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

# warm-up + prefill
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
# burn 5 decode tokens to reach steady state
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

print("profiling 50 decode tokens...", file=sys.stderr)
N = 50
times = []
def decode_loop():
    global last, start_pos
    for _ in range(N):
        t0 = time.perf_counter()
        last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
        start_pos += 1

p = cProfile.Profile()
p.enable()
decode_loop()
p.disable()

times.sort()
p10, p50, p90 = times[N//10], times[N//2], times[9*N//10]
print(f"\nDecode wall: p10={p10:.0f}us  p50={p50:.0f}us  p90={p90:.0f}us  mean={sum(times)/N:.0f}us", file=sys.stderr)

print("\n=== top 30 by cumulative time ===")
pstats.Stats(p).sort_stats("cumulative").print_stats(30)
print("\n=== top 30 by total (own) time ===")
pstats.Stats(p).sort_stats("tottime").print_stats(30)
