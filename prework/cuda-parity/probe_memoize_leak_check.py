"""iter 10c-cont v4: verify the memoize-walk cache no longer leaks.

v3 cache: dict[id(uop) -> frozenset[uop]]. The frozenset value held UOps
alive, so per-decode-token input UOps accumulated forever.

v4 cache: dict[id(uop) -> frozenset[id(uop)]] (just integers, doesn't hold
UOps alive) + weakref.finalize(uop, cache.pop, id(uop), None) registered
when each entry is created.

Test: run 200 decode tokens. Watch cache size every 50 tokens. Should
plateau at ~167 (the live tensor count) instead of growing linearly.

Also confirm: end-to-end perf is still as good as v3 (the id-based check
shouldn't be slower).
"""
import os, sys, gc, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401  -- now includes the v4 leak-free memoize-walk
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod

# pull the cache out so we can inspect it
_uop_dag_id_cache = monkeypatch._uop_dag_id_cache_mw

from examples.llama3 import build_transformer
from transformers import AutoTokenizer

model_path = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"
device = Device.DEFAULT

print("loading...", file=sys.stderr)
model = build_transformer(model_path, model_size="1B", quantize=None, device=device)
for c in Tensor._device_rng_counters.values(): c.realize()
tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

print(f"cache before any forward: {len(_uop_dag_id_cache)}")

# prefill
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
print(f"cache after prefill: {len(_uop_dag_id_cache)}")

# burn 5 to settle
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1
print(f"cache after burn-in: {len(_uop_dag_id_cache)}")

# Run 200 decode tokens, sample cache size every 50
N = 200
sample_every = 50
times_us = []
sizes_at = []
for i in range(N):
    t0 = time.perf_counter()
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    t1 = time.perf_counter()
    times_us.append((t1 - t0) * 1e6)
    start_pos += 1
    if (i + 1) % sample_every == 0:
        # force a GC to make sure finalizers fire
        gc.collect()
        sizes_at.append((i+1, len(_uop_dag_id_cache)))

print(f"\n=== cache size over {N} decode tokens (forced gc.collect every {sample_every}) ===")
for i, sz in sizes_at:
    print(f"  after token {i:>4}: cache={sz}")

# Final check after a final gc
gc.collect()
print(f"\n  final after gc.collect: cache={len(_uop_dag_id_cache)}")

times_us.sort()
print(f"\n=== decode wall ({N} tokens) ===")
print(f"  p10={times_us[N//10]:.0f}us  p50={times_us[N//2]:.0f}us  p90={times_us[9*N//10]:.0f}us")
print(f"  mean={sum(times_us)/N:.0f}us")

# Sanity check the cache: count entries pointing to live vs dead
live_uop_ids = set()
for tref in list(_tensor_mod.all_tensors):
    t = tref()
    if t is None: continue
    live_uop_ids.add(id(t.uop))

alive_in_cache = sum(1 for k in _uop_dag_id_cache if k in live_uop_ids)
print(f"\n=== cache health ===")
print(f"  live tensor uop ids: {len(live_uop_ids)}")
print(f"  cache entries whose key matches a live tensor: {alive_in_cache}")
print(f"  cache entries whose key is NOT a live tensor uop: {len(_uop_dag_id_cache) - alive_in_cache}")
print(f"  (the second number includes intermediate UOps reachable from live tensors —")
print(f"   that's fine; they're alive too. Concerning would be if cache >> total live UOps.)")
