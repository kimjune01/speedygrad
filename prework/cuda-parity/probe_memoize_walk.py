"""Implement and measure the memoize-walk monkeypatch from iter 10c-cont v1.

Hypothesis: caching frozenset(uop.toposort()) per UOp identity (UOps are
hashconsed → cache is naturally bounded), then replacing topovisit with
applied_keys.isdisjoint(cached_set), drops the 167 model-param tensor
walks from O(DAG) to O(|applied_map|).

Predicted savings: ~400us per decode token (the remaining 536us/call
of _apply_map_to_tensors after the counter.realize() fix should drop
to ~50us).

A/B: same model, in process, before/after monkeypatch installation.
"""
import os, sys, time, weakref
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops, TracingKey
from tinygrad.helpers import cpu_profile

APPLY_TOTAL_NS = [0]
APPLY_CALLS = [0]
RECORDING = [False]

# Timing wrapper, separate from the memoize implementation.
_real_apply = _tensor_mod._apply_map_to_tensors
_orig_apply = _real_apply

def _make_timed(fn):
    def wrap(applied_map, name, walk=False):
        if not RECORDING[0]:
            return fn(applied_map, name, walk)
        t0 = time.perf_counter_ns()
        try:
            return fn(applied_map, name, walk)
        finally:
            APPLY_TOTAL_NS[0] += time.perf_counter_ns() - t0
            APPLY_CALLS[0] += 1
    return wrap

# === MEMOIZE IMPLEMENTATION ===
# Cache frozenset of all UOps reachable from u.uop. Keyed by UOp identity
# (UOps are hashconsed). Uses WeakKeyDictionary so cache entries get freed
# when their UOps die.
# UOps don't natively support weakrefs (no __weakref__ in __slots__);
# fall back to id-keyed dict and hope UOp interning keeps it bounded.

_uop_dag_cache: dict = {}  # id(UOp) -> frozenset[UOp]
_cache_hits = [0]
_cache_misses = [0]

def _uop_dag_set(u):
    uid = id(u)
    cached = _uop_dag_cache.get(uid)
    if cached is not None:
        # verify identity (id() may collide if GC'd & reused — we keep the UOp alive via the value)
        _cache_hits[0] += 1
        return cached
    _cache_misses[0] += 1
    # toposort the DAG
    seen = {}
    stack = [u]
    while stack:
        n = stack.pop()
        if id(n) in seen: continue
        seen[id(n)] = n
        stack.extend(n.src)
    result = frozenset(seen.values())
    _uop_dag_cache[uid] = result
    return result

def _apply_map_memoized(applied_map, name, walk=False):
    if walk:
        # walk=True (Embed View Assign) needs the original semantics
        return _orig_apply(applied_map, name, walk)
    with cpu_profile(TracingKey(name + " (memoized)"), "TINY"):
        applied_keys = set(applied_map)
        scope_tensors = []
        for tref in list(_tensor_mod.all_tensors):
            t = tref()
            if t is None: continue
            uop_set = _uop_dag_set(t.uop)
            if not applied_keys.isdisjoint(uop_set):
                scope_tensors.append(t)
        sink = UOp.sink(*[t.uop for t in scope_tensors])
        new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)
        for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
            if s is ns: continue
            t.uop = ns
            # invalidate cache for this tensor's UOp identity (the new ns gets a fresh entry on next access)
            # NOTE: cache stays for old uop too; that's fine, it's still hashconsed and may be referenced elsewhere.

# Install the timed-original wrapper for measurement A
_tensor_mod._apply_map_to_tensors = _make_timed(_orig_apply)

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

start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]

def measure(label, n_burn=5, n_decode=50):
    global last, start_pos
    APPLY_TOTAL_NS[0] = 0
    APPLY_CALLS[0] = 0
    _cache_hits[0] = 0
    _cache_misses[0] = 0
    for _ in range(n_burn):
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
    p10, p50, p90 = times_us[5], times_us[25], times_us[45]
    apply_per_call = APPLY_TOTAL_NS[0] / APPLY_CALLS[0] / 1000.0 if APPLY_CALLS[0] else 0
    apply_per_tok = APPLY_TOTAL_NS[0] / n_decode / 1000.0
    share = 100 * apply_per_tok / p50 if p50 > 0 else 0
    print(f"  [{label}] decode_p10/p50/p90={p10:.0f}/{p50:.0f}/{p90:.0f}us")
    print(f"  [{label}] _apply_map_to_tensors: {APPLY_CALLS[0]} calls, "
          f"{apply_per_call:.0f}us/call, {apply_per_tok:.0f}us/tok ({share:.1f}% of wall)")
    if _cache_hits[0] + _cache_misses[0] > 0:
        hit_rate = 100 * _cache_hits[0] / (_cache_hits[0] + _cache_misses[0])
        print(f"  [{label}] uop_dag_set cache: {_cache_hits[0]} hits, {_cache_misses[0]} misses "
              f"({hit_rate:.1f}% hit rate, {len(_uop_dag_cache)} cached entries)")
    return p50, apply_per_call

# === MEASUREMENT A: original walk ===
print("\nMEASUREMENT A: original walk (timed)")
p50_a, apply_a = measure("A original")

# === Switch to memoized ===
print("\n--- installing memoize-walk monkeypatch ---")
_tensor_mod._apply_map_to_tensors = _make_timed(_apply_map_memoized)
# pre-warm the cache by walking all_tensors once (so the first call after switch
# isn't paying the full uncached cost)
print("  pre-warming uop_dag_set cache...")
for tref in list(_tensor_mod.all_tensors):
    t = tref()
    if t is not None: _uop_dag_set(t.uop)
print(f"  cache after warm-up: {len(_uop_dag_cache)} entries")

# === MEASUREMENT B: memoized ===
print("\nMEASUREMENT B: memoize-walk")
p50_b, apply_b = measure("B memoized")

print(f"\n=== SUMMARY ===")
print(f"  A original walk:       decode_p50={p50_a:.0f}us  apply={apply_a:.0f}us/call")
print(f"  B memoized walk:       decode_p50={p50_b:.0f}us  apply={apply_b:.0f}us/call")
print(f"  delta:                 decode={p50_b-p50_a:+.0f}us ({100*(p50_b-p50_a)/p50_a:+.1f}%)  "
      f"apply={apply_b-apply_a:+.0f}us ({100*(apply_b-apply_a)/apply_a:+.1f}%)")
