"""iter 10c continuation: inner-cost breakdown of _apply_map_to_tensors.

Iter 10c established the function takes 1.98 ms / decode token (25% of wall).
This probe asks: WHERE in the 1.98 ms? Without this, picking memoize vs.
Cython-port-topovisit is a guess.

Reimplements _apply_map_to_tensors with per-stage and per-tensor instrumentation,
sampled only on steady-state decode (load and prefill skipped).

Reports:
  - Per-call stage breakdown:
      walk      = the list-comp over all_tensors with topovisit
      sink      = UOp.sink(*[t.uop for t in scope_tensors])
      subst     = sink.substitute(applied_map, ...)
      assign    = the t.uop = ns reassignment loop
  - Per-tensor walk time histogram (log buckets), grouped by t.uop.op
  - Per-tensor DAG-newly-visited-UOp-count histogram (proxy for walk cost)
  - Top-10 slowest tensors per call (averaged across tokens)
  - "Dead weight" stat: time spent on tensors that ultimately returned False
    from the visitor (the wasted-work fraction the killed gate was trying to skip)
"""
import os, sys, time, weakref
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, TracingKey
from tinygrad.helpers import cpu_profile

PHASE = ["load"]
SAMPLES = []  # list of per-call records, only filled in "decode" phase

_orig_apply = _tensor_mod._apply_map_to_tensors

def _instrumented_apply(applied_map, name, walk=False):
    if PHASE[0] != "decode":
        return _orig_apply(applied_map, name, walk)

    record = {"name": name, "walk_us": 0.0, "sink_us": 0.0, "subst_us": 0.0,
              "assign_us": 0.0, "n_total": 0, "n_in_scope": 0,
              "per_tensor": [],  # list of (op_name, walk_us, new_uops_added, in_scope_result)
              "dag_size_total": 0}

    with cpu_profile(TracingKey(name), "TINY"):
        in_scope: dict = {}
        def visitor(node):
            return True if node in applied_map else any(in_scope.get(s, False) for s in node.src)

        # === stage 1: walk all_tensors with per-tensor timing ===
        t_walk_start = time.perf_counter_ns()
        scope_tensors = []
        prev_in_scope_size = 0
        for tref in list(_tensor_mod.all_tensors):
            t = tref()
            if t is None:
                continue
            t0 = time.perf_counter_ns()
            result = t.uop.topovisit(visitor, in_scope)
            t1 = time.perf_counter_ns()
            new_added = len(in_scope) - prev_in_scope_size
            prev_in_scope_size = len(in_scope)
            op_name = t.uop.op.name if hasattr(t.uop.op, 'name') else str(t.uop.op)
            # also capture shape/dtype/device for the slowest tensors so we can identify them
            try:
                shape = tuple(t.uop.shape)
                dtype = str(t.uop.dtype)
                dev = str(t.uop.device)
            except Exception:
                shape, dtype, dev = None, None, None
            record["per_tensor"].append((op_name, (t1 - t0) / 1000.0, new_added, bool(result), shape, dtype, dev))
            record["n_total"] += 1
            if result:
                scope_tensors.append(t)
                record["n_in_scope"] += 1
        record["walk_us"] = (time.perf_counter_ns() - t_walk_start) / 1000.0
        record["dag_size_total"] = len(in_scope)

        # === stage 2: sink ===
        t0 = time.perf_counter_ns()
        sink = UOp.sink(*[t.uop for t in scope_tensors])
        record["sink_us"] = (time.perf_counter_ns() - t0) / 1000.0

        # === stage 3: substitute ===
        t0 = time.perf_counter_ns()
        new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)
        record["subst_us"] = (time.perf_counter_ns() - t0) / 1000.0

        # === stage 4: assign ===
        t0 = time.perf_counter_ns()
        for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
            if s is ns:
                continue
            t.uop = ns
        record["assign_us"] = (time.perf_counter_ns() - t0) / 1000.0

    SAMPLES.append(record)

# rebind
_tensor_mod._apply_map_to_tensors = _instrumented_apply

# --- run the bench ---
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
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]

PHASE[0] = "decode_burn"
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

PHASE[0] = "decode"
N = 50
print(f"sampling {N} steady-state decode tokens (instrumented inner)...", file=sys.stderr)
for _ in range(N):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# --- analyze ---
print()
print(f"=== _apply_map_to_tensors INNER breakdown ({len(SAMPLES)} calls, decode steady-state only) ===")
print(f"(only call name == 'buffers' is decode-relevant; others are filtered)")
buffers_samples = [s for s in SAMPLES if s["name"] == "buffers"]
print(f"calls with name=='buffers': {len(buffers_samples)}")
other_names = Counter(s["name"] for s in SAMPLES if s["name"] != "buffers")
print(f"calls with other names: {dict(other_names)}")
print()

if not buffers_samples:
    print("no 'buffers' samples captured — exiting"); sys.exit(0)

# Stage breakdown
def avg(field):
    return sum(s[field] for s in buffers_samples) / len(buffers_samples)
walk = avg("walk_us"); sink = avg("sink_us"); subst = avg("subst_us"); asn = avg("assign_us")
total = walk + sink + subst + asn
print(f"=== STAGE breakdown (mean per call) ===")
print(f"  {'stage':<12} {'us':>10} {'%':>8}")
print(f"  {'walk':<12} {walk:>10.1f} {100*walk/total:>7.1f}%")
print(f"  {'sink':<12} {sink:>10.1f} {100*sink/total:>7.1f}%")
print(f"  {'substitute':<12} {subst:>10.1f} {100*subst/total:>7.1f}%")
print(f"  {'assign':<12} {asn:>10.1f} {100*asn/total:>7.1f}%")
print(f"  {'TOTAL':<12} {total:>10.1f}")
print()

# Tensor counts
print(f"=== Tensor counts (mean per call) ===")
print(f"  n_total (live tensors):  {avg('n_total'):.1f}")
print(f"  n_in_scope (visitor=True): {avg('n_in_scope'):.2f}")
print(f"  unique UOps in walk cache: {avg('dag_size_total'):.1f}")
print()

# Per-tensor walk-time histogram, grouped by op
buckets = [(0, 1), (1, 5), (5, 10), (10, 50), (50, 100), (100, 500), (500, float('inf'))]
hist_all = Counter()
op_time = defaultdict(float)
op_count = defaultdict(int)
inscope_count = 0
inscope_time = 0.0
deadweight_time = 0.0
deadweight_count = 0
slowest_per_call = []
for s in buffers_samples:
    slowest = None
    for rec in s["per_tensor"]:
        op, us, new_added, in_scope_result = rec[0], rec[1], rec[2], rec[3]
        bucket = next(label for (lo, hi), label in zip(buckets, [f"<{hi}us" if hi != float('inf') else f">={lo}us" for lo, hi in buckets]) if lo <= us < hi)
        hist_all[bucket] += 1
        op_time[op] += us
        op_count[op] += 1
        if in_scope_result:
            inscope_count += 1
            inscope_time += us
        else:
            deadweight_count += 1
            deadweight_time += us
        if slowest is None or us > slowest[1]:
            slowest = rec
    slowest_per_call.append(slowest)

print(f"=== Per-tensor walk-time histogram (all tensors, all calls combined) ===")
total_tensor_visits = sum(hist_all.values())
for bucket_label in [f"<{hi}us" if hi != float('inf') else f">={lo}us" for lo, hi in buckets]:
    n = hist_all.get(bucket_label, 0)
    bar = '#' * int(60 * n / max(total_tensor_visits, 1))
    print(f"  {bucket_label:<12} {n:>8}  {bar}")
print()

print(f"=== Top 10 op types by total walk time across all calls ===")
print(f"  {'op':<20} {'total_us':>12} {'count':>8} {'mean_us':>10} {'share':>8}")
total_walk_us = sum(op_time.values())
for op, t_us in sorted(op_time.items(), key=lambda x: -x[1])[:10]:
    cnt = op_count[op]
    print(f"  {op:<20} {t_us:>12.1f} {cnt:>8} {t_us/cnt:>10.2f} {100*t_us/total_walk_us:>7.1f}%")
print()

print(f"=== Dead-weight fraction (the 'wasted' work the killed gate aimed at) ===")
total_per_tensor = inscope_time + deadweight_time
print(f"  in_scope tensors:  {inscope_count} visits, {inscope_time:.0f} us total ({100*inscope_time/total_per_tensor:.1f}%)")
print(f"  dead-weight:       {deadweight_count} visits, {deadweight_time:.0f} us total ({100*deadweight_time/total_per_tensor:.1f}%)")
print(f"  per call: dead-weight = {deadweight_time/len(buffers_samples):.0f} us / {total_per_tensor/len(buffers_samples):.0f} us walk")
print(f"  (this is the upper bound on what a working skip-walk gate would save)")
print()

# Also sample DAG-newly-visited histogram
new_added_buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 100), (100, float('inf'))]
new_hist = Counter()
for s in buffers_samples:
    for rec in s["per_tensor"]:
        new_added = rec[2]
        bucket = next(label for (lo, hi), label in zip(new_added_buckets, [f"<{hi}" if hi != float('inf') else f">={lo}" for lo, hi in new_added_buckets]) if lo <= new_added < hi)
        new_hist[bucket] += 1
print(f"=== Slowest tensor per call (identifying the dominant cost) ===")
slow_shape_dtype = Counter()
for rec in slowest_per_call:
    op, us, new_added, in_scope_result, shape, dtype, dev = rec
    key = f"{op:<10} shape={shape} dtype={dtype} dev={dev} new_uops={new_added}"
    slow_shape_dtype[key] += 1
for key, count in sorted(slow_shape_dtype.items(), key=lambda x: -x[1])[:5]:
    print(f"  ({count}/{len(slowest_per_call)} calls)  {key}")
slow_us = [rec[1] for rec in slowest_per_call]
print(f"  slowest-tensor walk time: min={min(slow_us):.0f}us  median={sorted(slow_us)[len(slow_us)//2]:.0f}us  max={max(slow_us):.0f}us")
print()

print(f"=== Per-tensor 'new UOps added to in_scope' histogram ===")
print(f"(0 = entire DAG already cached from earlier tensors in this call)")
for bucket_label in [f"<{hi}" if hi != float('inf') else f">={lo}" for lo, hi in new_added_buckets]:
    n = new_hist.get(bucket_label, 0)
    bar = '#' * int(60 * n / max(sum(new_hist.values()), 1))
    print(f"  {bucket_label:<10} {n:>8}  {bar}")
