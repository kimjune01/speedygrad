"""iter 10c-cont v2: confirm CapturedJit.ret identity, decompose the AFTER chain,
categorize dead-weight tensors.

Iter 10c-cont's v1 probe found:
  - 73.6% of walk cost is ONE tensor per call (AFTER, shape=(2,) uint, +1613 UOps)
  - Identity is identical across all 50/50 calls
  - Hypothesis: this is CapturedJit.ret from tinygrad/engine/jit.py:289

This v2 probe answers:
  Q1: Is the dominant tensor actually `model.forward_jit.captured.ret`?
  Q2: What's in its 1613-UOp DAG? (Op-type histogram)
  Q3: Of the 167 dead-weight tensors per call, how do they break down?
       (model parameters, KV-cache slots, intermediate held-tensors, other)
  Q4: How many UNIQUE UOps are reachable from all live Tensors combined?
       (Tells us the upper bound on memoize-cache size)
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
from tinygrad.nn.state import get_state_dict

PHASE = ["load"]
SAMPLES = []
MODEL_PARAM_UOP_IDS: set = set()  # id(t.uop) for every parameter tensor — populated post-load
RET_TENSOR_ID = [None]  # id(model.forward_jit.captured.ret), populated after first JIT call

_orig_apply = _tensor_mod._apply_map_to_tensors

def _instrumented_apply(applied_map, name, walk=False):
    if PHASE[0] != "decode":
        return _orig_apply(applied_map, name, walk)

    record = {"name": name, "walk_us": 0.0, "n_total": 0, "n_in_scope": 0,
              "per_tensor": []}

    with cpu_profile(TracingKey(name), "TINY"):
        in_scope: dict = {}
        def visitor(node):
            return True if node in applied_map else any(in_scope.get(s, False) for s in node.src)

        scope_tensors = []
        for tref in list(_tensor_mod.all_tensors):
            t = tref()
            if t is None: continue
            t0 = time.perf_counter_ns()
            result = t.uop.topovisit(visitor, in_scope)
            t1 = time.perf_counter_ns()
            op_name = t.uop.op.name if hasattr(t.uop.op, 'name') else str(t.uop.op)
            try: shape = tuple(t.uop.shape)
            except Exception: shape = None
            try: dtype = str(t.uop.dtype)
            except Exception: dtype = None
            # categorize: model_param if t.uop is in our param-id set, ret_tensor if matches captured.ret,
            # else 'other'
            tid = id(t)
            tuop_id = id(t.uop)
            category = "other"
            if RET_TENSOR_ID[0] is not None and tid == RET_TENSOR_ID[0]:
                category = "captured_ret"
            elif tuop_id in MODEL_PARAM_UOP_IDS:
                category = "model_param"
            record["per_tensor"].append((op_name, (t1 - t0) / 1000.0, bool(result), shape, dtype, category, tid, tuop_id, t.uop))
            record["n_total"] += 1
            if result:
                scope_tensors.append(t)
                record["n_in_scope"] += 1

        sink = UOp.sink(*[t.uop for t in scope_tensors])
        new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)
        for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
            if s is ns: continue
            t.uop = ns

        record["walk_us"] = sum(rec[1] for rec in record["per_tensor"])
        record["in_scope_uop_size"] = len(in_scope)

    SAMPLES.append(record)

_tensor_mod._apply_map_to_tensors = _instrumented_apply

# --- run the bench ---
from examples.llama3 import build_transformer
from transformers import AutoTokenizer

model_path = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"
device = Device.DEFAULT

print("loading...", file=sys.stderr)
PHASE[0] = "load"
model = build_transformer(model_path, model_size="1B", quantize=None, device=device)

# capture parameter UOp ids — these are the model weight tensors
sd = get_state_dict(model)
for name, t in sd.items():
    if isinstance(t, Tensor):
        MODEL_PARAM_UOP_IDS.add(id(t.uop))
print(f"captured {len(MODEL_PARAM_UOP_IDS)} model-parameter uop ids", file=sys.stderr)

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

# JIT is now captured (cnt>=2). Snapshot the CapturedJit.ret tensor identity.
if model.forward_jit is not None and model.forward_jit.captured is not None:
    cap_ret = model.forward_jit.captured.ret
    if isinstance(cap_ret, Tensor):
        RET_TENSOR_ID[0] = id(cap_ret)
        print(f"captured_ret: id={RET_TENSOR_ID[0]} type={type(cap_ret).__name__} "
              f"uop.op={cap_ret.uop.op.name} uop.shape={cap_ret.uop.shape}", file=sys.stderr)
    else:
        print(f"captured_ret is not a Tensor (got {type(cap_ret).__name__}): {cap_ret!r}", file=sys.stderr)

PHASE[0] = "decode"
N = 50
print(f"sampling {N} steady-state decode tokens (instrumented inner v2)...", file=sys.stderr)
for _ in range(N):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# --- analyze ---
print()
print(f"=== iter 10c-cont v2: inner breakdown with categorization ===")
buffers_samples = [s for s in SAMPLES if s["name"] == "buffers"]
print(f"calls (decode steady, name='buffers'): {len(buffers_samples)}")
print(f"model-parameter uops captured: {len(MODEL_PARAM_UOP_IDS)}")
print(f"CapturedJit.ret tensor id captured: {RET_TENSOR_ID[0] is not None}")
print()

# Q1 + per-tensor identity-stability check on the slowest tensor
print(f"=== Q1: dominant-tensor identity (slowest tensor per call) ===")
slow_per_call = []
for s in buffers_samples:
    slow = max(s["per_tensor"], key=lambda r: r[1])
    slow_per_call.append(slow)
slow_ids = Counter()
for op, us, in_scope, shape, dtype, category, tid, tuop_id, _u in slow_per_call:
    slow_ids[(op, shape, dtype, category, tid)] += 1
print(f"  slowest-tensor identity appearances (top 5):")
for (op, shape, dtype, category, tid), count in sorted(slow_ids.items(), key=lambda x: -x[1])[:5]:
    print(f"    {count}/{len(slow_per_call)} calls  op={op} shape={shape} dtype={dtype} category={category} tensor_id={tid}")
slow_us = [r[1] for r in slow_per_call]
print(f"  slowest-tensor walk-time: min={min(slow_us):.0f}us  median={sorted(slow_us)[len(slow_us)//2]:.0f}us  max={max(slow_us):.0f}us")
print()

# Q2: AFTER chain decomposition for the dominant tensor (use one sample)
print(f"=== Q2: dominant-tensor UOp DAG decomposition (Op-type histogram) ===")
sample = slow_per_call[0]
dominant_uop = sample[8]  # the actual UOp
all_uops = list(dominant_uop.toposort().keys())
op_hist = Counter(u.op.name for u in all_uops)
print(f"  total UOps in DAG: {len(all_uops)}")
print(f"  top 15 op types:")
for op, cnt in sorted(op_hist.items(), key=lambda x: -x[1])[:15]:
    bar = '#' * int(40 * cnt / len(all_uops))
    print(f"    {op:<20} {cnt:>5}  {bar}")
# DAG depth: longest path from root to leaf
def dag_depth(u, cache=None):
    if cache is None: cache = {}
    if u in cache: return cache[u]
    if not u.src: cache[u] = 1
    else: cache[u] = 1 + max(dag_depth(s, cache) for s in u.src)
    return cache[u]
try:
    depth = dag_depth(dominant_uop)
    print(f"  DAG depth (longest path): {depth}")
except RecursionError:
    print(f"  DAG depth: RecursionError (very deep)")
print()

# Q3: dead-weight categorization
print(f"=== Q3: per-tensor walk time, grouped by category × op (decode steady) ===")
cat_op_time = defaultdict(float)
cat_op_count = defaultdict(int)
for s in buffers_samples:
    for op, us, in_scope, shape, dtype, category, tid, tuop_id, _u in s["per_tensor"]:
        cat_op_time[(category, op)] += us
        cat_op_count[(category, op)] += 1
total_walk_us = sum(cat_op_time.values())
print(f"  {'category':<14} {'op':<14} {'total_us':>12} {'count':>8} {'mean_us':>10} {'share':>8}")
for (cat, op), tt in sorted(cat_op_time.items(), key=lambda x: -x[1])[:20]:
    cnt = cat_op_count[(cat, op)]
    print(f"  {cat:<14} {op:<14} {tt:>12.1f} {cnt:>8} {tt/cnt:>10.2f} {100*tt/total_walk_us:>7.1f}%")
print()

# Q4: cache size estimate
print(f"=== Q4: cache size for memoize-walk ===")
in_scope_sizes = [s["in_scope_uop_size"] for s in buffers_samples]
print(f"  unique UOps reachable from all live tensors per call: "
      f"min={min(in_scope_sizes)} median={sorted(in_scope_sizes)[len(in_scope_sizes)//2]} max={max(in_scope_sizes)}")
print(f"  (this is the upper bound on cache size for the toposort-set memoize approach;")
print(f"   model-weight UOps stay constant across calls, only the per-iteration input adds new entries)")
print()

# Bonus: which model-parameter uops are walked? Should be ALL of them every call.
mp_walked = set()
for s in buffers_samples:
    for op, us, in_scope, shape, dtype, category, tid, tuop_id, _u in s["per_tensor"]:
        if category == "model_param":
            mp_walked.add(tuop_id)
print(f"  model-parameter uops actually visited in walk: {len(mp_walked)} / {len(MODEL_PARAM_UOP_IDS)}")
mp_total_us = sum(us for s in buffers_samples for op, us, _, _, _, cat, _, _, _ in s["per_tensor"] if cat == "model_param")
print(f"  total time spent walking model-parameter tensors: {mp_total_us:.0f} us across {len(buffers_samples)} calls = "
      f"{mp_total_us/len(buffers_samples):.1f} us/call")
captured_ret_total_us = sum(us for s in buffers_samples for op, us, _, _, _, cat, _, _, _ in s["per_tensor"] if cat == "captured_ret")
captured_ret_count = sum(1 for s in buffers_samples for op, us, _, _, _, cat, _, _, _ in s["per_tensor"] if cat == "captured_ret")
print(f"  total time spent walking captured_ret tensor: {captured_ret_total_us:.0f} us across {captured_ret_count} appearances "
      f"= {captured_ret_total_us/max(captured_ret_count,1):.1f} us/call")
other_total_us = sum(us for s in buffers_samples for op, us, _, _, _, cat, _, _, _ in s["per_tensor"] if cat == "other")
other_count = sum(1 for s in buffers_samples for op, us, _, _, _, cat, _, _, _ in s["per_tensor"] if cat == "other")
print(f"  total time spent walking 'other' tensors: {other_total_us:.0f} us across {other_count} appearances "
      f"= {other_total_us/len(buffers_samples):.1f} us/call ({other_count/len(buffers_samples):.1f} other/call)")
