"""Identify what the dominant 'other' AFTER shape=(2,) uint tensor IS.

The v2 probe found: across 50/50 decode calls, the slowest tensor in
_apply_map_to_tensors is the SAME object (tensor_id stable) — but it's
not CapturedJit.ret. It has 1615 UOps in its DAG including 114 STOREs
and 114 AFTERs. Need to identify it.

Strategy:
1. Run prefill + decode_burn to reach steady state.
2. Walk all_tensors, find the one with the deepest AFTER chain (shape=(2,) uint).
3. Print its identity diagnostics:
   - id(t)
   - id check against named attributes of model.forward_jit
   - id check against Transformer/TransformerBlock/Attention attributes (look for cache_kv etc.)
   - BUFFER leaves of its UOp DAG: shapes and dtypes
   - count of distinct UOps by op type
"""
import os, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops

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

# prefill + burn
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# Find the dominant tensor
print(f"\n=== scanning all_tensors for AFTER tensors with deep DAGs ===", file=sys.stderr)
candidates = []
for tref in list(_tensor_mod.all_tensors):
    t = tref()
    if t is None: continue
    if t.uop.op is not Ops.AFTER: continue
    try: shape = tuple(t.uop.shape)
    except Exception: shape = None
    n_uops = len(t.uop.toposort())
    candidates.append((id(t), t, shape, n_uops))

candidates.sort(key=lambda c: -c[3])  # by uop count, descending
print(f"top 10 AFTER tensors by DAG size:")
for tid, t, shape, n_uops in candidates[:10]:
    dtype = str(t.uop.dtype)
    print(f"  id={tid} shape={shape} dtype={dtype} n_uops={n_uops}")

# Identify the dominant one
print(f"\n=== identifying the dominant tensor ===")
# heuristic: look for shape=(2,) uint
dominant = None
for tid, t, shape, n_uops in candidates:
    if shape == (2,) and 'uint' in str(t.uop.dtype):
        dominant = (tid, t, shape, n_uops)
        break
if dominant is None:
    print("no shape=(2,) uint AFTER tensor found among candidates; using deepest:")
    dominant = candidates[0]

tid, t, shape, n_uops = dominant
print(f"DOMINANT: id={tid} shape={shape} dtype={t.uop.dtype} n_uops={n_uops}")
print(f"           op={t.uop.op.name}")
print(f"           __class__={t.__class__.__name__}")
print(f"           device={t.uop.device}")

# Walk the DAG, find BUFFER leaves and their shapes
print(f"\n=== BUFFER leaves in dominant tensor's UOp DAG ===")
all_uops = list(t.uop.toposort().keys())
buffers = [u for u in all_uops if u.op is Ops.BUFFER]
print(f"  {len(buffers)} BUFFER UOps. First 20 by shape:")
buf_shape_counts = Counter()
for b in buffers:
    try: bshape = tuple(b.shape)
    except Exception: bshape = None
    buf_shape_counts[(bshape, str(b.dtype), str(b.device) if hasattr(b, 'device') else '?')] += 1
for (bshape, bdtype, bdev), cnt in sorted(buf_shape_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"    {cnt:>4} x  shape={bshape} dtype={bdtype} dev={bdev}")
print()

# Search for this tensor's identity in model attributes
print(f"=== identity search: which model attribute is this tensor? ===")
def search(obj, path, seen, max_depth=4):
    if max_depth <= 0 or id(obj) in seen: return
    seen.add(id(obj))
    if isinstance(obj, Tensor):
        if id(obj) == tid:
            print(f"  FOUND at: {path}")
        return
    if isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            search(x, f"{path}[{i}]", seen, max_depth-1)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            search(v, f"{path}[{k!r}]", seen, max_depth-1)
        return
    if hasattr(obj, '__dict__'):
        for attr_name, val in vars(obj).items():
            if attr_name.startswith('_'): continue
            search(val, f"{path}.{attr_name}", seen, max_depth-1)

search(model, "model", set(), max_depth=6)

# Also check sample's function attributes
import extra.models.llama as llama_mod
print(f"  sample function attrs: {[a for a in dir(llama_mod.sample) if not a.startswith('_')]}")
for attr in dir(llama_mod.sample):
    if attr.startswith('_'): continue
    val = getattr(llama_mod.sample, attr)
    if isinstance(val, Tensor) and id(val) == tid:
        print(f"  FOUND at: extra.models.llama.sample.{attr}")

# Check the JIT's CapturedJit
if model.forward_jit is not None and model.forward_jit.captured is not None:
    cap = model.forward_jit.captured
    if id(cap.ret) == tid:
        print(f"  FOUND at: model.forward_jit.captured.ret")

print(f"\n  if not found: this tensor exists in all_tensors but is not reachable from model/sample.")
print(f"  candidates: a closure variable in TinyJit, a free intermediate held by something else,")
print(f"  or a Tensor created inside a pattern matcher / rewrite that survived.")
