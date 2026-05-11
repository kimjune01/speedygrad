"""Dump every PROGRAM kernel in the captured JIT linear, with its name,
shape signature, and a brief description, so we can categorize each.

Iter 10c-cont v3: post-fix nsys gives per-kernel GPU times (in
prework/cuda-parity/sg3_kern_node.csv) but the names like
r_512_16_512_512_4_4 are opaque. To compare against llama.cpp by
operation type, need to know which kernel is matmul vs RoPE vs RMSNorm
vs attention vs softmax.

Walks the captured.linear UOp graph recursively (through CUSTOM_FUNCTION
"graph" wrappers) to find every PROGRAM, prints (kernel_name, output
shape, input shapes, render type if PTX). Match against nsys names by
the 'name' field of the PROGRAM's arg.
"""
import os, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins

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

# warmup to capture
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
for _ in range(3):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

cap = model.forward_jit.captured
linear = cap.linear
print(f"\n=== captured linear ===")
print(f"linear.src count: {len(linear.src)}")

# Walk all CALLs (graphs and individual programs)
def find_programs(u, depth=0, found=None):
    if found is None: found = []
    if u.op is Ops.PROGRAM:
        found.append(u)
        return found
    for s in u.src:
        find_programs(s, depth+1, found)
    return found

# Each call in linear.src wraps either a single PROGRAM or a CUSTOM_FUNCTION
# (with name "graph") that contains a sub-LINEAR with multiple PROGRAMs.
all_programs = []
top_level_count = Counter()
for call in linear.src:
    head = call.src[0]
    top_level_count[head.op.name] += 1
    progs = find_programs(call)
    all_programs.extend(progs)

print(f"top-level call types in linear.src: {dict(top_level_count)}")
print(f"total PROGRAMs found (recursively): {len(all_programs)}")

# Group by kernel name
by_name = Counter()
shape_by_name = {}
for p in all_programs:
    try:
        kname = p.arg.name if hasattr(p.arg, 'name') else str(p.arg)[:40]
    except Exception:
        kname = "?"
    by_name[kname] += 1
    if kname not in shape_by_name:
        # capture global/local size for this kernel from the PROGRAM arg
        try:
            gs = tuple(p.arg.global_size) if hasattr(p.arg, 'global_size') else None
            ls = tuple(p.arg.local_size) if hasattr(p.arg, 'local_size') else None
        except Exception:
            gs, ls = None, None
        shape_by_name[kname] = (gs, ls)

print(f"\n=== unique kernels ({len(by_name)}) ===")
print(f"  {'count':>5} {'global':<24} {'local':<16} kernel_name")
for kname, cnt in sorted(by_name.items(), key=lambda x: -x[1]):
    gs, ls = shape_by_name.get(kname, (None, None))
    print(f"  {cnt:>5} {str(gs):<24} {str(ls):<16} {kname}")

# Also dump: for each PROGRAM, what are its input/output shapes and dtypes?
# This helps identify "matmul of (1,2048) x (2048, 2048)" vs "rmsnorm of (1,2048)".
print(f"\n=== first occurrence per kernel: input/output shapes ===")
seen = set()
for call in linear.src:
    head = call.src[0]
    if head.op is Ops.CUSTOM_FUNCTION:
        # look inside the CUSTOM_FUNCTION's body for PROGRAMs
        for sub in head.src:
            if sub.op is Ops.LINEAR:
                for sub_call in sub.src:
                    sh = sub_call.src[0]
                    if sh.op is Ops.PROGRAM:
                        kname = sh.arg.name if hasattr(sh.arg, 'name') else "?"
                        if kname in seen: continue
                        seen.add(kname)
                        try:
                            args = get_call_arg_uops(sub_call)
                            outs, ins = get_call_outs_ins(sub_call)
                            in_shapes = [tuple(args[i].shape) if i < len(args) else None for i in ins[:4]]
                            out_shapes = [tuple(args[i].shape) if i < len(args) else None for i in outs[:2]]
                            in_dtypes = [str(args[i].dtype) for i in ins[:4] if i < len(args)]
                            print(f"  {kname:<35} ins={in_shapes}  outs={out_shapes}")
                        except Exception as e:
                            print(f"  {kname:<35} (failed to extract: {e})")
    elif head.op is Ops.PROGRAM:
        kname = head.arg.name if hasattr(head.arg, 'name') else "?"
        if kname in seen: continue
        seen.add(kname)
        try:
            args = get_call_arg_uops(call)
            outs, ins = get_call_outs_ins(call)
            in_shapes = [tuple(args[i].shape) if i < len(args) else None for i in ins[:4]]
            out_shapes = [tuple(args[i].shape) if i < len(args) else None for i in outs[:2]]
            print(f"  {kname:<35} ins={in_shapes}  outs={out_shapes}")
        except Exception as e:
            print(f"  {kname:<35} (failed to extract: {e})")
