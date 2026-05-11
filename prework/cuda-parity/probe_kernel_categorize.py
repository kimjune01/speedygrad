"""Categorize speedygrad's per-token GPU kernel cost by what it's
computing (matmul, attention, RMSNorm, RoPE, etc.).

Iter 10c-cont v3: post-fix nsys shows 5200us GPU per decode token vs
llama.cpp's ~3700us = 1500us gap. Need to know which categories of
kernels contribute most to that gap.

Strategy: hook into tinygrad's PROGRAM UOps (which carry the kernel
name and the AST that produced them) and tag each kernel with what
operation it represents. Then runtime accounting on a steady-state
decode loop.

Method: monkey-patch run_linear / run_program to collect (kernel_name,
elapsed_ns, ast_summary) per call. Run 50 decode tokens. Aggregate
by kernel signature, compute mean/median per call.

Then compare to llama.cpp categories from sg_kern_node.csv:
  - matmul: mul_mat_vec_f<...>
  - rmsnorm: rms_norm_f32
  - rope: rope_norm
  - softmax: soft_max_f32
  - kv-cache write: k_set_rows
"""
import os, sys, time
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device
import tinygrad.engine.realize as _realize
from tinygrad.uop.ops import UOp, Ops

# Capture per-CALL info via instrumented run_linear
KERNEL_TIMES_NS = defaultdict(list)  # kernel_name -> [ns, ns, ...]
KERNEL_AST_SHAPE = {}  # kernel_name -> shape signature for categorization
RECORDING = [False]

_orig_run_linear = _realize.run_linear

def _instrumented_run_linear(linear, var_vals=None, input_uops=None, jit=False, do_update_stats=True):
    # The "name" in the linear's PROGRAM UOps doesn't easily come back; instead
    # we time the whole linear call and bucket by linear.src signature.
    if not RECORDING[0]:
        return _orig_run_linear(linear, var_vals, input_uops, jit=jit, do_update_stats=do_update_stats)
    # Note: this measures host+launch, not pure GPU. nsys gives GPU. Use this
    # for per-call distribution / "how often is which kernel called" rather
    # than pure GPU kernel time (we already have that from nsys).
    n_kernels = sum(1 for s in linear.src if s.src and s.src[0].op is Ops.PROGRAM)
    t0 = time.perf_counter_ns()
    ret = _orig_run_linear(linear, var_vals, input_uops, jit=jit, do_update_stats=do_update_stats)
    dt = time.perf_counter_ns() - t0
    KERNEL_TIMES_NS[f"linear_{n_kernels}_kernels"].append(dt)
    return ret

_realize.run_linear = _instrumented_run_linear
# Also rebind in jit module if it imported it directly
import tinygrad.engine.jit as _jit
_jit.run_linear = _instrumented_run_linear
import tinygrad.tensor as _tensor_mod
_tensor_mod.run_linear = _instrumented_run_linear

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

# warmup
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# Identify the captured JIT linear and dump its kernel signatures
if model.forward_jit is not None and model.forward_jit.captured is not None:
    cap = model.forward_jit.captured
    linear = cap.linear
    print(f"\n=== captured JIT linear has {len(linear.src)} src calls ===")
    op_signature = Counter()
    program_calls = []
    for call in linear.src:
        if not call.src: continue
        head = call.src[0]
        if head.op is Ops.PROGRAM:
            # PROGRAM args usually carry the kernel name as arg.name
            try:
                kname = head.arg.name if hasattr(head.arg, 'name') else str(head.arg)[:60]
            except Exception:
                kname = "?"
            program_calls.append(kname)
            op_signature[kname] += 1
        elif head.op is Ops.CUSTOM_FUNCTION:
            op_signature[f"CUSTOM_FUNCTION:{head.arg}"] += 1
        else:
            op_signature[f"OTHER_OP:{head.op.name}"] += 1
    print(f"unique kernel signatures in captured graph: {len(op_signature)}")
    print(f"top 20 kernels (by count in captured graph):")
    for kname, cnt in sorted(op_signature.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cnt:>4} x  {kname[:80]}")

print(f"\n=== run 50 steady-state decode tokens, recording linear-call times ===")
RECORDING[0] = True
times_us = []
for _ in range(50):
    t0 = time.perf_counter()
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    t1 = time.perf_counter()
    times_us.append((t1 - t0) * 1e6)
    start_pos += 1
RECORDING[0] = False

times_us.sort()
print(f"  decode_p50={times_us[25]:.0f}us  p10={times_us[5]:.0f}us  p90={times_us[45]:.0f}us")

print(f"\n=== run_linear calls observed during decode ===")
for sig, ns_list in sorted(KERNEL_TIMES_NS.items(), key=lambda x: -sum(x[1])):
    n = len(ns_list)
    total_us = sum(ns_list) / 1000.0
    mean_us = total_us / n
    median_us = sorted(ns_list)[len(ns_list)//2] / 1000.0
    print(f"  {sig:<30} count={n:>4}  total={total_us:>9.0f}us  mean={mean_us:>7.1f}us  median={median_us:>7.1f}us")
