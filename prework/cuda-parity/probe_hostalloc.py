"""iter 10b probe: count cuMemHostAlloc per phase, capture caller stacks.

Wraps cuMemHostAlloc to log size + traceback for the first few calls in each phase
(load, prefill, decode), and tallies counts to identify which phase drives the hot loop.
"""
import os, sys, traceback
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
import tinygrad.runtime.autogen.cuda as cuda

PHASE = ["load"]
CALLS = Counter()
SIZES = Counter()
SAMPLE_STACKS = {"load": [], "prefill": [], "decode": []}
SAMPLE_LIMIT = 3

_orig = cuda.cuMemHostAlloc
def wrapped(pp, bytesize, flags):
    p = PHASE[0]
    CALLS[p] += 1
    SIZES[(p, int(bytesize))] += 1
    if len(SAMPLE_STACKS[p]) < SAMPLE_LIMIT:
        stack = "".join(traceback.format_stack(limit=20))
        SAMPLE_STACKS[p].append((int(bytesize), stack))
    return _orig(pp, bytesize, flags)
cuda.cuMemHostAlloc = wrapped

from tinygrad import Tensor, Device
from examples.llama3 import build_transformer
from transformers import AutoTokenizer

model_path = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"
device = Device.DEFAULT

print("=== load ===", file=sys.stderr)
model = build_transformer(model_path, model_size="1B", quantize=None, device=device)
tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)
print(f"load: cuMemHostAlloc calls = {CALLS['load']}", file=sys.stderr)

PHASE[0] = "prefill"
print("=== prefill ===", file=sys.stderr)
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
print(f"prefill: cuMemHostAlloc calls = {CALLS['prefill']} for {len(toks)-1} tokens", file=sys.stderr)

PHASE[0] = "decode"
N_DECODE = 50
print(f"=== decode ({N_DECODE} tokens) ===", file=sys.stderr)
last = toks[-1]
per_token_cuda_calls = []
prev_count = CALLS["decode"]
for i in range(N_DECODE):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1
    new = CALLS["decode"]
    if new > prev_count:
        per_token_cuda_calls.append((i, new - prev_count))
        prev_count = new
print(f"decode: cuMemHostAlloc calls = {CALLS['decode']} for {N_DECODE} tokens (~{CALLS['decode']/N_DECODE:.3f}/tok)", file=sys.stderr)
print(f"decode tokens that allocated: {per_token_cuda_calls}", file=sys.stderr)

print("\n=== size histogram ===", file=sys.stderr)
for (p, sz), n in sorted(SIZES.items(), key=lambda x: -x[1])[:20]:
    print(f"  phase={p:8s} size={sz:>10d}  count={n}", file=sys.stderr)

print("\n=== sample stacks (first few per phase) ===", file=sys.stderr)
for phase, samples in SAMPLE_STACKS.items():
    for sz, stack in samples:
        print(f"\n--- phase={phase}  size={sz} ---", file=sys.stderr)
        print(stack, file=sys.stderr)
