"""Quick smoke test for Qwen 3 8B Q4_K_M on speedygrad.

Reproduces the 1 tok/s decode regression and confirms basic facts:
- Did the model load successfully?
- Are weights actually on GPU (not falling back to CPU)?
- Per-decode-token wall-clock distribution
- _apply_map_to_tensors host cost (per session's earlier instrumentation pattern)
- GPU memory used vs available
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa: F401
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import fetch
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import models, SimpleTokenizer
import tinygrad.tensor as _tensor_mod

# instrument _apply_map_to_tensors
APPLY_TOTAL_NS = [0]
APPLY_CALLS = [0]
RECORDING = [False]
_orig_apply = _tensor_mod._apply_map_to_tensors
def _instrumented_apply(applied_map, name, walk=False):
    if not RECORDING[0]: return _orig_apply(applied_map, name, walk)
    t0 = time.perf_counter_ns()
    try: return _orig_apply(applied_map, name, walk)
    finally:
        APPLY_TOTAL_NS[0] += time.perf_counter_ns() - t0
        APPLY_CALLS[0] += 1
_tensor_mod._apply_map_to_tensors = _instrumented_apply

device = Device.DEFAULT
print(f"device: {device}", file=sys.stderr)

# check VRAM before load
try:
    from tinygrad.runtime.autogen import cuda
    free = (cuda.c_size_t * 1)(0)
    total = (cuda.c_size_t * 1)(0)
    cuda.cuMemGetInfo_v2(free, total)
    print(f"VRAM before load: free={free[0]/1e9:.2f}GB / total={total[0]/1e9:.2f}GB", file=sys.stderr)
except Exception as e:
    print(f"could not query VRAM: {e}", file=sys.stderr)

print(f"loading qwen3:8b...", file=sys.stderr)
gguf_path = fetch(models["qwen3:8b"])
print(f"  GGUF size: {gguf_path.stat().st_size/1e9:.2f}GB at {gguf_path}", file=sys.stderr)

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(gguf_path, max_context=4096)
load_s = time.perf_counter() - t0
print(f"  loaded in {load_s:.1f}s", file=sys.stderr)

# check VRAM after load
try:
    cuda.cuMemGetInfo_v2(free, total)
    print(f"VRAM after load: free={free[0]/1e9:.2f}GB / total={total[0]/1e9:.2f}GB "
          f"(used during load: {(total[0]-free[0])/1e9:.2f}GB)", file=sys.stderr)
except Exception as e:
    print(f"could not query VRAM: {e}", file=sys.stderr)

# count parameters
n_params = sum(x.numel() for x in __import__('tinygrad').nn.state.get_parameters(model))
print(f"  {n_params:,} params (~{n_params*0.5/1e9:.1f}GB at Q4_K_M weights)", file=sys.stderr)

# devices of all parameter buffers
params = __import__('tinygrad').nn.state.get_parameters(model)
device_counts = {}
for p in params:
    d = str(p.device)
    device_counts[d] = device_counts.get(d, 0) + 1
print(f"  parameter devices: {device_counts}", file=sys.stderr)

tok = SimpleTokenizer.from_gguf_kv(kv)

# tiny prompt
ids = tok.role("user") + tok.encode("Hi.") + tok.end_turn() + tok.role("assistant")
print(f"  prompt_len: {len(ids)}", file=sys.stderr)

print(f"\nstarting generate(), timing each token...", file=sys.stderr)
gen = model.generate(list(ids), temperature=0.0)

# first token = prefill + first decode
t0 = time.perf_counter()
first_tok = next(gen)
ttft_s = time.perf_counter() - t0
print(f"  TTFT (prefill + first decode): {ttft_s:.1f}s", file=sys.stderr)

# next 10 tokens, time each
RECORDING[0] = True
times_us = []
APPLY_TOTAL_NS[0] = 0
APPLY_CALLS[0] = 0
for i in range(10):
    t0 = time.perf_counter()
    next_tok = next(gen)
    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1000
    times_us.append(dt_ms * 1000)
    print(f"  token {i+1}: {dt_ms:.0f}ms  tok={next_tok}  ({tok.decode([next_tok])!r})", file=sys.stderr)
RECORDING[0] = False

times_us.sort()
mid = len(times_us) // 2
print(f"\nDecode summary (10 tokens):", file=sys.stderr)
print(f"  p50: {times_us[mid]/1000:.1f}ms ({1e6/times_us[mid]:.1f} tok/s)", file=sys.stderr)
print(f"  min: {times_us[0]/1000:.1f}ms, max: {times_us[-1]/1000:.1f}ms", file=sys.stderr)
print(f"  _apply_map_to_tensors: {APPLY_CALLS[0]} calls, "
      f"{APPLY_TOTAL_NS[0]/1000.0:.0f}us total = {APPLY_TOTAL_NS[0]/APPLY_CALLS[0]/1000.0:.1f}us/call",
      file=sys.stderr)
print(f"  apply share of decode wall: {100*APPLY_TOTAL_NS[0]/1000.0/sum(times_us):.1f}%", file=sys.stderr)
