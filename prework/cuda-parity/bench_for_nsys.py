"""Minimal bench wrapper for nsys profiling (post counter.realize() fix).

Just enough to generate ~50 steady-state decode tokens with the iter
10c-cont v2 fix applied, no Python overhead from JSON/sorting/etc.
nsys captures the whole process, but we want the bulk of the
recorded time to be steady-state decode so the per-kernel averages
are representative.
"""
import os, sys
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

# iter 10c-cont v2 fix
for _counter in Tensor._device_rng_counters.values():
  _counter.realize()

tok = AutoTokenizer.from_pretrained(str(model_path))
prompt = tok.apply_chat_template([{"role":"user","content":"Hi."}],
                                  add_generation_prompt=True, tokenize=False)
toks = tok.encode(prompt, add_special_tokens=False)

# prefill
start_pos = 0
for t in toks[:-1]:
    model(Tensor([[t]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
last = toks[-1]

# burn
for _ in range(5):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

# 50 steady-state decode tokens (this is what we want nsys to focus on)
print("starting steady-state decode (50 tokens, instrumented)", file=sys.stderr)
for _ in range(50):
    last = model(Tensor([[last]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
    start_pos += 1

print(f"done; {start_pos - len(toks)} decode tokens", file=sys.stderr)
