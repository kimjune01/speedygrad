"""Minimal nsys-friendly Qwen 3 8B Q4_K_M decode runner.

Just enough to capture ~10 steady-state decode tokens for kernel attribution.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa
from tinygrad import Tensor, Device
from tinygrad.helpers import fetch
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import models, SimpleTokenizer

device = Device.DEFAULT
print(f"loading qwen3:8b...", file=sys.stderr)
gguf_path = fetch(models["qwen3:8b"])
model, kv = Transformer.from_gguf(gguf_path, max_context=4096)
tok = SimpleTokenizer.from_gguf_kv(kv)

ids = tok.role("user") + tok.encode("Hi.") + tok.end_turn() + tok.role("assistant")
gen = model.generate(list(ids), temperature=0.0)

# burn 3 tokens (TTFT + JIT capture)
print(f"burning 3 tokens (compile + capture)...", file=sys.stderr)
for _ in range(3): next(gen)

# 10 steady-state decode tokens — this is what nsys should focus on
print(f"steady-state decode (10 tokens, profiled)...", file=sys.stderr)
for i in range(10):
    next(gen)
print(f"done", file=sys.stderr)
