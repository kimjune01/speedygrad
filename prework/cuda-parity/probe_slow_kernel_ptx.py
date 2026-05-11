"""H₃: Dump PTX source for the slowest element-wise kernel in Qwen 8B's
captured graph.

iter 11c diagnosed E_24576_4_2_2_16_8 at 18ms mean as the most-suspect
single finding (element-wise op should be sub-ms). This probe walks the
captured graph, finds it (or similar slow element-wise kernels), and
dumps the generated PTX.
"""
import os, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import monkeypatch  # noqa
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops
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

# burn 3 tokens to ensure JIT capture is complete
print(f"burning 3 tokens to capture rollout JIT...", file=sys.stderr)
for _ in range(3): next(gen)

# Walk the captured rollout JIT's linear, find all PROGRAM kernels
captured = model.rollout_jit.captured
if captured is None:
    print(f"ERROR: rollout JIT not captured", file=sys.stderr)
    sys.exit(1)

linear = captured.linear

def walk_programs(u, found=None):
    if found is None: found = []
    if u.op is Ops.PROGRAM:
        found.append(u)
    for s in u.src:
        walk_programs(s, found)
    return found

all_programs = []
for call in linear.src:
    all_programs.extend(walk_programs(call))

print(f"\nfound {len(all_programs)} PROGRAM kernels in captured rollout graph", file=sys.stderr)

# First, dump prefix counts to understand what naming scheme is in use
prefix_counts = Counter()
for p in all_programs:
    try:
        name = p.arg.name if hasattr(p.arg, 'name') else "?"
        prefix_counts[name.split("_")[0]] += 1
    except Exception:
        prefix_counts["?"] += 1
print(f"\nname prefixes:")
for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
    print(f"  {prefix:<10} {count}")
# Print 10 example names
print(f"\nfirst 15 unique kernel names:")
seen_names = set()
for p in all_programs:
    try:
        name = p.arg.name if hasattr(p.arg, 'name') else "?"
        if name not in seen_names:
            seen_names.add(name)
            if len(seen_names) > 15: break
            print(f"  {name}")
    except Exception: pass

# Identify slow element-wise kernels (E_* prefix in name)
elementwise = []
for p in all_programs:
    try:
        name = p.arg.name if hasattr(p.arg, 'name') else "?"
        if name.startswith("E"):
            elementwise.append((name, p))
    except Exception: pass

# Group by unique kernel name
by_name = {}
for name, p in elementwise:
    if name not in by_name:
        by_name[name] = (p, 1)
    else:
        by_name[name] = (by_name[name][0], by_name[name][1] + 1)

print(f"\n{len(by_name)} unique element-wise kernel names in captured graph:")
for name, (p, count) in sorted(by_name.items()):
    try:
        gs = tuple(p.arg.global_size) if hasattr(p.arg, 'global_size') else None
        ls = tuple(p.arg.local_size) if hasattr(p.arg, 'local_size') else None
    except Exception:
        gs, ls = None, None
    print(f"  {count:>4}x  {name:<35} global={gs} local={ls}")

# Compute per-call shape size (product of global_size * local_size threads)
# and identify the heaviest rollout kernels by dim
import re
ANSI = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(s): return ANSI.sub('', s)

print(f"\n--- ALL unique rollout kernels by dim, with sources ---")
all_unique = {}
for p in all_programs:
    try:
        name = strip_ansi(p.arg.name) if hasattr(p.arg, 'name') else "?"
    except Exception: continue
    if name not in all_unique:
        all_unique[name] = (p, 1)
    else:
        all_unique[name] = (all_unique[name][0], all_unique[name][1] + 1)

def gs_size(p):
    try:
        gs = p.arg.global_size
        # may contain UOps; convert to int where possible
        total = 1
        for x in gs:
            if isinstance(x, int): total *= x
            elif hasattr(x, 'arg') and isinstance(x.arg, int): total *= x.arg
            else: total *= 100  # guess for symbolic
        return total
    except Exception: return 0

# Sort by approximate work size = global*local size
ranked = sorted(all_unique.items(), key=lambda kv: -gs_size(kv[1][0]))
print(f"top 5 rollout kernels by global-size product:")
for name, (p, count) in ranked[:5]:
    try:
        gs = p.arg.global_size
        ls = p.arg.local_size
        print(f"\n=== {name}  count={count}x  global={gs}  local={ls} ===")
        src = p.arg.src
        if isinstance(src, bytes): src = src.decode('utf-8', errors='replace')
        # show first 60 lines
        lines = src.splitlines()
        for line in lines[:60]:
            print(f"  {line}")
        if len(lines) > 60:
            print(f"  [... {len(lines)} total lines]")
    except Exception as e:
        print(f"  source dump failed: {e}")
import sys; sys.exit(0)

# Try to find E_24576_4_2_2_16_8 specifically (or the most-similar)
target_candidates = [
    "E_24576_4_2_2_16_8",
    "E_12288_2_2_2_2_16_16",
]
print(f"\n--- looking for known-slow kernels ---")
for target in target_candidates:
    if target in by_name:
        p, count = by_name[target]
        print(f"\n=== {target} (appears {count}x in captured graph) ===")
        print(f"  global: {p.arg.global_size}")
        print(f"  local:  {p.arg.local_size}")
        # dump PTX source
        try:
            src = p.arg.src
            print(f"\n--- PTX source ({len(src)} bytes) ---")
            # decode if bytes
            if isinstance(src, bytes):
                src = src.decode('utf-8', errors='replace')
            print(src[:4000])
            if len(src) > 4000:
                print(f"\n[... truncated, total {len(src)} bytes ...]")
        except Exception as e:
            print(f"  (could not extract source: {e})")
        break  # just dump the first match
else:
    # neither found; dump the largest element-wise kernel by global_size
    print(f"\nneither target kernel found; dumping the largest by global size:")
    largest = max(by_name.values(), key=lambda v: max(v[0].arg.global_size if hasattr(v[0].arg, 'global_size') else (1,)))
    p, count = largest
    name = p.arg.name
    print(f"\n=== {name} (largest, appears {count}x) ===")
    print(f"  global: {p.arg.global_size}")
    print(f"  local:  {p.arg.local_size}")
    try:
        src = p.arg.src
        if isinstance(src, bytes): src = src.decode('utf-8', errors='replace')
        print(f"\n--- source ({len(src)} bytes) ---")
        print(src[:4000])
    except Exception as e:
        print(f"  (could not extract source: {e})")
