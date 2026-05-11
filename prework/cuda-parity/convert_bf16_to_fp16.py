"""
One-time bf16 -> fp16 conversion for unsloth/Llama-3.2-1B-Instruct.

Speedygrad's PTXRenderer has no bf16 support. We cast on disk via torch+safetensors
and write a sibling dir that has the same config but fp16 weights.

Reads:  ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/<sha>/
Writes: ~/.cache/llama32-1b-fp16/
"""
import os, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

SRC_BASE = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" / "models--unsloth--Llama-3.2-1B-Instruct" / "snapshots"
SRC = next(SRC_BASE.iterdir())  # one snapshot dir
DST = Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"

print(f"src: {SRC}")
print(f"dst: {DST}")
DST.mkdir(parents=True, exist_ok=True)

# Convert weights
weights = load_file(str(SRC / "model.safetensors"))
fp16 = {k: v.to(torch.float16).contiguous() for k, v in weights.items()}
print(f"converted {len(fp16)} tensors")
save_file(fp16, str(DST / "model.safetensors"))
print(f"wrote {(DST / 'model.safetensors').stat().st_size / 1e9:.2f} GB")

# Copy + patch config so torch_dtype reads fp16
with open(SRC / "config.json") as f: cfg = json.load(f)
cfg["torch_dtype"] = "float16"
with open(DST / "config.json", "w") as f: json.dump(cfg, f, indent=2)

# Copy the rest verbatim (tokenizer, generation_config, etc.)
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "generation_config.json", "tokenizer.model"]:
  src = SRC / name
  if src.exists():
    shutil.copy2(src, DST / name)
    print(f"copied {name}")

print("done.")
