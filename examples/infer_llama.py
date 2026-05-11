"""
Single-script Llama 3.2 1B inference demo for speedygrad.

Usage:
  python examples/infer_llama.py "Once upon a time"
  python examples/infer_llama.py "What is the capital of France?" --max-tokens 50

By default uses Q6_K GGUF (auto-downloads ~1GB on first run from a non-gated mirror).
Pass --fp16 to use fp16 safetensors (auto-converts from unsloth/Llama-3.2-1B-Instruct
on first run; ~2.5GB and a one-time bf16->fp16 conversion).

Run from repo root with PYTHONPATH=. and DEV=CUDA. monkeypatch is imported to enable
speedygrad's Cython rewrites + cy_runtime fast path + GRAPH_ONE_KERNEL.
"""
import argparse, os, sys, time, json, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import monkeypatch  # noqa: F401

from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import fetch

def ensure_q6k_model():
  """Download Q6_K GGUF + tokenizer.model if not cached."""
  fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
        "tokenizer.model", subdir="llama3-1b-instruct")
  return fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
               "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")

def ensure_fp16_model():
  """Download unsloth bf16 safetensors + convert to fp16 on disk if not cached."""
  dst = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / ".cache" / "llama32-1b-fp16"
  if (dst / "model.safetensors").exists() and (dst / "tokenizer.model").exists():
    return dst

  print(f"[setup] downloading unsloth/Llama-3.2-1B-Instruct (~2.5GB) to HF cache...", file=sys.stderr)
  from huggingface_hub import snapshot_download
  src = Path(snapshot_download(repo_id="unsloth/Llama-3.2-1B-Instruct",
                               allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"]))

  print(f"[setup] converting bf16 -> fp16 (one-time, ~30s)...", file=sys.stderr)
  import torch
  from safetensors.torch import load_file, save_file
  weights = load_file(str(src / "model.safetensors"))
  fp16 = {k: v.to(torch.float16).contiguous() for k, v in weights.items()}
  dst.mkdir(parents=True, exist_ok=True)
  save_file(fp16, str(dst / "model.safetensors"))

  with open(src / "config.json") as f: cfg = json.load(f)
  cfg["torch_dtype"] = "float16"
  with open(dst / "config.json", "w") as f: json.dump(cfg, f, indent=2)
  for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    if (src / name).exists(): shutil.copy2(src / name, dst / name)

  # Also need tokenizer.model (Meta tiktoken format) for examples.llama3 Tokenizer
  tok_model_src = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
                        "tokenizer.model", subdir="llama3-1b-instruct")
  shutil.copy2(tok_model_src, dst / "tokenizer.model")
  print(f"[setup] fp16 model ready at {dst}", file=sys.stderr)
  return dst

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("prompt", type=str, help="prompt to generate from")
  ap.add_argument("--max-tokens", type=int, default=100, help="max new tokens to generate")
  ap.add_argument("--fp16", action="store_true", help="use fp16 safetensors (default: Q6_K GGUF)")
  ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 = greedy)")
  args = ap.parse_args()

  Tensor.manual_seed(42)
  device = Device.DEFAULT

  print(f"[setup] loading model (this includes JIT capture on first call)...", file=sys.stderr)
  if args.fp16:
    model_path = ensure_fp16_model()
  else:
    model_path = Path(ensure_q6k_model())  # path to .gguf file

  from examples.llama3 import build_transformer, Tokenizer
  t0 = time.perf_counter()
  model = build_transformer(model_path, model_size="1B", quantize=None, device=device)
  load_ms = (time.perf_counter() - t0) * 1000
  print(f"[setup] weights loaded in {load_ms:.0f}ms", file=sys.stderr)

  tok_path = model_path / "tokenizer.model" if model_path.is_dir() else model_path.parent / "tokenizer.model"
  tokenizer = Tokenizer(str(tok_path))

  # Encode user prompt as a chat message (matches the Llama 3 instruct format)
  st = tokenizer.special_tokens
  prefix = ([tokenizer.bos_id]
            + [st["<|start_header_id|>"]] + tokenizer.encode("user")
            + [st["<|end_header_id|>"]] + tokenizer.encode("\n\n")
            + tokenizer.encode(args.prompt.strip())
            + [st["<|eot_id|>"]]
            + [st["<|start_header_id|>"]] + tokenizer.encode("assistant")
            + [st["<|end_header_id|>"]] + tokenizer.encode("\n\n"))
  print(f"[gen] prompt encoded to {len(prefix)} tokens, generating up to {args.max_tokens} more...", file=sys.stderr)

  # Prefill
  start_pos = 0
  t0 = time.perf_counter()
  for tok in prefix[:-1]:
    GlobalCounters.reset()
    model(Tensor([[tok]], device=device), start_pos, args.temperature, 0, 0.0, 0.0, 0.0).realize()
    start_pos += 1
  prefill_ms = (time.perf_counter() - t0) * 1000

  # Decode
  last_tok = prefix[-1]
  decoded = []
  decode_t0 = time.perf_counter()
  for _ in range(args.max_tokens):
    GlobalCounters.reset()
    next_tok = model(Tensor([[last_tok]], device=device), start_pos, args.temperature, 0, 0.0, 0.0, 0.0).item()
    if next_tok in tokenizer.stop_tokens:
      break
    decoded.append(next_tok)
    last_tok = next_tok
    start_pos += 1
  decode_ms = (time.perf_counter() - decode_t0) * 1000

  text = tokenizer.decode(decoded)
  print()
  print(text)
  print()
  n = len(decoded)
  print(f"[stats] prefill: {prefill_ms:.0f} ms ({len(prefix)} tok)  decode: {decode_ms:.0f} ms ({n} tok, {n*1000/decode_ms:.1f} tok/s)", file=sys.stderr)

if __name__ == "__main__":
  main()
