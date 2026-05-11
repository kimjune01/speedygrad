"""
Speedygrad per-token decode bench for Llama 3.2 1B Instruct fp16.

Mirrors `bench/torch_llama32_1b.py` methodology so results are directly comparable:
  - One prefill pass over chat-template prompt
  - N decode tokens per run, one at a time, KV cache via Transformer's forward_jit
  - Each decode token timed individually with GlobalCounters / time.perf_counter
  - p10/p50/p90 across per-token measurements (first decode token of each run excluded)
  - Multiple full runs for stability

Reads model from --model (a directory with model.safetensors + tokenizer.model + config.json).
Default points at the fp16-converted unsloth mirror at ~/.cache/llama32-1b-fp16/.
"""
import os, sys, json, time, argparse
from pathlib import Path

# Ensure repo root on sys.path so `extra` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import monkeypatch  # noqa: F401  -- enables Cython rewrites + cy_runtime fast path
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Context

from examples.llama3 import build_transformer  # type: ignore
from transformers import AutoTokenizer

PROMPT = "Hello."
N_NEW_DEFAULT = 25

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", type=str, default=str(Path(os.environ["USERPROFILE"]) / ".cache" / "llama32-1b-fp16"))
  ap.add_argument("--runs", type=int, default=5)
  ap.add_argument("--n-new", type=int, default=N_NEW_DEFAULT)
  ap.add_argument("--out", type=str, default=None)
  args = ap.parse_args()

  Tensor.manual_seed(42)
  device = Device.DEFAULT

  model_path = Path(args.model)
  print(f"loading {model_path} on {device}", file=sys.stderr)
  model = build_transformer(model_path, model_size="1B", quantize=None, device=device)

  # iter 10c-cont v2: collapse the global RNG counter's AFTER chain that
  # accumulated during weight init (one .assign per random-init weight =
  # ~114-deep chain for 1B). Walked from scratch every _apply_map_to_tensors
  # call otherwise — 73% of decode-phase walk cost is this stale history.
  for _counter in Tensor._device_rng_counters.values():
    _counter.realize()

  # Use HF tokenizer for input encoding so token IDs match torch bench exactly
  tokenizer = AutoTokenizer.from_pretrained(str(model_path))
  prompt_str = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                              add_generation_prompt=True, tokenize=False)
  toks = tokenizer.encode(prompt_str, add_special_tokens=False)
  if not isinstance(toks, list):
    toks = list(toks)
  prompt_len = len(toks)
  print(f"prompt_len={prompt_len}, n_new={args.n_new}, runs={args.runs}", file=sys.stderr)

  all_decode_us = []
  prefill_ms_list = []
  generated_text_first = None

  for r in range(args.runs):
    # Reset KV cache by re-instantiating the model? No -- the existing `prefill` uses
    # last_seen_toks dedup. Easiest: do prefill from start_pos=0 each run, accept that
    # KV cache will be overwritten via cache_kv assignment.
    start_pos = 0
    prefill_toks = toks[:-1]
    last_tok = toks[-1]

    # Prefill (one token at a time so each call goes through the JIT path)
    GlobalCounters.reset()
    t0 = time.perf_counter()
    for tok in prefill_toks:
      model(Tensor([[tok]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).realize()
      start_pos += 1
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000
    prefill_ms_list.append(prefill_ms)

    # Decode
    decoded_ids = []
    decode_us = []
    for i in range(args.n_new):
      GlobalCounters.reset()
      t0 = time.perf_counter()
      next_tok = model(Tensor([[last_tok]], device=device), start_pos, 0.0, 0, 0.0, 0.0, 0.0).item()
      t1 = time.perf_counter()
      decode_us.append((t1 - t0) * 1e6)
      decoded_ids.append(next_tok)
      last_tok = next_tok
      start_pos += 1

    if generated_text_first is None:
      generated_text_first = tokenizer.decode(decoded_ids, skip_special_tokens=False)

    run_decode_us = decode_us[1:] if len(decode_us) > 1 else decode_us
    all_decode_us.extend(run_decode_us)
    run_p50 = sorted(run_decode_us)[len(run_decode_us)//2]
    print(f"  run{r}: prefill={prefill_ms:.1f}ms, decode_p50={run_p50:.0f}us, n_decode={len(run_decode_us)}", file=sys.stderr)

  all_decode_us.sort()
  n = len(all_decode_us)
  p10_us = all_decode_us[max(0, n//10)]
  p50_us = all_decode_us[n//2]
  p90_us = all_decode_us[min(n-1, 9*n//10)]
  prefill_p50 = sorted(prefill_ms_list)[len(prefill_ms_list)//2]

  result = {
    "framework": "speedygrad", "model_path": str(model_path), "device": str(device), "dtype": "fp16",
    "prompt": PROMPT, "prompt_len": prompt_len, "n_new_tokens": args.n_new,
    "runs": args.runs, "n_decode_samples": n,
    "decode_us_p10": p10_us, "decode_us_p50": p50_us, "decode_us_p90": p90_us,
    "decode_tps_p50": 1e6 / p50_us, "decode_tps_p10": 1e6 / p90_us, "decode_tps_p90": 1e6 / p10_us,
    "prefill_ms_p50": prefill_p50,
    "generated_text_run0": generated_text_first,
  }
  print(json.dumps(result, indent=2))
  if args.out:
    with open(args.out, "w") as f: json.dump(result, f, indent=2)

if __name__ == "__main__":
  main()
