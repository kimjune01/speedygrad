"""
Speedygrad per-token decode bench for Qwen 3 0.6B Instruct (Q8_0 GGUF).

Mirrors `bench/speedygrad_llama32_1b.py` methodology adapted for the new
`tinygrad/llm/model.py` Transformer + Transformer.from_gguf path used by
the modern Qwen support:
  - Model loaded from cached GGUF (auto-downloaded on first run via fetch())
  - SimpleTokenizer derived from GGUF metadata (matches the GGUF tokenizer
    bytewise; do NOT swap in HF AutoTokenizer)
  - Greedy decode (temperature=0.0) for deterministic output, matched to torch baseline
  - Per-token wall-clock timing on the .generate() generator
  - p10/p50/p90 across decode tokens, multiple runs for stability

Output JSON schema is identical to the Llama bench so cross-model comparisons
are direct.
"""
import os, sys, json, time, argparse
from pathlib import Path

# Ensure repo root on sys.path so `tinygrad` and `monkeypatch` import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import monkeypatch  # noqa: F401  -- enables Cython rewrites + cy_runtime + memoize-walk
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import fetch
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import models, SimpleTokenizer

DEFAULT_MODEL = "qwen3:0.6b"
PROMPT = "Hello."
N_NEW_DEFAULT = 25

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                  help=f"Model key in tinygrad.llm.cli.models or path to a local GGUF")
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--runs", type=int, default=5)
  ap.add_argument("--n-new", type=int, default=N_NEW_DEFAULT)
  ap.add_argument("--out", type=str, default=None)
  args = ap.parse_args()

  Tensor.manual_seed(42)
  device = Device.DEFAULT
  print(f"loading {args.model} on {device}", file=sys.stderr)

  # fetch returns a local Path (cached after first call)
  gguf_url_or_path = models.get(args.model, args.model)
  gguf_path = fetch(gguf_url_or_path)
  model, kv = Transformer.from_gguf(gguf_path, max_context=args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # iter 10c-cont v2 trick: collapse the global RNG counter chain that may
  # have accumulated during weight processing or model build. Cheap no-op
  # if the dict is empty (no rand op has triggered _next_counter yet).
  for _counter in Tensor._device_rng_counters.values():
    _counter.realize()

  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  n_params = sum(x.numel() for x in __import__('tinygrad').nn.state.get_parameters(model))
  print(f"loaded \"{model_name}\" ({n_params:,} params)", file=sys.stderr)

  # encode prompt with the tokenizer baked into the GGUF (matches what the model expects)
  ids = tok.role("user") + tok.encode(PROMPT) + tok.end_turn() + tok.role("assistant")
  prompt_len = len(ids)
  print(f"prompt_len={prompt_len}, n_new={args.n_new}, runs={args.runs}", file=sys.stderr)

  # NOTE: Must use a single generate() call across all measurements. Calling
  # model.generate() multiple times in one process triggers a JIT-graph
  # invalid-argument error on the second invocation, likely because intermediate
  # tensors from the first call are GC'd while the captured graph still
  # references their buffers. Pattern matches cli.py:213-218 — one gen, many next().
  total_decode_tokens = args.runs * args.n_new
  prompt_ids = list(ids)
  gen = model.generate(prompt_ids, temperature=0.0)

  # First yield = prefill + first decode token (TTFT). Single TTFT measurement.
  GlobalCounters.reset()
  t0 = time.perf_counter()
  first_tok = next(gen)
  t1 = time.perf_counter()
  prefill_ms = (t1 - t0) * 1000
  prefill_ms_list = [prefill_ms]

  # Subsequent yields = per-token decode. Burn N_BURN to clear capture/compile noise.
  N_BURN = 5
  decoded_ids = [first_tok]
  for _ in range(N_BURN):
    decoded_ids.append(next(gen))

  all_decode_us = []
  for i in range(total_decode_tokens - N_BURN - 1):
    GlobalCounters.reset()
    t0 = time.perf_counter()
    next_tok = next(gen)
    t1 = time.perf_counter()
    all_decode_us.append((t1 - t0) * 1e6)
    decoded_ids.append(next_tok)
    if tok.is_end(next_tok):
      print(f"  hit end-of-text at decode token {i+N_BURN+1}", file=sys.stderr)
      break
  generated_text_first = tok.decode(decoded_ids)
  print(f"  total decoded tokens: {len(decoded_ids)} (TTFT={prefill_ms:.1f}ms, "
        f"burn={N_BURN}, measured={len(all_decode_us)})", file=sys.stderr)

  if not all_decode_us:
    print("ERROR: no decode samples collected", file=sys.stderr)
    sys.exit(1)

  all_decode_us.sort()
  n = len(all_decode_us)
  p10_us = all_decode_us[max(0, n//10)]
  p50_us = all_decode_us[n//2]
  p90_us = all_decode_us[min(n-1, 9*n//10)]
  prefill_p50 = sorted(prefill_ms_list)[len(prefill_ms_list)//2]

  result = {
    "framework": "speedygrad", "model_path": str(gguf_path), "device": str(device),
    "dtype": "fp16", "quant": "Q8_0",
    "model_name": model_name,
    "prompt": PROMPT, "prompt_len": prompt_len, "n_new_tokens": args.n_new,
    "runs": args.runs, "n_decode_samples": n,
    "decode_us_p10": p10_us, "decode_us_p50": p50_us, "decode_us_p90": p90_us,
    "decode_tps_p50": 1e6 / p50_us, "decode_tps_p10": 1e6 / p90_us, "decode_tps_p90": 1e6 / p10_us,
    "prefill_ms_p50": prefill_p50,  # includes first decode token
    "generated_text_run0": generated_text_first,
  }
  print(json.dumps(result, indent=2))
  if args.out:
    with open(args.out, "w") as f: json.dump(result, f, indent=2)

if __name__ == "__main__":
  main()
