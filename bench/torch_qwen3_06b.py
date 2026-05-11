"""
torch + HF transformers baseline: Qwen 3 0.6B Instruct fp16 decode tok/s.

Apples-to-apples comparator for `bench/speedygrad_qwen3_06b.py`.

Note on quantization: speedygrad bench uses Q8_0 GGUF (8-bit weights);
this torch bench uses fp16 (16-bit weights). Both are "what users
actually run" — disclose in output via the "quant" field. For strict
apples-to-apples on Q8 you'd need a torch path through bitsandbytes
or auto-gptq, deferred.

Methodology matches speedygrad's bench:
  - One prefill pass over the chat-template prompt
  - N decode tokens per run, one at a time, past_key_values cached
  - Each decode token timed individually with cuda.synchronize before/after
  - Greedy via .argmax(-1) (matches speedygrad temperature=0)
  - p10/p50/p90 reported across the per-token measurements
  - Multiple full runs for stability
"""
import argparse, time, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Hello."
N_NEW_TOKENS = 25

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", type=int, default=5, help="full prefill+decode runs")
  ap.add_argument("--n-new", type=int, default=N_NEW_TOKENS)
  ap.add_argument("--out", type=str, default=None)
  args = ap.parse_args()

  assert torch.cuda.is_available(), "CUDA required"
  dev = "cuda"
  dtype = torch.float16

  print(f"loading {MODEL_ID} fp16 on {torch.cuda.get_device_name(0)}", file=sys.stderr)
  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=dev)
  model.eval()

  messages = [{"role": "user", "content": PROMPT}]
  enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
  input_ids = enc["input_ids"].to(dev)
  prompt_len = input_ids.shape[-1]
  print(f"prompt_len={prompt_len}, n_new={args.n_new}, runs={args.runs}", file=sys.stderr)

  all_decode_us = []
  prefill_ms_list = []
  generated_text_first = None

  with torch.no_grad():
    for r in range(args.runs):
      # prefill
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      out = model(input_ids, use_cache=True)
      logits = out.logits[:, -1, :]
      past = out.past_key_values
      next_tok = logits.argmax(-1, keepdim=True)
      torch.cuda.synchronize()
      t1 = time.perf_counter()
      prefill_ms = (t1 - t0) * 1000
      prefill_ms_list.append(prefill_ms)

      # decode
      decoded_ids = [next_tok.item()]
      decode_us = []
      for i in range(args.n_new - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(next_tok, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_tok = logits.argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        decode_us.append((t1 - t0) * 1e6)
        decoded_ids.append(next_tok.item())

      if generated_text_first is None:
        generated_text_first = tok.decode(decoded_ids, skip_special_tokens=False)

      # exclude first decode token of each run as warmup
      run_decode_us = decode_us[1:] if len(decode_us) > 1 else decode_us
      all_decode_us.extend(run_decode_us)
      if run_decode_us:
        run_p50 = sorted(run_decode_us)[len(run_decode_us)//2]
        print(f"  run{r}: prefill={prefill_ms:.1f}ms, decode_p50={run_p50:.0f}us, n_decode={len(run_decode_us)}", file=sys.stderr)

  all_decode_us.sort()
  n = len(all_decode_us)
  p10_us = all_decode_us[max(0, n//10)]
  p50_us = all_decode_us[n//2]
  p90_us = all_decode_us[min(n-1, 9*n//10)]
  prefill_p50 = sorted(prefill_ms_list)[len(prefill_ms_list)//2]

  result = {
    "framework": "torch+HF",
    "model": MODEL_ID, "device": torch.cuda.get_device_name(0),
    "dtype": "fp16", "quant": "fp16",
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
