"""Multi-model, multi-framework decode-tps scaling sweep for the README.

Runs each (model, framework) combination as a SUBPROCESS to avoid JIT cache
cross-pollination. Collects JSON outputs, formats a Markdown table.

Frameworks compared:
  - vanilla tinygrad: bench/speedygrad_qwen3_06b.py with SPEEDYGRAD_VANILLA=1 env
    (monkeypatch becomes a no-op; same bench code, no fork-specific optimizations)
  - speedygrad:       bench/speedygrad_qwen3_06b.py (default)
  - torch+HF eager:   bench/torch_qwen3_06b.py

Models swept (Qwen 3 family, GGUF for tinygrad/speedygrad, fp16 for torch):
  - qwen3:0.6b (Q8_0 GGUF, ~640MB / fp16 ~1.5GB)
  - qwen3:1.7b (Q4_K_M GGUF, ~1.1GB / fp16 ~3.4GB)
  - qwen3:8b   (Q4_K_M GGUF, ~5GB / fp16 ~16GB; may not fit on 4080)

Per-row Markdown output ready for paste into README.
"""
import os, sys, json, subprocess, time, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEEDYGRAD_BENCH = REPO / "bench" / "speedygrad_qwen3_06b.py"
TORCH_BENCH = REPO / "bench" / "torch_qwen3_06b.py"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

# (display name, speedygrad model key, torch HF model id, n_new)
MODELS = [
  ("Qwen 3 0.6B", "qwen3:0.6b",  "Qwen/Qwen3-0.6B", 25),
  ("Qwen 3 1.7B", "qwen3:1.7b",  "Qwen/Qwen3-1.7B", 25),
  ("Qwen 3 8B",   "qwen3:8b",    "Qwen/Qwen3-8B",   25),
]

def run_bench(cmd, env_extra=None, timeout=900):
  """Run a bench subprocess, parse trailing JSON from stdout."""
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO)
  if env_extra: env.update(env_extra)
  print(f"  $ {' '.join(str(c) for c in cmd)}{' (env: ' + str(env_extra) + ')' if env_extra else ''}", file=sys.stderr)
  t0 = time.perf_counter()
  try:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env)
  except subprocess.TimeoutExpired:
    print(f"  TIMEOUT after {timeout}s", file=sys.stderr)
    return {"error": f"timeout {timeout}s"}
  elapsed = time.perf_counter() - t0
  print(f"  done in {elapsed:.0f}s, exit={r.returncode}", file=sys.stderr)
  if r.returncode != 0:
    last = r.stderr.splitlines()[-3:] if r.stderr else r.stdout.splitlines()[-3:]
    print(f"  STDERR tail: {last}", file=sys.stderr)
    return {"error": f"exit {r.returncode}", "stderr_tail": "\n".join(last)}
  # parse trailing JSON block from stdout
  out = r.stdout
  # find the last { ... } block
  start = out.rfind("{")
  end = out.rfind("}")
  if start < 0 or end < 0 or end < start:
    return {"error": "no JSON in stdout"}
  try:
    return json.loads(out[start:end+1])
  except json.JSONDecodeError as e:
    return {"error": f"JSON parse: {e}"}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", type=int, default=3)
  ap.add_argument("--n-new", type=int, default=20)
  ap.add_argument("--skip-vanilla", action="store_true")
  ap.add_argument("--skip-torch", action="store_true")
  ap.add_argument("--out", type=str, default="bench_scaling.json")
  args = ap.parse_args()

  results = []
  for display_name, sg_key, hf_id, _n_new in MODELS:
    n_new = args.n_new
    print(f"\n=== {display_name} (sg={sg_key}, hf={hf_id}) ===", file=sys.stderr)
    row = {"display": display_name, "sg_key": sg_key, "hf_id": hf_id}

    # 1. vanilla tinygrad (same bench code, monkeypatch disabled)
    if not args.skip_vanilla:
      print(f"--- vanilla tinygrad ---", file=sys.stderr)
      row["vanilla"] = run_bench(
        [PYTHON, SPEEDYGRAD_BENCH, "--model", sg_key, "--runs", args.runs, "--n-new", n_new],
        env_extra={"SPEEDYGRAD_VANILLA": "1"})

    # 2. speedygrad (default)
    print(f"--- speedygrad ---", file=sys.stderr)
    row["speedygrad"] = run_bench(
      [PYTHON, SPEEDYGRAD_BENCH, "--model", sg_key, "--runs", args.runs, "--n-new", n_new])

    # 3. torch+HF
    if not args.skip_torch:
      print(f"--- torch+HF ---", file=sys.stderr)
      row["torch"] = run_bench(
        [PYTHON, TORCH_BENCH, "--model", hf_id, "--runs", args.runs, "--n-new", n_new])

    results.append(row)

  # save full JSON
  with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nSaved full results to {args.out}", file=sys.stderr)

  # format Markdown table
  print("\n## Decode tokens/sec — Qwen 3 size scaling, RTX 4080\n")
  print("| Model | Vanilla tinygrad | **Speedygrad** | speedygrad/vanilla | torch+HF (eager) | speedygrad/torch |")
  print("|---|---:|---:|---:|---:|---:|")
  for row in results:
    def tps(d):
      if d is None or d.get("error"): return ("—", None)
      v = d.get("decode_tps_p50")
      return (f"{v:.0f} tok/s", v) if v is not None else ("—", None)
    v_str, v_val = tps(row.get("vanilla"))
    s_str, s_val = tps(row.get("speedygrad"))
    t_str, t_val = tps(row.get("torch"))
    sv_ratio = f"**{s_val/v_val:.1f}×**" if (v_val and s_val) else "—"
    st_ratio = f"**{s_val/t_val:.1f}×**" if (t_val and s_val) else "—"
    print(f"| {row['display']} | {v_str} | **{s_str}** | {sv_ratio} | {t_str} | {st_ratio} |")
  print("\n*Speedygrad uses Q8_0 GGUF (8-bit weights); torch+HF uses fp16 (16-bit).*")
  print("*Reproduce: `python bench/scaling_table.py --runs 3 --n-new 20`*")

if __name__ == "__main__":
  main()
