"""Speedygrad vs PyTorch on CUDA, 11 workloads, p10/p50/p90.

Runs both with fresh JIT (no abduct disk cache assumptions). Uses CUDA event timing
on the torch side and host clock on speedygrad — host clock is consistent with
bench/workloads.py so the comparison is apples-to-apples.

Environment knobs picked up from os.environ:
- DEV         : speedygrad device (default CUDA)
- ALLOW_TF32  : 0|1 — TF32 tensor core gate for speedygrad fp32 matmul
- BEAM        : abduction depth
- IGNORE_SEARCH_CACHE : 0|1
"""
import os, time, json, sys

DEV = os.environ.get("DEV", "CUDA")
TRIALS = int(os.environ.get("BENCH_TRIALS", 30))
WARMUP = int(os.environ.get("BENCH_WARMUP", 15))

def stats(ts):
  ts = sorted(ts)
  return ts[len(ts)//10], ts[len(ts)//2], ts[len(ts)*9//10]

def run_speedygrad():
  from tinygrad import Tensor, Device, TinyJit
  def bench(fn):
    j = TinyJit(fn)
    for _ in range(WARMUP):
      j(); Device.default.synchronize()
    ts = []
    for _ in range(TRIALS):
      Device.default.synchronize()
      t0 = time.perf_counter()
      j()
      Device.default.synchronize()
      ts.append((time.perf_counter() - t0) * 1e6)
    return stats(ts)

  results = {}
  N = 1024
  a = Tensor.randn(N,N).realize(); b = Tensor.randn(N,N).realize()
  results["gemm_1024"] = bench(lambda: (a@b).realize())

  a256 = Tensor.randn(256,256).realize(); b256 = Tensor.randn(256,256).realize()
  results["gemm_256"] = bench(lambda: (a256@b256).realize())

  x = Tensor.randn(4096).realize(); y = Tensor.randn(4096).realize()
  results["add_4096"] = bench(lambda: (x+y).realize())

  ams = Tensor.randn(N,N).realize(); bms = Tensor.randn(N,N).realize()
  results["mul_sum"] = bench(lambda: (ams*bms).sum().realize())

  r = Tensor.randn(4096).realize()
  results["relu_4096"] = bench(lambda: r.relu().realize())

  e = Tensor.randn(2048).realize()
  results["exp_2048"] = bench(lambda: e.exp().realize())

  s = Tensor.randn(4096).realize()
  results["sum_4096"] = bench(lambda: s.sum().realize())

  p = Tensor.randn(256,256).realize()
  results["permute"] = bench(lambda: p.permute(1,0).contiguous().realize())

  sf = Tensor.randn(256,256).realize()
  results["softmax"] = bench(lambda: sf.softmax().realize())

  ln = Tensor.randn(256,256).realize()
  results["layernorm"] = bench(lambda: Tensor.layernorm(ln).realize())

  mat = Tensor.randn(4096,4096).realize(); vec = Tensor.randn(1,4096).realize()
  results["matvec"] = bench(lambda: (mat@vec.T).realize())
  return results

def run_torch():
  import torch
  d = "cuda" if torch.cuda.is_available() else "cpu"
  def bench(fn):
    for _ in range(WARMUP):
      fn(); torch.cuda.synchronize() if d=="cuda" else None
    ts=[]
    for _ in range(TRIALS):
      if d=="cuda": torch.cuda.synchronize()
      t0=time.perf_counter()
      fn()
      if d=="cuda": torch.cuda.synchronize()
      ts.append((time.perf_counter()-t0)*1e6)
    return stats(ts)

  results = {}
  g = torch.Generator(device=d).manual_seed(0)
  N = 1024
  a = torch.randn(N,N,device=d,generator=g); b = torch.randn(N,N,device=d,generator=g)
  results["gemm_1024"] = bench(lambda: (a@b))
  a256 = torch.randn(256,256,device=d,generator=g); b256 = torch.randn(256,256,device=d,generator=g)
  results["gemm_256"] = bench(lambda: (a256@b256))
  x = torch.randn(4096,device=d,generator=g); y = torch.randn(4096,device=d,generator=g)
  results["add_4096"] = bench(lambda: (x+y))
  ams = torch.randn(N,N,device=d,generator=g); bms = torch.randn(N,N,device=d,generator=g)
  results["mul_sum"] = bench(lambda: (ams*bms).sum())
  r = torch.randn(4096,device=d,generator=g)
  results["relu_4096"] = bench(lambda: r.relu())
  e = torch.randn(2048,device=d,generator=g)
  results["exp_2048"] = bench(lambda: e.exp())
  s = torch.randn(4096,device=d,generator=g)
  results["sum_4096"] = bench(lambda: s.sum())
  p = torch.randn(256,256,device=d,generator=g)
  results["permute"] = bench(lambda: p.permute(1,0).contiguous())
  sf = torch.randn(256,256,device=d,generator=g)
  results["softmax"] = bench(lambda: sf.softmax(dim=-1))
  ln_layer = torch.nn.LayerNorm(256, elementwise_affine=False).to(d)
  ln = torch.randn(256,256,device=d,generator=g)
  results["layernorm"] = bench(lambda: ln_layer(ln))
  mat = torch.randn(4096,4096,device=d,generator=g); vec = torch.randn(1,4096,device=d,generator=g)
  results["matvec"] = bench(lambda: (mat@vec.T))
  return results

if __name__ == "__main__":
  side = sys.argv[1] if len(sys.argv) > 1 else "both"
  if side in ("speedygrad", "both"):
    sg = run_speedygrad()
    print(json.dumps({"speedygrad": {k: list(v) for k,v in sg.items()}}, indent=2))
  if side in ("torch", "both"):
    tr = run_torch()
    print(json.dumps({"torch": {k: list(v) for k,v in tr.items()}}, indent=2))
