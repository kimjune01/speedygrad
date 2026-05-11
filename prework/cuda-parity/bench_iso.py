"""Per-workload isolated bench. Spawns a fresh subprocess per workload to avoid
JIT cache / kernel-cache cross-pollution that biased bench.py's sequential runs.

Activates monkeypatch (Cython runtime fast path if cy_runtime.pyd built).

Usage:
  python bench_iso.py speedygrad   # runs all workloads in subprocesses
  python bench_iso.py torch
  python bench_iso.py both
  python bench_iso.py worker speedygrad gemm_1024  # internal: single workload
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try: import monkeypatch  # noqa: F401
except Exception: pass

WORKLOADS = ["gemm_1024", "gemm_256", "add_4096", "mul_sum", "relu_4096", "exp_2048",
             "sum_4096", "permute", "softmax", "layernorm", "matvec"]

def make_speedygrad(name):
  from tinygrad import Tensor
  N = 1024
  if name == "gemm_1024":
    a = Tensor.randn(N,N).realize(); b = Tensor.randn(N,N).realize()
    return lambda: (a@b).realize()
  if name == "gemm_256":
    a = Tensor.randn(256,256).realize(); b = Tensor.randn(256,256).realize()
    return lambda: (a@b).realize()
  if name == "add_4096":
    x = Tensor.randn(4096).realize(); y = Tensor.randn(4096).realize()
    return lambda: (x+y).realize()
  if name == "mul_sum":
    a = Tensor.randn(N,N).realize(); b = Tensor.randn(N,N).realize()
    return lambda: (a*b).sum().realize()
  if name == "relu_4096":
    r = Tensor.randn(4096).realize()
    return lambda: r.relu().realize()
  if name == "exp_2048":
    e = Tensor.randn(2048).realize()
    return lambda: e.exp().realize()
  if name == "sum_4096":
    s = Tensor.randn(4096).realize()
    return lambda: s.sum().realize()
  if name == "permute":
    p = Tensor.randn(256,256).realize()
    return lambda: p.permute(1,0).contiguous().realize()
  if name == "softmax":
    sf = Tensor.randn(256,256).realize()
    return lambda: sf.softmax().realize()
  if name == "layernorm":
    ln = Tensor.randn(256,256).realize()
    return lambda: Tensor.layernorm(ln).realize()
  if name == "matvec":
    mat = Tensor.randn(4096,4096).realize(); vec = Tensor.randn(1,4096).realize()
    return lambda: (mat@vec.T).realize()
  raise ValueError(name)

def make_torch(name):
  import torch
  d = "cuda" if torch.cuda.is_available() else "cpu"
  g = torch.Generator(device=d).manual_seed(0)
  N = 1024
  if name == "gemm_1024":
    a = torch.randn(N,N,device=d,generator=g); b = torch.randn(N,N,device=d,generator=g)
    return lambda: (a@b)
  if name == "gemm_256":
    a = torch.randn(256,256,device=d,generator=g); b = torch.randn(256,256,device=d,generator=g)
    return lambda: (a@b)
  if name == "add_4096":
    x = torch.randn(4096,device=d,generator=g); y = torch.randn(4096,device=d,generator=g)
    return lambda: (x+y)
  if name == "mul_sum":
    a = torch.randn(N,N,device=d,generator=g); b = torch.randn(N,N,device=d,generator=g)
    return lambda: (a*b).sum()
  if name == "relu_4096":
    r = torch.randn(4096,device=d,generator=g)
    return lambda: r.relu()
  if name == "exp_2048":
    e = torch.randn(2048,device=d,generator=g)
    return lambda: e.exp()
  if name == "sum_4096":
    s = torch.randn(4096,device=d,generator=g)
    return lambda: s.sum()
  if name == "permute":
    p = torch.randn(256,256,device=d,generator=g)
    return lambda: p.permute(1,0).contiguous()
  if name == "softmax":
    sf = torch.randn(256,256,device=d,generator=g)
    return lambda: sf.softmax(dim=-1)
  if name == "layernorm":
    ln_layer = torch.nn.LayerNorm(256, elementwise_affine=False).to(d)
    ln = torch.randn(256,256,device=d,generator=g)
    return lambda: ln_layer(ln)
  if name == "matvec":
    mat = torch.randn(4096,4096,device=d,generator=g); vec = torch.randn(1,4096,device=d,generator=g)
    return lambda: (mat@vec.T)
  raise ValueError(name)

def bench_speedygrad(name, warmup=20, trials=50):
  from tinygrad import Device, TinyJit
  fn = make_speedygrad(name)
  j = TinyJit(fn)
  for _ in range(warmup):
    j(); Device.default.synchronize()
  ts = []
  for _ in range(trials):
    Device.default.synchronize()
    t0 = time.perf_counter()
    j()
    Device.default.synchronize()
    ts.append((time.perf_counter() - t0) * 1e6)
  ts.sort()
  return ts[len(ts)//10], ts[len(ts)//2], ts[len(ts)*9//10]

def bench_torch(name, warmup=20, trials=50):
  import torch
  d = "cuda" if torch.cuda.is_available() else "cpu"
  fn = make_torch(name)
  for _ in range(warmup):
    fn()
    if d == "cuda": torch.cuda.synchronize()
  ts = []
  for _ in range(trials):
    if d == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if d == "cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e6)
  ts.sort()
  return ts[len(ts)//10], ts[len(ts)//2], ts[len(ts)*9//10]

if __name__ == "__main__":
  if sys.argv[1] == "worker":
    side, name = sys.argv[2], sys.argv[3]
    if side == "speedygrad": p10, p50, p90 = bench_speedygrad(name)
    else: p10, p50, p90 = bench_torch(name)
    print(json.dumps({"name": name, "p10": p10, "p50": p50, "p90": p90}))
    sys.exit(0)

  side = sys.argv[1] if len(sys.argv) > 1 else "both"
  sides = ["speedygrad", "torch"] if side == "both" else [side]
  results = {s: {} for s in sides}
  for s in sides:
    for w in WORKLOADS:
      env = os.environ.copy()
      r = subprocess.run([sys.executable, __file__, "worker", s, w], capture_output=True, text=True, env=env)
      try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
        results[s][w] = (d["p10"], d["p50"], d["p90"])
      except Exception as e:
        print(f"FAIL {s} {w}: {r.stderr[-500:]}", file=sys.stderr)
        results[s][w] = (-1, -1, -1)
  print(json.dumps(results, indent=2))
