"""Benchmark the 11 standard workloads — kernel time via TinyJit with tight measurement."""
import time, statistics
from tinygrad import Tensor, Device, TinyJit

def bench(name, fn, warmup=20, trials=50):
  jitted = TinyJit(fn)
  for _ in range(warmup):
    jitted()
    Device.default.synchronize()
  times = []
  for _ in range(trials):
    Device.default.synchronize()
    t0 = time.perf_counter()
    jitted()
    Device.default.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1e6)
  times.sort()
  p10 = times[len(times)//10]
  p50 = times[len(times)//2]
  p90 = times[9*len(times)//10]
  print(f"{name:20s}  {p50:8.0f} us  (p10={p10:.0f}, p90={p90:.0f})")
  return p50

if __name__ == "__main__":
  N = 1024
  results = {}

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  results["gemm_1024"] = bench("gemm_1024", lambda: (a @ b).realize())

  a256 = Tensor.randn(256, 256).realize()
  b256 = Tensor.randn(256, 256).realize()
  results["gemm_256"] = bench("gemm_256", lambda: (a256 @ b256).realize())

  x = Tensor.randn(4096).realize()
  y = Tensor.randn(4096).realize()
  results["add_4096"] = bench("add_4096", lambda: (x + y).realize())

  a_ms = Tensor.randn(N, N).realize()
  b_ms = Tensor.randn(N, N).realize()
  results["mul_sum"] = bench("mul_sum", lambda: (a_ms * b_ms).sum().realize())

  r = Tensor.randn(4096).realize()
  results["relu_4096"] = bench("relu_4096", lambda: r.relu().realize())

  e = Tensor.randn(2048).realize()
  results["exp_2048"] = bench("exp_2048", lambda: e.exp().realize())

  s = Tensor.randn(4096).realize()
  results["sum_4096"] = bench("sum_4096", lambda: s.sum().realize())

  p = Tensor.randn(256, 256).realize()
  results["permute"] = bench("permute", lambda: p.permute(1, 0).contiguous().realize())

  sf = Tensor.randn(256, 256).realize()
  results["softmax"] = bench("softmax", lambda: sf.softmax().realize())

  ln = Tensor.randn(256, 256).realize()
  results["layernorm"] = bench("layernorm", lambda: Tensor.layernorm(ln).realize())

  mat = Tensor.randn(4096, 4096).realize()
  vec = Tensor.randn(1, 4096).realize()
  results["matvec"] = bench("matvec", lambda: (mat @ vec.T).realize())

  print(f"\n{'='*50}")
  print(f"Device: {Device.DEFAULT}")
  print(f"{'='*50}")
