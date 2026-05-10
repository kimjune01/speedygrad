"""Profile a single add_4096 call after JIT capture.

Goal: identify the ~25us speedygrad/torch delta on small ops. Use cProfile to
break down where time is spent in JIT replay.
"""
import os, sys, time, cProfile, pstats, io
os.environ.setdefault("DEV", "CUDA")

from tinygrad import Tensor, Device, TinyJit

x = Tensor.randn(4096).realize()
y = Tensor.randn(4096).realize()
j = TinyJit(lambda: (x + y).realize())

# Warmup to capture
for _ in range(20):
  j(); Device.default.synchronize()

# Profile 1000 calls
pr = cProfile.Profile()
pr.enable()
for _ in range(1000):
  Device.default.synchronize()
  j()
  Device.default.synchronize()
pr.disable()

# Print top 30 by cumulative
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())

# Wall-clock check
ts = []
for _ in range(50):
  Device.default.synchronize(); t0 = time.perf_counter(); j(); Device.default.synchronize()
  ts.append((time.perf_counter() - t0) * 1e6)
ts.sort()
print(f"\nadd_4096 wall: p10={ts[5]:.1f} p50={ts[25]:.1f} p90={ts[45]:.1f} us")

# Pure GPU kernel time via DEBUG=2 hint
print(f"\nFor reference: torch add_4096 baseline ~24us p10")
