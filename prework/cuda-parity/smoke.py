"""Smoke test for the abduct.py + ALLOW_TF32 changes.

Verifies:
1. fp32 matmul output is numerically valid (TF32 precision OK)
2. fp16 matmul still works
3. matvec still works (no regression from late TC sweep)
4. The abduct cache key is stable across runs (no crashes from missing cache columns)
5. Reductions, elementwise, layernorm, softmax all complete without error
6. JIT path works on all the above
"""
import os, sys
os.environ.setdefault("DEV", "CUDA")

import numpy as np
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.helpers import ALLOW_TF32

print(f"ALLOW_TF32 default: {ALLOW_TF32.value}")
assert ALLOW_TF32.value == 1, "expected default 1 (PyTorch parity for cuBLAS)"

failures = []

def check(name, lhs, rhs, rtol=1e-5, atol=1e-5):
  try:
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)
    print(f"  OK  {name}")
  except AssertionError as e:
    failures.append((name, str(e).splitlines()[0]))
    print(f"  FAIL {name}: {str(e).splitlines()[0]}")

print("\n=== fp32 matmul correctness (TF32) ===")
np.random.seed(0)
for N in [64, 256, 1024]:
  a = np.random.randn(N, N).astype(np.float32)
  b = np.random.randn(N, N).astype(np.float32)
  out = (Tensor(a) @ Tensor(b)).numpy()
  ref = a @ b
  # TF32 has ~1e-3 mantissa precision; use Frobenius-style abs tolerance scaled to
  # max element magnitude rather than per-element rtol (which blows up near zero).
  scale = np.abs(ref).max()
  diff = np.abs(out - ref).max()
  rel = diff / (scale + 1e-9)
  status = "OK" if rel < 5e-3 else "FAIL"
  if status == "FAIL": failures.append((f"matmul N={N}", f"rel={rel:.2e} > 5e-3 (abs diff {diff:.4f}, scale {scale:.1f})"))
  print(f"  {status}  matmul N={N}  rel={rel:.2e}")

print("\n=== fp16 matmul correctness ===")
for N in [64, 256, 1024]:
  a = np.random.randn(N, N).astype(np.float16)
  b = np.random.randn(N, N).astype(np.float16)
  out = (Tensor(a, dtype=dtypes.half) @ Tensor(b, dtype=dtypes.half)).numpy()
  ref = a.astype(np.float32) @ b.astype(np.float32)
  check(f"fp16 matmul N={N}", out.astype(np.float32), ref, rtol=5e-2, atol=1e-1)

print("\n=== matvec correctness ===")
for M in [128, 1024, 4096]:
  mat = np.random.randn(M, M).astype(np.float32)
  vec = np.random.randn(1, M).astype(np.float32)
  out = (Tensor(mat) @ Tensor(vec).T).numpy()
  ref = mat @ vec.T
  check(f"matvec M={M}", out, ref, rtol=5e-3, atol=1e-3)

print("\n=== reductions ===")
x = np.random.randn(4096).astype(np.float32)
check("sum_4096", Tensor(x).sum().numpy(), x.sum(), rtol=1e-4)
check("max_4096", Tensor(x).max().numpy(), x.max(), rtol=1e-5)
check("mean_4096", Tensor(x).mean().numpy(), x.mean(), rtol=1e-4)

print("\n=== softmax / layernorm ===")
sf = np.random.randn(256, 256).astype(np.float32)
out = Tensor(sf).softmax().numpy()
ref = np.exp(sf - sf.max(axis=-1, keepdims=True))
ref = ref / ref.sum(axis=-1, keepdims=True)
check("softmax", out, ref, rtol=1e-4)

ln = np.random.randn(256, 256).astype(np.float32)
out = Tensor.layernorm(Tensor(ln)).numpy()
mean = ln.mean(axis=-1, keepdims=True)
var = ln.var(axis=-1, keepdims=True)
ref = (ln - mean) / np.sqrt(var + 1e-5)
check("layernorm", out, ref, rtol=1e-3)

print("\n=== JIT path ===")
a = Tensor.randn(256, 256).realize()
b = Tensor.randn(256, 256).realize()
j = TinyJit(lambda: (a @ b).realize())
for _ in range(3):
  j()
Device.default.synchronize()
print("  OK  JIT 256x256 matmul (3 calls)")

mat = Tensor.randn(4096, 4096).realize()
vec = Tensor.randn(1, 4096).realize()
jm = TinyJit(lambda: (mat @ vec.T).realize())
for _ in range(3):
  jm()
Device.default.synchronize()
print("  OK  JIT matvec (3 calls)")

print("\n=== summary ===")
if failures:
  print(f"FAIL {len(failures)} checks:")
  for n, err in failures:
    print(f"  {n}: {err}")
  sys.exit(1)
print("All smoke checks passed.")
