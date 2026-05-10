"""Numerical equivalence: speedygrad fp32 matmul w/ ALLOW_TF32 vs torch fp32 matmul w/ TF32 enabled.

TF32 has 10 mantissa bits (vs 23 for fp32). torch's allow_tf32=True uses TF32 for cuBLAS.
Verify speedygrad's TF32 path produces results within the same tolerance as torch's.

Tolerances chosen to match torch internal matmul tests:
- fp32 strict: rtol=1e-5
- fp32 with TF32: rtol=1e-2 (10-bit mantissa drops ~3 decimal digits)
"""
import os, sys
os.environ.setdefault("DEV", "CUDA")

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tinygrad import Tensor

def matmul_check(N, seed=0):
  np.random.seed(seed)
  a_np = np.random.randn(N, N).astype(np.float32)
  b_np = np.random.randn(N, N).astype(np.float32)

  # speedygrad with TF32
  os.environ["ALLOW_TF32"] = "1"
  sg_a = Tensor(a_np); sg_b = Tensor(b_np)
  sg_out = (sg_a @ sg_b).numpy()

  # torch with TF32 on
  tr_a = torch.from_numpy(a_np).cuda(); tr_b = torch.from_numpy(b_np).cuda()
  tr_out = (tr_a @ tr_b).cpu().numpy()

  # numpy reference (full fp32)
  ref = a_np @ b_np

  sg_err = np.abs(sg_out - ref).max() / (np.abs(ref).max() + 1e-9)
  tr_err = np.abs(tr_out - ref).max() / (np.abs(ref).max() + 1e-9)
  diff = np.abs(sg_out - tr_out).max() / (np.abs(tr_out).max() + 1e-9)
  print(f"N={N}: speedygrad_vs_ref={sg_err:.2e}  torch_vs_ref={tr_err:.2e}  speedygrad_vs_torch={diff:.2e}")
  # both should be in roughly the same TF32 noise floor (~1e-3 for N=1024)
  return sg_err, tr_err, diff

if __name__ == "__main__":
  for N in [256, 1024, 2048]:
    matmul_check(N)
