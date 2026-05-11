"""Standalone CUDA online-softmax kernel bench.

Compiles via nvcc → PTX, loads via cuModuleLoadData, measures cuEventElapsedTime
over the kernel only (excludes host overhead). Compares against speedygrad's
3-kernel softmax GPU time on (256, 256) and (1024, 1024).

Algorithm: Milakov-Gimelshein (2018) — single pass maintaining (m, d) where
  m = running max
  d = sum(exp(x_i - m))
  on merge of two partials (m1,d1) and (m2,d2):
     m = max(m1,m2);  d = d1*exp(m1-m) + d2*exp(m2-m)
Then second pass writes out exp(x_i - m) / d.

One block per row, one warp per block (32 threads). Warp-reduce via shfl_down.
"""
import os, sys, ctypes, subprocess, tempfile, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("DEV", "CUDA")
import monkeypatch  # noqa
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check

KERNEL = r'''
extern "C" __global__ void online_softmax(float* __restrict__ out,
                                          const float* __restrict__ inp,
                                          int cols) {
  int row = blockIdx.x;
  int lid = threadIdx.x;
  const float* row_in = inp + row * cols;
  float* row_out = out + row * cols;
  // -FLT_MAX (not -INFINITY) so warp-reduce of two empty lanes computes
  // exp(-FLT_MAX - -FLT_MAX) = exp(0) = 1 instead of exp(NaN) = NaN.
  // Matters for cols < 32 (some lanes never iterate) and for masked attention
  // (lanes that only see -inf positions stay at the initial value).
  float m = -3.402823466e+38f, d = 0.0f;
  for (int i = lid; i < cols; i += 32) {
    float x = row_in[i];
    float m_new = fmaxf(m, x);
    d = d * __expf(m - m_new) + __expf(x - m_new);
    m = m_new;
  }
  // warp reduce
  for (int o = 16; o >= 1; o >>= 1) {
    float m2 = __shfl_down_sync(0xFFFFFFFF, m, o);
    float d2 = __shfl_down_sync(0xFFFFFFFF, d, o);
    float mn = fmaxf(m, m2);
    d = d * __expf(m - mn) + d2 * __expf(m2 - mn);
    m = mn;
  }
  m = __shfl_sync(0xFFFFFFFF, m, 0);
  d = __shfl_sync(0xFFFFFFFF, d, 0);
  float inv_d = 1.0f / d;
  for (int i = lid; i < cols; i += 32) {
    row_out[i] = __expf(row_in[i] - m) * inv_d;
  }
}
'''

def compile_ptx(src: str, arch: str = "sm_89") -> bytes:
  with tempfile.TemporaryDirectory() as td:
    cu = os.path.join(td, "k.cu")
    px = os.path.join(td, "k.ptx")
    with open(cu, "w") as f: f.write(src)
    r = subprocess.run(["nvcc", "-arch="+arch, "-ptx", "-O3", "--use_fast_math",
                        "--maxrregcount=32", "-o", px, cu], capture_output=True, text=True)
    if r.returncode != 0:
      # nvcc emits fatal messages (e.g. missing cl.exe) on stdout, not stderr
      raise RuntimeError(f"nvcc failed (rc={r.returncode}):\nstdout: {r.stdout}\nstderr: {r.stderr}")
    with open(px, "rb") as f: return f.read()

def main():
  # ensure CUDA context exists
  Device["CUDA"]  # init
  ctx = Device["CUDA"].context
  check(cuda.cuCtxSetCurrent(ctx))

  ptx = compile_ptx(KERNEL)
  module = cuda.CUmodule()
  check(cuda.cuModuleLoadData(ctypes.byref(module), ptx))
  prg = cuda.CUfunction()
  check(cuda.cuModuleGetFunction(ctypes.byref(prg), module, b"online_softmax"))

  shapes = [(256, 256), (1024, 1024), (4096, 4096)]
  # bug-hunt round 1: also exercise NaN-poisoning edge cases.
  # mask_frac=0.5 mimics causal attention; cols<32 stresses inactive warp lanes.
  edge_cases = [(64, 16, "cols<32"), (128, 128, "masked-causal")]

  for rows, cols in shapes:
    inp_np = np.random.randn(rows, cols).astype(np.float32)
    expected = np.exp(inp_np - inp_np.max(axis=-1, keepdims=True))
    expected = expected / expected.sum(axis=-1, keepdims=True)

    inp_t = Tensor(inp_np).contiguous().realize()
    out_t = Tensor.zeros(rows, cols).contiguous().realize()
    Device["CUDA"].synchronize()

    inp_buf = inp_t.uop.buf_uop.realized._buf
    out_buf = out_t.uop.buf_uop.realized._buf

    # kernelParams = array of pointers to each arg (CUDA Driver API standard path)
    p_cols = ctypes.c_int(cols)
    kparams = (ctypes.c_void_p * 3)(
      ctypes.cast(ctypes.byref(out_buf), ctypes.c_void_p),
      ctypes.cast(ctypes.byref(inp_buf), ctypes.c_void_p),
      ctypes.cast(ctypes.byref(p_cols), ctypes.c_void_p),
    )

    def launch():
      check(cuda.cuLaunchKernel(prg, rows, 1, 1, 32, 1, 1, 0, None, kparams, None))

    # warmup: 2000 iters does NOT lock GPU to boost — bug-hunt round 3+5 finding.
    # At ~10us/launch the GPU drains between dispatches (8K threads on a 117K-thread
    # GPU = 7% util), so it stays at intermediate P-states regardless of warmup
    # depth. Numbers reported here are at idle/intermediate clocks; comparison
    # vs speedygrad's 17us baseline is fair because that was measured at the same
    # launch cadence. Higher iteration count costs little, kept at 2000.
    for _ in range(2000):
      launch()
    Device["CUDA"].synchronize()

    # GPU time via cuEvents (kernel-only, no host)
    e0 = cuda.CUevent(); e1 = cuda.CUevent()
    check(cuda.cuEventCreate(ctypes.byref(e0), 0))
    check(cuda.cuEventCreate(ctypes.byref(e1), 0))
    times_us = []
    for _ in range(50):
      check(cuda.cuEventRecord(e0, None))
      launch()
      check(cuda.cuEventRecord(e1, None))
      check(cuda.cuEventSynchronize(e1))
      ms = ctypes.c_float()
      check(cuda.cuEventElapsedTime(ctypes.byref(ms), e0, e1))
      times_us.append(ms.value * 1000.0)
    times_us.sort()

    # wall time
    wall_us = []
    for _ in range(50):
      Device["CUDA"].synchronize()
      t0 = time.perf_counter()
      launch()
      Device["CUDA"].synchronize()
      wall_us.append((time.perf_counter()-t0)*1e6)
    wall_us.sort()

    out_np = out_t.numpy()
    diff = np.abs(out_np - expected).max()
    p10g, p50g = times_us[5], times_us[25]
    p10w, p50w = wall_us[5], wall_us[25]
    ok = "PASS" if diff < 1e-4 else "FAIL"
    print(f"{rows}x{cols}: GPU p10={p10g:.2f}us p50={p50g:.2f}us | wall p10={p10w:.2f}us p50={p50w:.2f}us | diff={diff:.2e} {ok}")
    check(cuda.cuEventDestroy_v2(e0))
    check(cuda.cuEventDestroy_v2(e1))

  # ---- correctness-only edge cases (bug-hunt round 1: NaN poisoning) ----
  print("\n=== edge cases (correctness only) ===")
  for rows, cols, label in edge_cases:
    if label == "cols<32":
      inp_np = np.random.randn(rows, cols).astype(np.float32)
    elif label == "masked-causal":
      inp_np = np.random.randn(rows, cols).astype(np.float32)
      # causal mask: position j masked-out for query i if j > i
      idx = np.arange(cols)[None, :]
      qi = np.arange(rows)[:, None] % cols  # cycle query positions through cols
      inp_np = np.where(idx <= qi, inp_np, -np.inf).astype(np.float32)
    expected = np.exp(inp_np - np.nanmax(np.where(np.isfinite(inp_np), inp_np, -1e38), axis=-1, keepdims=True))
    expected = expected / expected.sum(axis=-1, keepdims=True)

    inp_t = Tensor(inp_np).contiguous().realize()
    out_t = Tensor.zeros(rows, cols).contiguous().realize()
    Device["CUDA"].synchronize()
    inp_buf = inp_t.uop.buf_uop.realized._buf
    out_buf = out_t.uop.buf_uop.realized._buf
    p_cols = ctypes.c_int(cols)
    kparams = (ctypes.c_void_p * 3)(
      ctypes.cast(ctypes.byref(out_buf), ctypes.c_void_p),
      ctypes.cast(ctypes.byref(inp_buf), ctypes.c_void_p),
      ctypes.cast(ctypes.byref(p_cols), ctypes.c_void_p),
    )
    check(cuda.cuLaunchKernel(prg, rows, 1, 1, 32, 1, 1, 0, None, kparams, None))
    Device["CUDA"].synchronize()
    out_np = out_t.numpy()
    has_nan = bool(np.isnan(out_np).any())
    diff = float(np.nanmax(np.abs(out_np - expected))) if not has_nan else float('nan')
    status = "FAIL(NaN)" if has_nan else ("PASS" if diff < 1e-4 else f"FAIL(diff={diff:.2e})")
    print(f"  {label:>16}  rows={rows} cols={cols}  {status}")

  check(cuda.cuModuleUnload(module))

if __name__ == "__main__":
  main()
