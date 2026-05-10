"""Minimal abduction engine: grid search over reduction opt parameters.

Tests whether measurement-driven search matches or beats the heuristic.
The abduction engine's job is to find GROUPTOP, UPCAST, LOCAL values
that the heuristic hardcodes — without knowing the hardware.
"""
from tinygrad import Tensor, Device, TinyJit
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.search import get_kernel_actions, beam_search
from tinygrad.codegen import to_program, full_rewrite_to_sink
from tinygrad.uop.ops import Ops
import time, os

def time_op(fn, warmup=20, trials=50):
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
    return times[trials // 2]

if __name__ == "__main__":
    workloads = {
        "softmax_4096":  (64, 4096),
        "softmax_1024":  (256, 1024),
        "softmax_256":   (1024, 256),
        "softmax_4093":  (64, 4093),
    }

    print("Heuristic baseline (universal padder active):")
    print(f"{'workload':20s} {'time_us':>8s}")
    for name, (rows, cols) in workloads.items():
        x = Tensor.randn(rows, cols).realize()
        t = time_op(lambda: x.softmax().realize())
        print(f"{name:20s} {t:8.0f}")

    print()
    print("BEAM=2 search (measurement-driven):")
    print(f"{'workload':20s} {'time_us':>8s}")
    for name, (rows, cols) in workloads.items():
        x = Tensor.randn(rows, cols).realize()
        os.environ["BEAM"] = "2"
        t = time_op(lambda: x.softmax().realize())
        print(f"{name:20s} {t:8.0f}")
    del os.environ["BEAM"]

    print()
    print("Online softmax (single kernel, no heuristic):")
    from bench.online_softmax import online_softmax_kernel
    from tinygrad.runtime.ops_metal import MetalCompiler, MetalProgram
    compiler = MetalCompiler()
    dev = Device['METAL']
    print(f"{'workload':20s} {'time_us':>8s}")
    for name, (rows, cols) in workloads.items():
        lib = compiler.compile(online_softmax_kernel(rows, cols))
        prg = MetalProgram(dev, 'online_softmax', lib)
        inp = Tensor.randn(rows, cols).contiguous().realize()
        out = Tensor.zeros(rows, cols).contiguous().realize()
        Device.default.synchronize()
        ib = inp.uop.buf_uop.realized
        ob = out.uop.buf_uop.realized
        for _ in range(20):
            prg(ob._buf, ib._buf, global_size=(rows,1,1), local_size=(32,1,1), wait=True)
        times = sorted([prg(ob._buf, ib._buf, global_size=(rows,1,1), local_size=(32,1,1), wait=True)*1e6 for _ in range(50)])
        print(f"{name:20s} {times[25]:8.0f}")
