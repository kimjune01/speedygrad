"""
Perceive Benchmark v2: measures optimization lift and structural gaps.

Runs each workload at multiple optimization levels, reports kernel counts,
and compares against PyTorch MPS as external baseline.

Metrics:
  heuristic_lift  = noopt / heuristic   (how much the optimizer improves raw lowering)
  vs_torch        = heuristic / torch    (competitive gap users see)
  beam_vs_heur    = beam / heuristic     (whether search improves or regresses)

Usage:
  python3 test/speed/bench_perceive.py
  BEAM=2 python3 test/speed/bench_perceive.py
  BIG=1 python3 test/speed/bench_perceive.py
"""
import os, time, subprocess, sys, json

BEAM = int(os.environ.get("BEAM", "0"))
BIG = int(os.environ.get("BIG", "0"))
CNT = int(os.environ.get("CNT", "8"))

WORKLOADS = [
    # (name, tinygrad_expr, torch_expr, shape_a, shape_b)
    ("gemm_1024",    "a @ b",                          "a @ b",                              (1024, 1024), (1024, 1024)),
    ("gemm_256",     "a @ b",                          "a @ b",                              (256, 256),   (256, 256)),
    ("add_4096",     "a + b",                          "a + b",                              (4096, 4096), (4096, 4096)),
    ("mul_sum",      "(a * b).sum()",                  "(a * b).sum()",                      (4096, 4096), (4096, 4096)),
    ("relu_4096",    "a.relu()",                       "torch.nn.functional.relu(a)",        (4096, 4096), None),
    ("exp_2048",     "a.exp()",                        "a.exp()",                            (2048, 2048), None),
    ("sum_4096",     "a.sum()",                        "a.sum()",                            (4096, 4096), None),
    ("permute",      "a.permute(1,0).contiguous()",    "a.permute(1,0).contiguous()",        (1024, 1024), None),
    ("softmax",      "a.softmax(-1)",                  "torch.nn.functional.softmax(a,-1)",  (256, 4096),  None),
    ("layernorm",    "a.layernorm()",                  "torch.nn.functional.layer_norm(a, a.shape[-1:])", (256, 128, 1024), None),
    ("matvec",       "a @ b",                          "a @ b",                              (4096,),      (4096, 4096)),
]

if BIG:
    WORKLOADS += [
        ("gemm_4096",  "a @ b",  "a @ b",  (4096, 4096), (4096, 4096)),
    ]

TINYGRAD_SCRIPT = '''
import os, time, json, sys
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
CNT = {cnt}

from tinygrad import Tensor, Device, GlobalCounters

shape_a = {shape_a}
shape_b = {shape_b}

dev = Device[Device.DEFAULT]

a = Tensor.rand(*shape_a)
b = Tensor.rand(*shape_b) if shape_b else None

# warmup: 3 runs to JIT and stabilize
for _ in range(3):
    ret = {expr}
    ret.realize()
    dev.synchronize()

# timed runs — fresh tensors each iteration to defeat caching
times_wall = []
times_kernel = []
for i in range(CNT):
    a = Tensor.rand(*shape_a)
    if shape_b: b = Tensor.rand(*shape_b)
    a.realize()
    if b is not None: b.realize()
    dev.synchronize()

    GlobalCounters.reset()
    st = time.perf_counter()
    ret = {expr}
    ret.realize()
    dev.synchronize()
    et = (time.perf_counter() - st) * 1000
    times_wall.append(et)
    times_kernel.append(GlobalCounters.time_sum_s * 1000 if GlobalCounters.time_sum_s > 0 else et)

times_wall.sort()
times_kernel.sort()
tw = times_wall[1:-1] if len(times_wall) > 3 else times_wall
tk = times_kernel[1:-1] if len(times_kernel) > 3 else times_kernel

result = {{
    "min_ms": min(times_wall),
    "median_ms": sorted(times_wall)[len(times_wall)//2],
    "mean_trimmed_ms": sum(tw)/len(tw),
    "kernel_ms": sum(tk)/len(tk),
    "kernel_count": GlobalCounters.kernel_count,
}}
print("RESULT:" + json.dumps(result), flush=True)
'''

TORCH_SCRIPT = '''
import os, time, json
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import torch
torch.set_num_threads(1)
CNT = {cnt}

shape_a = {shape_a}
shape_b = {shape_b}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

a = torch.rand(*shape_a, device=device)
b = torch.rand(*shape_b, device=device) if shape_b else None

def run():
    return {expr}

with torch.no_grad():
    for _ in range(3):
        ret = run()
        if device.type == "mps": torch.mps.synchronize()

    times = []
    for i in range(CNT):
        a = torch.rand(*shape_a, device=device)
        if shape_b: b = torch.rand(*shape_b, device=device)
        if device.type == "mps": torch.mps.synchronize()

        st = time.perf_counter()
        ret = run()
        if device.type == "mps": torch.mps.synchronize()
        et = (time.perf_counter() - st) * 1000
        times.append(et)

times.sort()
trimmed = times[1:-1] if len(times) > 3 else times

result = {{
    "min_ms": min(times),
    "median_ms": sorted(times)[len(times)//2],
    "mean_trimmed_ms": sum(trimmed)/len(trimmed),
    "device": str(device),
}}
print("RESULT:" + json.dumps(result), flush=True)
'''

KERNEL_INSPECT_SCRIPT = '''
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["DEBUG"] = "2"
import sys, json
from tinygrad import Tensor, Device, GlobalCounters

shape_a = {shape_a}
shape_b = {shape_b}

a = Tensor.rand(*shape_a)
b = Tensor.rand(*shape_b) if shape_b else None

ret = {expr}
ret.realize()
Device[Device.DEFAULT].synchronize()
'''

def run_subprocess(script, env_overrides=None, timeout=300):
    env = os.environ.copy()
    env.pop("BEAM", None)
    env.pop("NOOPT", None)
    env.pop("DEBUG", None)
    env.pop("IGNORE_BEAM_CACHE", None)
    if env_overrides:
        env.update(env_overrides)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout, env=env
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        stderr_lines = proc.stderr.strip().split("\n") if proc.stderr else []
        return {"error": stderr_lines[-1][:120] if stderr_lines else f"no result (rc={proc.returncode})",
                "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:120]}

def run_tinygrad(expr, shape_a, shape_b, env_overrides=None):
    env = dict(env_overrides or {})
    env.setdefault("DEBUG", "2")
    script = TINYGRAD_SCRIPT.format(expr=expr, shape_a=shape_a, shape_b=shape_b or "None", cnt=CNT)
    r = run_subprocess(script, env)
    if r and "kernel_ms" not in r: r["kernel_ms"] = r.get("mean_trimmed_ms")
    return r

def run_torch(expr, shape_a, shape_b):
    script = TORCH_SCRIPT.format(expr=expr, shape_a=shape_a, shape_b=shape_b or "None", cnt=CNT)
    return run_subprocess(script)

def inspect_kernels(expr, shape_a, shape_b, env_overrides=None):
    script = KERNEL_INSPECT_SCRIPT.format(expr=expr, shape_a=shape_a, shape_b=shape_b or "None")
    env = os.environ.copy()
    env.pop("BEAM", None)
    env.pop("NOOPT", None)
    env.pop("DEBUG", None)
    if env_overrides:
        env.update(env_overrides)
    env["DEBUG"] = "2"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60, env=env
        )
        return proc.stderr or ""
    except:
        return ""

def fmt(ms):
    if ms is None: return "      —"
    if ms < 1: return f"{ms*1000:5.0f} us"
    if ms < 100: return f"{ms:5.2f} ms"
    return f"{ms:5.0f} ms"

def ratio(a, b):
    if a is None or b is None or b == 0: return "   —"
    r = a / b
    if r >= 100: return f"{r:4.0f}x"
    if r >= 10: return f"{r:4.1f}x"
    return f"{r:4.2f}x"

def main():
    import platform
    print("=" * 90)
    print("Perceive Benchmark v2")
    print("=" * 90)
    print(f"Platform:  {platform.machine()}, macOS {platform.mac_ver()[0]}")
    print(f"Chip:      Apple Silicon (Metal)")
    print(f"PyTorch:   MPS backend")
    print(f"tinygrad:  Metal backend")
    print(f"Dtype:     float32")
    print(f"Iters:     {CNT} (trimmed mean excludes min/max)")
    print(f"Warmup:    3 runs excluded")
    print(f"Compile:   excluded from timing (post-JIT)")
    if BEAM:
        print(f"BEAM:      {BEAM} (fresh search, IGNORE_BEAM_CACHE=1)")
    print("=" * 90)
    print()

    # Phase 1: timing
    levels = [("noopt", {"NOOPT": "1"}), ("heur", {})]
    if BEAM:
        levels.append(("beam", {"BEAM": str(BEAM), "IGNORE_BEAM_CACHE": "1"}))

    hdr = f"{'workload':14s} {'torch':>7s}"
    for lname, _ in levels:
        hdr += f" {lname:>7s}"
    hdr += f" {'kernel':>7s} {'#k':>3s} {'lift':>6s} {'v_torch':>7s}"
    if BEAM:
        hdr += f" {'b/h':>6s} {'#k_b':>4s}"
    print(hdr)
    print("—" * len(hdr))

    results = []

    for name, tg_expr, th_expr, shape_a, shape_b in WORKLOADS:
        t = {}

        print(f"\r  {name:14s} torch ...", end="", flush=True)
        r = run_torch(th_expr, shape_a, shape_b)
        t["torch"] = r.get("mean_trimmed_ms") if "min_ms" in r else None

        for lname, env in levels:
            print(f"\r  {name:14s} {lname} ...   ", end="", flush=True)
            r = run_tinygrad(tg_expr, shape_a, shape_b, env)
            t[lname] = r.get("mean_trimmed_ms") if "min_ms" in r else None
            if lname == "heur":
                t["k_heur"] = r.get("kernel_count", "?")
                t["kernel_heur"] = r.get("kernel_ms") if "kernel_ms" in r else None
            if lname == "beam":
                t["k_beam"] = r.get("kernel_count", "?")
            if "error" in r:
                print(f"\n    {lname} error: {r['error']}")

        lift = t.get("noopt", 0) / t["heur"] if t.get("noopt") and t.get("heur") else None
        vs = t.get("kernel_heur", 0) / t["torch"] if t.get("kernel_heur") and t.get("torch") else None
        bvh = t.get("beam", 0) / t["heur"] if t.get("beam") and t.get("heur") else None

        line = f"\r{name:14s} {fmt(t.get('torch')):>7s}"
        for lname, _ in levels:
            line += f" {fmt(t.get(lname)):>7s}"
        line += f" {fmt(t.get('kernel_heur')):>7s}"
        line += f" {str(t.get('k_heur','')):>3s}"
        line += f" {ratio(t.get('noopt'), t.get('heur')):>6s}"
        line += f" {ratio(t.get('kernel_heur'), t.get('torch')):>7s}"
        if BEAM:
            line += f" {ratio(t.get('beam'), t.get('heur')):>6s}"
            line += f" {str(t.get('k_beam','')):>4s}"
        print(line)

        results.append({"name": name, **t, "lift": lift, "vs_torch": vs, "beam_vs_heur": bvh})

    # Phase 2: summary
    print()
    lifts = [r["lift"] for r in results if r.get("lift") and r["lift"] > 0]
    vst = [r["vs_torch"] for r in results if r.get("vs_torch")]
    if lifts:
        print(f"heuristic lift   min {min(lifts):.2f}x  median {sorted(lifts)[len(lifts)//2]:.2f}x  max {max(lifts):.2f}x")
    if vst:
        print(f"vs torch         min {min(vst):.2f}x  median {sorted(vst)[len(vst)//2]:.2f}x  max {max(vst):.2f}x")
    if BEAM:
        bvhs = [r["beam_vs_heur"] for r in results if r.get("beam_vs_heur")]
        if bvhs:
            regressed = [r["name"] for r in results if r.get("beam_vs_heur") and r["beam_vs_heur"] > 1.05]
            improved = [r["name"] for r in results if r.get("beam_vs_heur") and r["beam_vs_heur"] < 0.95]
            print(f"beam vs heur     min {min(bvhs):.2f}x  median {sorted(bvhs)[len(bvhs)//2]:.2f}x  max {max(bvhs):.2f}x")
            if regressed: print(f"  BEAM regressed: {', '.join(regressed)}")
            if improved: print(f"  BEAM improved:  {', '.join(improved)}")

    # Phase 3: kernel inspection for structural gap workloads
    print()
    print("=" * 90)
    print("Kernel inspection: workloads with vs_torch > 2.5x")
    print("=" * 90)
    for r in results:
        if r.get("vs_torch") and r["vs_torch"] > 2.5:
            name = r["name"]
            wl = next(w for w in WORKLOADS if w[0] == name)
            print(f"\n--- {name} (vs_torch={r['vs_torch']:.2f}x, kernels={r.get('k_heur','?')}) ---")
            debug_out = inspect_kernels(wl[1], wl[3], wl[4])
            kernel_lines = [l for l in debug_out.split("\n") if "kernel" in l.lower() or "*** " in l or "METAL" in l]
            for l in kernel_lines[:20]:
                print(f"  {l.strip()}")
            if not kernel_lines:
                print("  (no kernel debug output captured)")

    # Phase 4: legend
    print()
    print("—" * 50)
    print("lift:     NOOPT / heuristic — optimizer improvement over raw lowering")
    print("kernel:   GPU kernel time only (excludes Python scheduling overhead)")
    print("v_torch:  kernel / torch — competitive gap (1.0x = parity)")
    print("#k:       kernel count under heuristic")
    if BEAM:
        print("b/h:      beam / heuristic — <1.0 = search improved, >1.0 = search regressed")
        print("#k_b:     kernel count under BEAM")

if __name__ == "__main__":
    main()
