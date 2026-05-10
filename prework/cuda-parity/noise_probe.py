"""Run abduction N times on the same workload, log winning kernel + measured time per run."""
import os, sys, json, sqlite3, pickle, subprocess, time
os.environ.setdefault("DEV", "CUDA")

def make(name):
  from tinygrad import Tensor
  if name == "sum_4096":
    s = Tensor.randn(4096).realize()
    return lambda: s.sum().realize()
  if name == "gemm_256":
    a = Tensor.randn(256,256).realize(); b = Tensor.randn(256,256).realize()
    return lambda: (a@b).realize()
  if name == "exp_2048":
    e = Tensor.randn(2048).realize()
    return lambda: e.exp().realize()
  if name == "gemm_1024":
    a = Tensor.randn(1024,1024).realize(); b = Tensor.randn(1024,1024).realize()
    return lambda: (a@b).realize()
  raise ValueError(name)

def bench_worker():
  from tinygrad import Device, TinyJit
  fn = make(sys.argv[2])
  j = TinyJit(fn)
  for _ in range(15): j(); Device.default.synchronize()
  ts = []
  for _ in range(30):
    Device.default.synchronize(); t0 = time.perf_counter(); j(); Device.default.synchronize()
    ts.append((time.perf_counter() - t0) * 1e6)
  ts.sort()
  print(json.dumps({"p10": ts[3], "p50": ts[15], "p90": ts[27]}))

if len(sys.argv) > 1 and sys.argv[1] == "worker":
  bench_worker()
  sys.exit(0)

def clear_abduct():
  db = os.path.expanduser('~/.cache/tinygrad/cache.db')
  conn = sqlite3.connect(db); cur = conn.cursor()
  cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
  for (t,) in cur.fetchall():
    if 'abduct' in t.lower(): cur.execute(f"DROP TABLE '{t}'")
  conn.commit(); conn.close()

def cache_dump():
  db = os.path.expanduser('~/.cache/tinygrad/cache.db')
  conn = sqlite3.connect(db); cur = conn.cursor()
  cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
  out = {}
  for (t,) in cur.fetchall():
    if 'abduct' in t.lower():
      try:
        cur.execute(f"SELECT val FROM '{t}'")
        rows = cur.fetchall()
        out[t] = [pickle.loads(r[0]) for r in rows]
      except Exception as e:
        out[t] = f"err: {e}"
  conn.close()
  return out

WORKLOAD = sys.argv[1]
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

for i in range(N_RUNS):
  clear_abduct()
  r = subprocess.run([sys.executable, __file__, "worker", WORKLOAD], capture_output=True, text=True, env=os.environ)
  cache = cache_dump()
  try:
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")][-1]
    d = json.loads(line)
  except Exception as e:
    print(f"run {i}: ERROR parse  stderr={r.stderr[-200:]}")
    continue
  cache_keys = []
  for table_data in cache.values():
    for entry in table_data:
      cache_keys.append(str(entry))
  print(f"run {i}: p10={d['p10']:7.1f} p50={d['p50']:7.1f} p90={d['p90']:7.1f}  opts={cache_keys}")
