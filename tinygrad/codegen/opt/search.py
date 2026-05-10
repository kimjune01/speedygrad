import math, time, traceback, signal
from dataclasses import replace
from tinygrad.uop.ops import sym_infer, AxisType, UOp
from tinygrad.device import Device, Buffer
from tinygrad.helpers import prod, DEBUG, getenv, Context, unwrap
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.tensor import Tensor
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen.opt.postrange import Scheduler

actions = [Opt(op=OptOps.UPCAST, axis=axis, arg=amt) for amt in [0,2,3,4,5,7] for axis in range(8)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, arg=amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, arg=amt) for amt in [2,3,4,8,13,16,29] for axis in range(6)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, arg=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, arg=amt) for amt in [0,4,8,16] for axis in range(3)]
actions += [Opt(op=OptOps.LOCAL, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=6, arg=2)]
actions += [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, getenv("TC", 1)))]
actions += [Opt(op=OptOps.TC, axis=axis, arg=(-1, getenv("TC_OPT", 2), getenv("TC", 1))) for axis in range(9)]
actions += [Opt(op=OptOps.SWAP, axis=axis_0, arg=axis_1) for axis_0 in range(5) for axis_1 in range(axis_0+1, 5)]
actions += [Opt(op=OptOps.THREAD, axis=axis, arg=amt) for amt in [2,3,4,5,8,12,16,24,32,64] for axis in range(3)]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size = [sym_infer(sz, var_vals) for sz in global_size]
  input_size = prod(test_global_size)
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        break
  return test_global_size, input_size / prod(test_global_size)

def _time_program(prg:UOp, var_vals:dict[str, int], rawbufs:list[Buffer], early_stop:float|None=None,
                  allow_test_size:int=True, max_global_size:int|None=65536, clear_l2=False, cnt=3, name="test", dev_timeout=False) -> list[float]:
  timeout = int(early_stop * 1e3) if dev_timeout and early_stop is not None and early_stop < math.inf else None
  factor = 1
  if allow_test_size and max_global_size is not None:
    global_size, factor = get_test_global_size(prg.arg.global_size, max_global_size, var_vals)
    prg = prg.replace(arg=replace(prg.arg, global_size=tuple(global_size)))
  try: rt = get_runtime(prg.src[1].arg, prg)
  except AssertionError: return [math.inf] * cnt
  global_size, local_size = prg.arg.launch_dims(var_vals)
  bufs = [rawbufs[i]._buf for i in prg.arg.globals]
  tms = []
  for _ in range(cnt):
    if clear_l2:
      if hasattr(dev:=Device[prg.src[1].arg], 'invalidate_caches'): dev.invalidate_caches()
      else:
        with Context(DEBUG=0, SEARCH=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024,1024).contiguous().realize(do_update_stats=False)
    tms.append(unwrap(rt(*bufs, global_size=global_size, local_size=local_size, vals=prg.arg.vals(var_vals), wait=True, timeout=timeout))*factor)
    if early_stop is not None and early_stop < min(tms): break
  return tms

class TimeoutException(Exception): pass
def timeout_handler(signum, frame):
  if DEBUG >= 2: print("*** COMPILE TIMEOUT")
  raise TimeoutException()

def _try_compile(x:tuple[int,Scheduler]) -> tuple[int, tuple[UOp, float]|None]:
  if hasattr(signal, "alarm"):
    signal.signal(getattr(signal, 'SIGALRM'), timeout_handler)
    signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  ret = None
  try:
    st = time.perf_counter()
    from tinygrad.codegen import to_program
    prg = to_program(x[1].copy().get_optimized_ast(name_override="test"), x[1].ren)
    et = time.perf_counter() - st
    uops = prg.src[2].src
    if len(uops) >= (uops_max:=getenv("BEAM_UOPS_MAX", 3000)) > 0:
      raise RuntimeError("too many uops")
    ret = (prg, et)
  except RuntimeError:
    if DEBUG >= 4: traceback.print_exc()
  except Exception as e:
    if getenv("BEAM_STRICT_MODE"): raise e
  finally:
    if hasattr(signal, "alarm"): signal.alarm(0)
  return x[0], ret

def _ensure_buffer_alloc(bufs:list[Buffer]) -> list[Buffer]: return [buf.ensure_allocated() if buf is not None else buf for buf in bufs]

def get_kernel_actions(s:Scheduler, include_0=True, max_up:int|None=None) -> dict[int, Scheduler]:
  acted, max_up, max_lcl = {0:s} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256) if max_up is None else max_up, getenv("BEAM_LOCAL_MAX", 1024)
  kernel_actions = actions.copy()

  for i,a in enumerate(kernel_actions):
    if a.axis is not None and a.op is not OptOps.TC:
      try: ax = s.real_axis(a.op, a.axis)
      except KernelOptError: continue
      if (ax >= s.shape_len) or (s.full_shape[ax] == a.arg and Opt(a.op, a.axis, 0) in kernel_actions): continue
    s2 = s.copy()
    try:
      s2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//tc.threads if hasattr(s2, 'tensor_core') and (tc:=s2.tensor_core) else 1
      for x,t in zip(s2.full_shape, s2.axis_types):
        if t in (AxisType.UPCAST, AxisType.UNROLL): up *= x
        elif t in (AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE): lcl *= x
      if up//tc_up > max_up or lcl > max_lcl: continue
      acted[i+1] = s2
    except KernelOptError: pass
  return acted
