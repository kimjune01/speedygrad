# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython runtime fast path for JIT replay.

Replaces tinygrad.engine.realize.run_linear (and inlines exec_kernel + the
single-device fast path of unwrap_multi + a no-contextmanager track_stats).

Imported via monkeypatch.py; module-attribute swap on:
  - tinygrad.engine.realize.run_linear
  - tinygrad.engine.jit.run_linear
  - tinygrad.tensor.run_linear

The handles imported by name elsewhere all get rebound. Fallback is the
original Python run_linear if Cython build is missing.
"""
from tinygrad.uop.ops import Ops
from tinygrad.helpers import GlobalCounters, DEBUG, PROFILE
from tinygrad.engine.realize import (
  ExecContext, exec_copy, exec_view, exec_validate, exec_graph, exec_encdec,
  exec_kernel, compile_linear, resolve_params, get_runtime, VALIDATE_WITH_CPU,
  pm_exec,
)
from tinygrad.device import MultiBuffer

cdef object _OPS_PROGRAM = Ops.PROGRAM
cdef object _OPS_COPY = Ops.COPY
cdef object _OPS_BUFFER_VIEW = Ops.BUFFER_VIEW
cdef object _OPS_CUSTOM_FUNCTION = Ops.CUSTOM_FUNCTION

cdef object _exec_copy = exec_copy
cdef object _exec_view = exec_view
cdef object _exec_validate = exec_validate
cdef object _exec_graph = exec_graph
cdef object _exec_encdec = exec_encdec
cdef object _exec_kernel_py = exec_kernel  # fallback for the multi-device path
cdef object _resolve_params = resolve_params
cdef object _get_runtime = get_runtime
cdef object _MultiBuffer = MultiBuffer
cdef object _GlobalCounters = GlobalCounters
cdef object _pm_exec = pm_exec
cdef object _compile_linear = compile_linear
cdef object _ExecContext = ExecContext
cdef object _VALIDATE_WITH_CPU = VALIDATE_WITH_CPU

cdef inline _exec_kernel_fast(ctx, call, ast):
  """Inlined single-device exec_kernel: skips unwrap_multi generator + track_stats
  contextmanager. Falls back to Python exec_kernel for MultiBuffer (multi-GPU)."""
  cdef list resolved = _resolve_params(call, ctx.input_uops)
  cdef list raw_bufs = [b.buffer for b in resolved]
  cdef Py_ssize_t i
  cdef bint multi = False
  for i in range(len(raw_bufs)):
    if isinstance(raw_bufs[i], _MultiBuffer):
      multi = True
      break
  if multi:
    _exec_kernel_py(ctx, call, ast)
    return

  arg = ast.arg
  cdef list prg_bufs = [raw_bufs[i].ensure_allocated() for i in arg.globals]
  device = prg_bufs[0].device
  rt = _get_runtime(device, ast)
  var_vals = ctx.var_vals
  global_size, local_size = arg.launch_dims(var_vals)
  vals = arg.vals(var_vals)
  cdef list buf_args = [b._buf for b in prg_bufs]

  # inlined track_stats: skip estimate_uop + sym_infer entirely on the fast path.
  # We still bump kernel_count so callers tracking it keep working.
  cdef bint want_stats = ctx.do_update_stats
  cdef bint want_debug = DEBUG >= 2
  cdef bint want_profile = bool(PROFILE)
  if not want_debug and not want_profile:
    if want_stats: _GlobalCounters.kernel_count += 1
    rt(*buf_args, global_size=global_size, local_size=local_size, vals=vals, wait=False)
    return

  # rare path: hand off to Python's exec_kernel which has the full
  # tracking + DEBUG-print + PROFILE-event logic. Same observable behavior.
  _exec_kernel_py(ctx, call, ast)


def cy_run_linear(linear, var_vals=None, input_uops=(), do_update_stats=True, jit=False):
  """Cython run_linear: direct-dispatch loop, fast-path for single-device PROGRAM call."""
  if not jit:
    linear = _compile_linear(linear, validate=_VALIDATE_WITH_CPU)
  ctx = _ExecContext(var_vals or {}, input_uops, do_update_stats, jit)

  for call in linear.src:
    ast = call.src[0]
    op = ast.op
    if op is _OPS_PROGRAM:
      _exec_kernel_fast(ctx, call, ast)
    elif op is _OPS_COPY:
      _exec_copy(ctx, call, ast)
    elif op is _OPS_BUFFER_VIEW:
      _exec_view(ctx, call, ast)
    elif op is _OPS_CUSTOM_FUNCTION:
      arg = ast.arg
      if arg == "validate": _exec_validate(ctx, call, ast)
      elif arg == "graph": _exec_graph(ctx, call, ast)
      elif arg == "encdec": _exec_encdec(ctx, call, ast)
      else: _pm_exec.rewrite(call, ctx)
    else:
      _pm_exec.rewrite(call, ctx)
