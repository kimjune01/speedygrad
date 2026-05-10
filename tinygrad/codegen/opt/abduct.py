"""Abduction search: hypothesis-driven kernel optimization.

Alternative to beam_search. Instead of trying all actions (O(actions)),
observe the default kernel, perturb one opt at a time, diff the result,
and follow only the hypotheses that the diff generates.

Two samples, one diff, the shape of the failure names the next experiment.
"""
from dataclasses import dataclass
from tinygrad.uop.ops import UOp
from tinygrad.device import Buffer
from tinygrad.helpers import DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, IGNORE_SEARCH_CACHE
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.search import _time_program, _try_compile, _ensure_buffer_alloc, get_kernel_actions
from tinygrad.codegen.opt.postrange import Scheduler

@dataclass
class Hypothesis:
  scheduler: Scheduler
  opt: Opt
  after: float
  speedup: float

# which op categories can follow which — the "transition graph"
# depth 0 tries everything; depth 1+ follows edges from the winner's category
_TRANSITIONS: dict[OptOps, set[OptOps]] = {
  OptOps.LOCAL:    {OptOps.LOCAL, OptOps.UPCAST, OptOps.UNROLL, OptOps.SWAP},
  OptOps.UPCAST:   {OptOps.UPCAST, OptOps.LOCAL, OptOps.UNROLL, OptOps.SWAP},
  OptOps.UNROLL:   {OptOps.UNROLL, OptOps.UPCAST, OptOps.LOCAL},
  OptOps.GROUPTOP: {OptOps.GROUPTOP, OptOps.LOCAL, OptOps.UPCAST, OptOps.UNROLL},
  OptOps.GROUP:    {OptOps.GROUP, OptOps.LOCAL, OptOps.UPCAST},
  OptOps.TC:       {OptOps.UPCAST, OptOps.LOCAL, OptOps.UNROLL},
  OptOps.SWAP:     {OptOps.SWAP, OptOps.LOCAL, OptOps.UPCAST},
  OptOps.PADTO:    {OptOps.PADTO, OptOps.UPCAST, OptOps.LOCAL},
}

def abduct_search(s:Scheduler, rawbufs:list[Buffer], max_depth:int=3, disable_cache=IGNORE_SEARCH_CACHE.value) -> Scheduler:
  """Hypothesis-driven search. O(depth * filtered_actions) trials.

  1. Time the default kernel
  2. Try each available action → keep the best hypothesis
  3. Next depth: only try actions reachable via transition graph from winner
  4. Stop when no hypothesis improves by >1%, or depth exhausted
  5. Final validation: re-time winner with higher cnt before caching
  """
  key = {"ast": s.ast.key, "device": s.ren.target.device, "suffix": s.ren.suffix,
         "max_depth": max_depth, "BEAM_UPCAST_MAX": getenv("BEAM_UPCAST_MAX", 256),
         "BEAM_LOCAL_MAX": getenv("BEAM_LOCAL_MAX", 1024), "BEAM_UOPS_MAX": getenv("BEAM_UOPS_MAX", 3000),
         "NOLOCALS": getenv("NOLOCALS", 0), "TC": getenv("TC", 1), "TC_OPT": getenv("TC_OPT", 2)}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("abduct_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k.expr:int(k.vmax+k.vmin)//2 for k in s.ast.variables()}

  _, compiled_default = _try_compile((0, s))
  if compiled_default is None:
    if DEBUG >= 1: print("ABDUCT: failed to compile default kernel")
    return s
  default_time = min(_time_program(compiled_default[0], var_vals, rawbufs, cnt=3))
  best_time = default_time
  best = s
  allowed_ops: set[OptOps]|None = None
  if DEBUG >= 2: print(f"ABDUCT: default {best_time*1e6:.0f}us {s.colored_shape()}")

  for depth in range(max_depth):
    candidates = get_kernel_actions(best, include_0=False)
    if not candidates: break

    # filter by transition graph after depth 0
    if allowed_ops is not None:
      candidates = {i: c for i, c in candidates.items()
                    if c.applied_opts and c.applied_opts[-1].op in allowed_ops}
      if not candidates: break

    # per-depth seen_libs (reset each depth so composite paths aren't suppressed)
    seen_libs: set[bytes] = set()
    # seed with default lib to prevent no-op candidates from winning by noise
    if compiled_default is not None and len(compiled_default[0].src) > 4:
      default_lib = compiled_default[0].src[4].arg
      if default_lib is not None: seen_libs.add(default_lib)

    hypotheses: list[Hypothesis] = []
    early_stop_margin = max(best_time * 1.25, best_time + 10e-6)
    for idx, candidate in candidates.items():
      _, compiled = _try_compile((idx, candidate))
      if compiled is None: continue
      lib = compiled[0].src[4].arg if len(compiled[0].src) > 4 else None
      if lib is not None:
        if lib in seen_libs: continue
        seen_libs.add(lib)
      try:
        tms = _time_program(compiled[0], var_vals, rawbufs, early_stop=early_stop_margin, cnt=3)
        t = min(tms)
      except (RuntimeError, AssertionError): continue
      hypotheses.append(Hypothesis(scheduler=candidate, opt=candidate.applied_opts[-1],
                                   after=t, speedup=best_time / t if t > 0 else 0))

    if not hypotheses: break

    winner = max(hypotheses, key=lambda h: h.speedup)
    if winner.speedup <= 1.01:
      if DEBUG >= 2: print(f"ABDUCT d{depth}: converged (best hypothesis {winner.speedup:.3f}x)")
      break

    best = winner.scheduler
    best_time = winner.after
    allowed_ops = _TRANSITIONS.get(winner.opt.op, {winner.opt.op})
    if DEBUG >= 2:
      print(f"ABDUCT d{depth}: {winner.opt} → {winner.after*1e6:.0f}us ({winner.speedup:.2f}x) "
            f"tried {len(hypotheses)} → follow {[o.name for o in allowed_ops]} {best.colored_shape()}")

  # final validation: re-time winner with higher cnt to defeat noise
  if best is not s:
    _, compiled_best = _try_compile((0, best))
    if compiled_best is not None:
      final_default = min(_time_program(compiled_default[0], var_vals, rawbufs, cnt=7))
      final_best = min(_time_program(compiled_best[0], var_vals, rawbufs, cnt=7))
      if final_best >= final_default * 0.99:
        if DEBUG >= 2: print(f"ABDUCT: validation rejected ({final_best*1e6:.0f}us >= default {final_default*1e6:.0f}us)")
        best = s

  if CACHELEVEL >= 1: diskcache_put("abduct_search", key, best.applied_opts)
  return best
