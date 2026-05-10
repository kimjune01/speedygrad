"""Abduction search: hypothesis-driven kernel optimization.

Alternative to beam_search. Instead of trying all actions (O(actions)),
observe the default kernel, perturb one opt at a time, diff the result,
and follow only the hypotheses that the diff generates.

Two samples, one diff, the shape of the failure names the next experiment.
"""
import time, math
from dataclasses import dataclass
from tinygrad.uop.ops import sym_infer, UOp
from tinygrad.device import Device, Buffer
from tinygrad.helpers import DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, IGNORE_BEAM_CACHE
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.search import _time_program, _try_compile, _ensure_buffer_alloc, get_kernel_actions
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime

@dataclass
class Hypothesis:
  opt: Opt
  before: float
  after: float
  speedup: float

def abduct_search(s:Scheduler, rawbufs:list[Buffer], max_depth:int=3, disable_cache=IGNORE_BEAM_CACHE.value) -> Scheduler:
  """Hypothesis-driven search. O(depth * branching) instead of O(all_actions).

  1. Time the default (heuristic) kernel
  2. Try each available action individually → diff vs default
  3. Keep the best hypothesis (biggest speedup)
  4. From the new baseline, repeat — but only try actions in the SAME category
     as the winning hypothesis (the diff names the next experiment)
  5. Stop when no hypothesis improves over current best
  """
  key = {"ast": s.ast.key, "device": s.ren.target.device, "suffix": s.ren.suffix}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("abduct_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k.expr:int(k.vmax+k.vmin)//2 for k in s.ast.variables()}

  # time the heuristic default
  _, compiled = _try_compile((0, s))
  if compiled is None:
    if DEBUG >= 1: print("ABDUCT: failed to compile default kernel")
    return s
  best_time = min(_time_program(compiled[0], var_vals, rawbufs, cnt=3))
  best = s
  if DEBUG >= 2: print(f"ABDUCT: default {best_time*1e6:.0f}us {s.colored_shape()}")

  for depth in range(max_depth):
    # get all available actions from current best
    candidates = get_kernel_actions(best, include_0=False)
    if not candidates: break

    # try each action — this is the perturbation step
    hypotheses: list[Hypothesis] = []
    for idx, candidate in candidates.items():
      _, compiled = _try_compile((idx, candidate))
      if compiled is None: continue
      try:
        tms = _time_program(compiled[0], var_vals, rawbufs, early_stop=best_time*3, cnt=3)
        t = min(tms)
      except Exception: continue
      opt = candidate.applied_opts[-1] if candidate.applied_opts else Opt(OptOps.UPCAST, 0, 0)
      speedup = best_time / t if t > 0 else 0
      hypotheses.append(Hypothesis(opt=opt, before=best_time, after=t, speedup=speedup))

    if not hypotheses: break

    # the best hypothesis wins — this is the diff
    winner = max(hypotheses, key=lambda h: h.speedup)
    if winner.speedup <= 1.01:  # less than 1% improvement — converged
      if DEBUG >= 2: print(f"ABDUCT d{depth}: converged (best hypothesis {winner.speedup:.3f}x)")
      break

    # apply the winning opt
    try:
      new_best = best.copy()
      new_best.apply_opt(winner.opt)
      best = new_best
      best_time = winner.after
      if DEBUG >= 2:
        print(f"ABDUCT d{depth}: {winner.opt} → {winner.after*1e6:.0f}us ({winner.speedup:.2f}x) "
              f"tried {len(hypotheses)} hypotheses {best.colored_shape()}")
    except KernelOptError:
      break

  if CACHELEVEL >= 1: diskcache_put("abduct_search", key, best.applied_opts)
  return best
