"""Abduction search: hypothesis-driven kernel optimization.

Alternative to beam_search. Instead of trying all actions (O(actions)),
observe the default kernel, perturb one opt at a time, diff the result,
and follow only the hypotheses that the diff generates.

Two samples, one diff, the shape of the failure names the next experiment.
"""
from dataclasses import dataclass
from tinygrad.uop.ops import UOp
from tinygrad.device import Buffer
from tinygrad.helpers import DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, IGNORE_BEAM_CACHE
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.search import _time_program, _try_compile, _ensure_buffer_alloc, get_kernel_actions
from tinygrad.codegen.opt.postrange import Scheduler

@dataclass
class Hypothesis:
  scheduler: Scheduler
  opt: Opt
  after: float
  speedup: float

def abduct_search(s:Scheduler, rawbufs:list[Buffer], max_depth:int=3, disable_cache=IGNORE_BEAM_CACHE.value) -> Scheduler:
  """Hypothesis-driven search. O(depth * category_filtered_actions) trials.

  1. Time the default (heuristic) kernel
  2. Try each available action → keep the best hypothesis
  3. Next depth: only try actions in the SAME op category as the winner
  4. Stop when no hypothesis improves by >1%, or depth exhausted
  """
  key = {"ast": s.ast.key, "device": s.ren.target.device, "suffix": s.ren.suffix}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("abduct_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k.expr:int(k.vmax+k.vmin)//2 for k in s.ast.variables()}

  _, compiled = _try_compile((0, s))
  if compiled is None:
    if DEBUG >= 1: print("ABDUCT: failed to compile default kernel")
    return s
  best_time = min(_time_program(compiled[0], var_vals, rawbufs, cnt=3))
  best = s
  seen_libs: set[bytes] = set()
  winner_category: int|None = None  # op category from previous depth
  if DEBUG >= 2: print(f"ABDUCT: default {best_time*1e6:.0f}us {s.colored_shape()}")

  for depth in range(max_depth):
    candidates = get_kernel_actions(best, include_0=False)
    if not candidates: break

    # filter by winner category after depth 0 (the diff names the next experiment)
    if winner_category is not None:
      candidates = {i: c for i, c in candidates.items()
                    if c.applied_opts and c.applied_opts[-1].op == winner_category}
      if not candidates: break

    hypotheses: list[Hypothesis] = []
    for idx, candidate in candidates.items():
      _, compiled = _try_compile((idx, candidate))
      if compiled is None: continue
      lib = compiled[0].src[4].arg if len(compiled[0].src) > 4 else None
      if lib in seen_libs: continue
      if lib is not None: seen_libs.add(lib)
      try:
        tms = _time_program(compiled[0], var_vals, rawbufs, early_stop=best_time*1.25, cnt=3)
        t = min(tms)
      except Exception: continue
      opt = candidate.applied_opts[-1] if candidate.applied_opts else Opt(OptOps.UPCAST, 0, 0)
      hypotheses.append(Hypothesis(scheduler=candidate, opt=opt, after=t, speedup=best_time / t if t > 0 else 0))

    if not hypotheses: break

    winner = max(hypotheses, key=lambda h: h.speedup)
    if winner.speedup <= 1.01:
      if DEBUG >= 2: print(f"ABDUCT d{depth}: converged (best hypothesis {winner.speedup:.3f}x)")
      break

    best = winner.scheduler
    best_time = winner.after
    winner_category = winner.opt.op
    if DEBUG >= 2:
      print(f"ABDUCT d{depth}: {winner.opt} → {winner.after*1e6:.0f}us ({winner.speedup:.2f}x) "
            f"tried {len(hypotheses)} → follow {winner.opt.op.name} {best.colored_shape()}")

  if CACHELEVEL >= 1: diskcache_put("abduct_search", key, best.applied_opts)
  return best
