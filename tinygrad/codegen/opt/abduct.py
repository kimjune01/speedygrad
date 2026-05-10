"""Abduction search: hypothesis-driven kernel optimization.

Hypothesis-driven kernel optimization. Instead of trying all actions (O(actions)),
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
         "NOLOCALS": getenv("NOLOCALS", 0), "TC": getenv("TC", 1), "TC_OPT": getenv("TC_OPT", 2),
         "ALLOW_TF32": getenv("ALLOW_TF32", 0), "NOOPT": getenv("NOOPT", 0), "USE_TC": getenv("USE_TC", 1),
         "DEVECTORIZE": getenv("DEVECTORIZE", 1), "EMULATED_DTYPES": getenv("EMULATED_DTYPES", 0),
         "IMAGE": getenv("IMAGE", 0), "DISABLE_FAST_IDIV": getenv("DISABLE_FAST_IDIV", 0),
         "TRANSCENDENTAL": getenv("TRANSCENDENTAL", 1)}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("abduct_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k.expr:int(k.vmax+k.vmin)//2 for k in s.ast.variables()}

  # phase 0: try TC IF it beats the un-opted default. Even this is too lax for matvec
  # (TC always beats unopted) so the no-TC search winner is the real comparison; we
  # restore TC at the end if its measured time beats the search winner.
  _, compiled_default = _try_compile((0, s))
  if compiled_default is None:
    if DEBUG >= 1: print("ABDUCT: failed to compile default kernel")
    return s
  default_time = min(_time_program(compiled_default[0], var_vals, rawbufs, cnt=3))

  tc_alternatives: list[tuple[Scheduler, tuple, float]] = []  # (scheduler, compiled, time)
  candidates0 = get_kernel_actions(s, include_0=False)
  for idx, tc_sched in candidates0.items():
    if tc_sched.applied_opts and tc_sched.applied_opts[-1].op == OptOps.TC:
      _, compiled = _try_compile((idx, tc_sched))
      if compiled is None: continue
      tc_time = min(_time_program(compiled[0], var_vals, rawbufs, cnt=3))
      tc_alternatives.append((tc_sched, compiled, tc_time))
      if DEBUG >= 2: print(f"ABDUCT: TC candidate {tc_sched.applied_opts[-1]} {tc_time*1e6:.0f}us (default {default_time*1e6:.0f}us)")

  best_time = default_time
  best = s
  allowed_ops: set[OptOps]|None = None
  if DEBUG >= 2: print(f"ABDUCT: default {best_time*1e6:.0f}us {s.colored_shape()}")

  for depth in range(max_depth):
    candidates = get_kernel_actions(best, include_0=False)
    if not candidates: break

    if allowed_ops is not None:
      candidates = {i: c for i, c in candidates.items()
                    if c.applied_opts and c.applied_opts[-1].op in allowed_ops}
      if not candidates: break

    seen_libs: set[bytes] = set()
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

    # Re-validate the top candidates at higher cnt before committing. The cnt=3 search
    # has enough noise to make a no-op-ish opt (e.g. UNROLL(0,0)) look like the winner
    # in ~17% of runs (gemm_256: 106us vs 55us). Re-time the top 3 at cnt=7 and pick
    # the best of those — bounds search noise without adding much cost.
    hypotheses.sort(key=lambda h: -h.speedup)
    top_k = hypotheses[:3]
    if len(top_k) > 1:
      revalidated = []
      for h in top_k:
        _, recompiled = _try_compile((0, h.scheduler))
        if recompiled is None: continue
        try:
          rt = min(_time_program(recompiled[0], var_vals, rawbufs, cnt=7))
        except (RuntimeError, AssertionError): continue
        revalidated.append(Hypothesis(scheduler=h.scheduler, opt=h.opt, after=rt,
                                       speedup=best_time / rt if rt > 0 else 0))
      if revalidated:
        winner = max(revalidated, key=lambda h: h.speedup)
      else:
        winner = top_k[0]
    else:
      winner = top_k[0]

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
      else:
        best_time = final_best

  # late TC sweep: if any TC alternative beats the search winner, drop into it and
  # run a short follow-up search from the TC starting point. This handles the case
  # where TC is the right answer but the no-TC search converged elsewhere.
  for tc_sched, tc_compiled, tc_time_initial in tc_alternatives:
    final_tc = min(_time_program(tc_compiled[0], var_vals, rawbufs, cnt=7))
    if final_tc < best_time * 0.95:
      if DEBUG >= 2: print(f"ABDUCT: TC late-adopt {tc_sched.applied_opts[-1]} {final_tc*1e6:.0f}us beats search winner {best_time*1e6:.0f}us")
      best = tc_sched
      best_time = final_tc
      compiled_default = tc_compiled
      # follow-up search from TC: deeper because TC needs LOCAL/UPCAST/UNROLL tuning
      tc_max_depth = max(max_depth, 5)
      allowed_ops = _TRANSITIONS.get(OptOps.TC, {OptOps.UPCAST, OptOps.LOCAL, OptOps.UNROLL})
      for depth in range(tc_max_depth):
        candidates = get_kernel_actions(best, include_0=False)
        candidates = {i: c for i, c in candidates.items()
                      if c.applied_opts and c.applied_opts[-1].op in allowed_ops}
        if not candidates: break
        seen_libs = set()
        if compiled_default is not None and len(compiled_default[0].src) > 4:
          dl = compiled_default[0].src[4].arg
          if dl is not None: seen_libs.add(dl)
        hypotheses = []
        early_stop = max(best_time * 1.25, best_time + 10e-6)
        for idx, candidate in candidates.items():
          _, compiled = _try_compile((idx, candidate))
          if compiled is None: continue
          lib = compiled[0].src[4].arg if len(compiled[0].src) > 4 else None
          if lib is not None:
            if lib in seen_libs: continue
            seen_libs.add(lib)
          try:
            tms = _time_program(compiled[0], var_vals, rawbufs, early_stop=early_stop, cnt=3)
            t = min(tms)
          except (RuntimeError, AssertionError): continue
          hypotheses.append(Hypothesis(scheduler=candidate, opt=candidate.applied_opts[-1],
                                       after=t, speedup=best_time / t if t > 0 else 0))
        if not hypotheses: break
        # same top-k re-validation as the main search loop
        hypotheses.sort(key=lambda h: -h.speedup)
        top_k = hypotheses[:3]
        if len(top_k) > 1:
          rev = []
          for h in top_k:
            _, rc = _try_compile((0, h.scheduler))
            if rc is None: continue
            try: rt = min(_time_program(rc[0], var_vals, rawbufs, cnt=7))
            except (RuntimeError, AssertionError): continue
            rev.append(Hypothesis(scheduler=h.scheduler, opt=h.opt, after=rt,
                                   speedup=best_time / rt if rt > 0 else 0))
          winner = max(rev, key=lambda h: h.speedup) if rev else top_k[0]
        else:
          winner = top_k[0]
        if winner.speedup <= 1.01: break
        best = winner.scheduler
        best_time = winner.after
        allowed_ops = _TRANSITIONS.get(winner.opt.op, {winner.opt.op})
        if DEBUG >= 2:
          print(f"ABDUCT TC d{depth}: {winner.opt} → {winner.after*1e6:.0f}us ({winner.speedup:.2f}x)")
      break

  if CACHELEVEL >= 1: diskcache_put("abduct_search", key, best.applied_opts)
  return best
