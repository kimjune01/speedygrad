"""Abduction engine: two samples, one diff, one hypothesis.

Not BEAM (induction — scale trials, pick winners).
Abduction: observe a surprising fact, diff against expectation,
the shape of the failure names the next experiment.

abduct(kernel, perturb) → list[Hypothesis]

Each hypothesis is: "this opt became available/unavailable when
the perturbation was applied, and it changed performance by X."
"""
from tinygrad import Tensor, Device
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops, UOp
from dataclasses import dataclass

@dataclass
class Hypothesis:
    figure: str          # what changed (the diff)
    ground: str          # what didn't change
    prediction: str      # consequence if the hypothesis is true
    perturbation: str    # what was poked

def get_opts(t: Tensor) -> list[tuple[str, list[Opt]]]:
    """Extract the heuristic's opt choices for each kernel in t's schedule."""
    linear = t.schedule_linear()
    results = []
    for call in linear.src:
        ast = call.src[0]
        if ast.op is Ops.SINK:
            # run the heuristic
            ren = Device.default.renderer
            k = Scheduler(ast, ren)
            try:
                k = hand_coded_optimizations(k)
            except Exception:
                pass
            results.append((str(ast.arg), list(k.applied_opts)))
    return results

def get_available_actions(t: Tensor) -> list[tuple[str, dict[int, Scheduler]]]:
    """Get all valid opt actions for each kernel."""
    linear = t.schedule_linear()
    results = []
    for call in linear.src:
        ast = call.src[0]
        if ast.op is Ops.SINK:
            ren = Device.default.renderer
            k = Scheduler(ast, ren)
            actions = get_kernel_actions(k)
            results.append((str(ast.arg), actions))
    return results

def abduct(make_tensor_a, make_tensor_b, label_a="before", label_b="after") -> list[Hypothesis]:
    """Two samples, one diff. Returns hypotheses about what changed and why.

    make_tensor_a, make_tensor_b: callables that return tensors to compare.
    The diff between their opt sequences IS the hypothesis.
    """
    opts_a = get_opts(make_tensor_a())
    opts_b = get_opts(make_tensor_b())

    actions_a = get_available_actions(make_tensor_a())
    actions_b = get_available_actions(make_tensor_b())

    hypotheses = []

    # diff the opt sequences kernel by kernel
    for i in range(min(len(opts_a), len(opts_b))):
        name_a, applied_a = opts_a[i]
        name_b, applied_b = opts_b[i]

        set_a = set(str(o) for o in applied_a)
        set_b = set(str(o) for o in applied_b)

        gained = set_b - set_a  # opts that appeared
        lost = set_a - set_b    # opts that disappeared

        if gained or lost:
            figure_parts = []
            if gained: figure_parts.append(f"gained: {gained}")
            if lost: figure_parts.append(f"lost: {lost}")

            h = Hypothesis(
                figure="; ".join(figure_parts),
                ground=f"kernel {i}: {set_a & set_b} unchanged",
                prediction=f"the {'gained' if gained else 'lost'} opts explain the performance difference",
                perturbation=f"{label_a} → {label_b}",
            )
            hypotheses.append(h)

    # diff available actions (what COULD be applied)
    for i in range(min(len(actions_a), len(actions_b))):
        _, avail_a = actions_a[i]
        _, avail_b = actions_b[i]

        # count action categories
        def categorize(actions):
            cats = {}
            for idx, sched in actions.items():
                if idx == 0: continue
                for o in sched.applied_opts:
                    cats.setdefault(o.op.name, 0)
                    cats[o.op.name] += 1
            return cats

        cats_a = categorize(avail_a)
        cats_b = categorize(avail_b)

        for cat in set(list(cats_a.keys()) + list(cats_b.keys())):
            count_a = cats_a.get(cat, 0)
            count_b = cats_b.get(cat, 0)
            if count_a != count_b:
                h = Hypothesis(
                    figure=f"kernel {i}: {cat} actions {count_a} → {count_b}",
                    ground=f"other action categories unchanged",
                    prediction=f"{'more' if count_b > count_a else 'fewer'} {cat} options changes the optimization surface",
                    perturbation=f"{label_a} → {label_b}",
                )
                hypotheses.append(h)

    if not hypotheses:
        hypotheses.append(Hypothesis(
            figure="no diff",
            ground="all opts identical",
            prediction="performance difference is from kernel execution, not opt choice",
            perturbation=f"{label_a} → {label_b}",
        ))

    return hypotheses


if __name__ == "__main__":
    print("=== Abduction: dim=4093 vs dim=4096 ===")
    print("Observation: 9.2x performance gap on softmax")
    print()

    hypotheses = abduct(
        lambda: Tensor.randn(64, 4093).softmax(),
        lambda: Tensor.randn(64, 4096).softmax(),
        label_a="dim=4093",
        label_b="dim=4096",
    )

    for i, h in enumerate(hypotheses):
        print(f"H{i}: {h.figure}")
        print(f"    ground: {h.ground}")
        print(f"    prediction: {h.prediction}")
        print(f"    perturbation: {h.perturbation}")
        print()

    print("=== Abduction: MV_ROWS_PER_THREAD=4 vs 16 ===")
    print("Observation: stride-aware matvec 62-105% BW gain")
    print()

    import os
    def matvec_4():
        os.environ["MV_ROWS_PER_THREAD"] = "4"
        t = (Tensor.randn(4096, 4096) @ Tensor.randn(1, 4096).T)
        del os.environ["MV_ROWS_PER_THREAD"]
        return t

    def matvec_16():
        return Tensor.randn(4096, 4096) @ Tensor.randn(1, 4096).T

    hypotheses = abduct(matvec_4, matvec_16, "MV_RPT=4", "MV_RPT=16")
    for i, h in enumerate(hypotheses):
        print(f"H{i}: {h.figure}")
        print(f"    prediction: {h.prediction}")
        print()
