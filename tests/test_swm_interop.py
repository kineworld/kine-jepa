"""Tests for the stable-worldmodel interop shim.

Two kinds of test live here and they have different obligations.

The first three run **everywhere**, including CI, because they touch only
``SWMCostModel``, which imports no third-party planner. They pin the one thing that can
silently corrupt every number downstream: the adapter's cost must equal the distance
``LatentPlanner._distance`` already computes, and the goal's own generating actions must
cost zero. An off-by-one on the action axis, a pool over the wrong dimension, or a
mean-vs-sum reduction drift each turn one of them red.

The rest exercise ``stable-worldmodel`` itself and therefore **skip, with a reason, when
the package is absent** -- CI installs ``requirements.txt`` only. A skip is reported as a
skip; it is never counted as a pass. The structural test near the top exists to stop the
other failure mode: someone moving the upstream import to module scope, which would turn
CI red with ``ModuleNotFoundError`` instead of a clear skip.

``test_upstream_solvers_do_not_clamp_unless_asked`` is a regression guard for a trap that
already cost this repository one wrong published result: only ``icem`` honours the action
space handed to ``configure``, so comparing ``cem`` or ``mppi`` against ``LatentPlanner``
(as shipped, which clamps) varies the search space as well as the search rule. The
distances are then attributed to the wrong one. That test fails if the wrapper ever stops
being explicit about which of the two it is doing.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kineworld_jepa.rollout import ActionRollout, LatentPlanner  # noqa: E402
from kineworld_jepa.interop.swm import (  # noqa: E402
    SWMCostModel,
    SWMPlanner,
    swm_available,
)

PASSED = []
ROOT = Path(__file__).resolve().parent.parent
TASK = {"dim": 32, "tokens": 24, "action_dim": 4, "horizon": 8}
MATCHED = {"num_samples": 64, "n_steps": 8, "topk": 6, "var_scale": 0.5}


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        raise AssertionError(name)
    PASSED.append(name)


def build_task(seed=2):
    """The planning task from test_rollout.py::test_planner_reaches_goal, unchanged."""
    torch.manual_seed(seed)
    model = ActionRollout(
        TASK["dim"], depth=4, heads=4, action_dim=TASK["action_dim"], style="add"
    ).eval()
    latent0 = torch.randn(1, TASK["tokens"], TASK["dim"])
    a_star = torch.randn(1, TASK["horizon"], TASK["action_dim"])
    with torch.no_grad():
        goal = model(latent0, a_star)[-1].detach()
        other = torch.randn(1, TASK["horizon"], TASK["action_dim"])
        base = (model(latent0, other)[-1].mean(1) - goal.mean(1)).pow(2).sum().item()
    return model, latent0, goal, a_star, base


def test_adapter_has_no_module_level_upstream_import():
    """The shim must import cleanly on an interpreter without stable-worldmodel.

    CI installs requirements.txt only. If the upstream import moves to module scope, this
    file stops being collectable and the failure surfaces as a confusing import error
    instead of the honest skip in test_swm_solver_beats_hand_rolled_cem. Parsed with ast
    rather than grepped, so a mention inside a docstring or a string literal -- of which
    there are many here -- does not trip it.
    """
    source = (ROOT / "kineworld_jepa" / "interop" / "swm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in tree.body:  # module scope only; imports inside functions are the intent
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if "stable_worldmodel" in a.name]
        elif isinstance(node, ast.ImportFrom):
            if node.module and "stable_worldmodel" in node.module:
                offenders.append(node.module)
    check("test_adapter_has_no_module_level_upstream_import", not offenders,
          f"module-level upstream imports: {offenders}")


def test_adapter_cost_equals_latent_planner_distance():
    """SWMCostModel.criterion and LatentPlanner._distance must agree elementwise.

    Both reduce by mean over tokens and then sum the squared error over features. The
    adapter's path runs S candidates through one batched rollout; the reference loops. So
    the two cannot be bit-identical in float32 -- the tolerance is the rounding noise of
    that difference, not a licence for a different formula. A reduction of mean instead of
    sum lands three orders of magnitude outside it.
    """
    model, latent0, goal, _, _ = build_task(2)
    cost_model = SWMCostModel(model, action_dim=TASK["action_dim"])
    torch.manual_seed(99)
    candidates = torch.randn(1, 5, TASK["horizon"], TASK["action_dim"])
    with torch.no_grad():
        cost = cost_model.get_cost({"latent": latent0, "goal_latent": goal}, candidates)
        reference = torch.cat([
            (model(latent0, candidates[:, s], horizon=TASK["horizon"])[-1].mean(1)
             - goal.mean(1)).pow(2).sum(-1)
            for s in range(candidates.shape[1])
        ])
    delta = (cost[0] - reference).abs().max().item()
    check("test_adapter_cost_equals_latent_planner_distance",
          cost.shape == (1, 5) and delta < 1e-3,
          f"max|adapter - planner| = {delta:.2e} over {candidates.shape[1]} candidates")


def test_goal_generating_actions_cost_zero():
    """Negative control: the actions that produced the goal must cost zero.

    The goal is, by construction, the rollout of ``a_star``. Any adapter that mis-reads
    the action axis, drops a step, or pools the wrong tensor still returns a plausible-
    looking cost for arbitrary candidates -- but it cannot return zero here. This is the
    assertion that fails loudly on exactly the bug class it was written for.
    """
    model, latent0, goal, a_star, base = build_task(2)
    cost_model = SWMCostModel(model, action_dim=TASK["action_dim"])
    with torch.no_grad():
        cost_star = cost_model.get_cost(
            {"latent": latent0, "goal_latent": goal}, a_star.unsqueeze(1)
        )[0, 0].item()
        cost_offtarget = cost_model.get_cost(
            {"latent": latent0, "goal_latent": goal}, torch.randn_like(a_star).unsqueeze(1)
        )[0, 0].item()
    check("test_goal_generating_actions_cost_zero",
          cost_star < 1e-4 < cost_offtarget,
          f"cost(a_star)={cost_star:.2e} cost(random)={cost_offtarget:.2f} base={base:.2f}")


def test_swm_solvers_reach_the_goal_at_matched_budget():
    """Matched budget, same task: upstream CEM and MPPI reach the goal.

    This test asserts **reachability**, not superiority. It deliberately does not claim to
    beat ``LatentPlanner``: measured at this budget the two are in the same tier, and at
    matched action space neither wins by much (see EXPERIMENTS/SWM-PLANNING-v0, revision
    2). Its earlier name said "beats_hand_rolled_cem" while asserting no such thing, which
    is a worse failure than a wrong threshold -- a reader trusts the name. The hand-rolled
    distance is still measured and printed so a regression in either direction is visible.

    Bounds are set well clear of the measured values (CEM 0.96, MPPI 0.30 against an
    off-target baseline of 25.88 at this seed) so the test is not a restatement of one run.
    It skips, with a reason, when the upstream package is absent rather than reporting a
    pass it did not earn.
    """
    if not swm_available():
        raise unittest.SkipTest(
            "stable-worldmodel not installed; pip install -r requirements-swm.txt"
        )

    model, latent0, goal, _, base = build_task(2)
    observed = {}
    for solver in ("cem", "mppi"):
        planner = SWMPlanner(
            model,
            goal_latent=goal,
            action_dim=TASK["action_dim"],
            horizon=TASK["horizon"],
            solver=solver,
            seed=2,
            **MATCHED,
        )
        _, distance, _ = planner.plan(latent0)
        observed[solver] = distance

    hand_rolled = LatentPlanner(
        model, goal_latent=goal, action_dim=TASK["action_dim"], horizon=TASK["horizon"]
    )
    _, hand_loss = hand_rolled.plan(
        latent0, iters=MATCHED["n_steps"], candidates=MATCHED["num_samples"],
        device="cpu", seed=2,
    )

    ok = all(d < 2.0 and d < 0.1 * base for d in observed.values())
    check("test_swm_solvers_reach_the_goal_at_matched_budget", ok,
          f"base={base:.2f} swm_cem={observed['cem']:.2f} swm_mppi={observed['mppi']:.2f} "
          f"kinejepa_cem={hand_loss:.2f} (reachability only; no superiority claimed)")


def test_upstream_solvers_do_not_clamp_unless_asked():
    """Only ``icem`` honours the action space; the wrapper must say which mode it is in.

    Two guarantees, and the second is the one that catches a real published error:

    1. With ``enforce_action_bounds=True`` every solver's returned actions are inside
       ``[-1, 1]``. A ``LatentPlanner`` comparison in this mode is an equal-box one.
    2. With the flag off -- the library's own behaviour -- at least one solver leaves the
       box. That is what makes "the box, not the search rule" an explanation rather than a
       hypothesis, and it is why the first experiment's iCEM row was wrong: iCEM clamps by
       itself, so it was the only upstream arm that never left the box while CEM and MPPI
       did.

    The margin on (2) is large (``cem`` peaks above 5 against a bound of 1), so this does
    not turn red on ordinary sampling noise. The determinism comes from fixed seeds, as
    everywhere else in this suite.
    """
    if not swm_available():
        raise unittest.SkipTest(
            "stable-worldmodel not installed; pip install -r requirements-swm.txt"
        )

    model, latent0, goal, _, _ = build_task(0)
    probe = {"cem": MATCHED, "mppi": MATCHED, "predictive_sampling": {"num_samples": 64}}

    free_peaks = {}
    boxed_peaks = {}
    for solver, kwargs in probe.items():
        _, _, free_peaks[solver] = _peak(
            model, latent0, goal, solver, -1.0, 1.0, False, kwargs
        )
        _, _, boxed_peaks[solver] = _peak(
            model, latent0, goal, solver, -1.0, 1.0, True, kwargs
        )

    boxed_ok = all(p <= 1.0 + 1e-6 for p in boxed_peaks.values())
    check("test_upstream_solvers_clamp_when_asked", boxed_ok,
          "boxed peaks: " + ", ".join(f"{k}={v:.3f}" for k, v in boxed_peaks.items()))

    escapes = {k: v for k, v in free_peaks.items() if v > 1.0 + 1e-6}
    check("test_upstream_solvers_ignore_action_bounds_by_default", bool(escapes),
          "unclamped peaks: " + ", ".join(f"{k}={v:.3f}" for k, v in free_peaks.items()))


def _peak(model, latent0, goal, solver, low, high, enforce, kwargs):
    """Run one solver and return (distance, seconds, max|action|)."""
    planner = SWMPlanner(
        model,
        goal_latent=goal,
        action_dim=TASK["action_dim"],
        horizon=TASK["horizon"],
        solver=solver,
        seed=0,
        action_low=low,
        action_high=high,
        enforce_action_bounds=enforce,
        **kwargs,
    )
    actions, distance, elapsed = planner.plan(latent0)
    return distance, elapsed, float(actions.abs().max().item())


if __name__ == "__main__":
    test_adapter_has_no_module_level_upstream_import()
    test_adapter_cost_equals_latent_planner_distance()
    test_goal_generating_actions_cost_zero()
    test_swm_solvers_reach_the_goal_at_matched_budget()
    test_upstream_solvers_do_not_clamp_unless_asked()
    print(f"\nall {len(PASSED)} swm-interop tests passed")
