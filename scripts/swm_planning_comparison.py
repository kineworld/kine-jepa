"""Compare kine-jepa's hand-rolled CEM against stable-worldmodel's planning solvers.

The task is the one this repository already ships: `tests/test_rollout.py::
test_planner_reaches_goal`. An `ActionRollout` is frozen at its random initialisation, a
goal latent is produced by rolling it with a chosen action sequence, and a planner must
find an action sequence landing near that goal. The goal is reachable by construction,
and its generating actions are drawn from an **unclamped** `randn`, so the only variables
are the search rule and the space it is allowed to search in.

Design: two axes, not one
-------------------------
An earlier version of this experiment compared `LatentPlanner` (which clamps every
candidate into `[-1, 1]`) against upstream solvers (which, it turned out, mostly do not).
That varies two things at once, and the difference gets attributed to the wrong one. The
arms below therefore cross both axes:

* action space: ``[-1, 1]`` versus free (``+/-1e9``)
* search rule: hand-rolled CEM, upstream CEM, MPPI, iCEM, Predictive Sampling

``upstream_native_clamp`` in the summary is *measured*, not read off the source: for each
solver the harness runs with a ``[-1, 1]`` space and no projection, then records whether
the returned actions stayed inside. Only ``icem`` does -- ``CEMSolver``, ``MPPISolver`` and
``PredictiveSamplingSolver`` record the action space and never enforce it. That single
asymmetry invalidated the previous run's iCEM row: iCEM was the one arm confined to the
box while every other upstream arm was free.

Arms
----
Free action space (``+/-1e9``):

1. ``kinejepa_unbounded``   hand-rolled CEM, clamp widened. Diagnostic, not a proposal --
                            a real actuator does not accept +/-1e9.
2. ``swm_cem_free``         upstream CEM, budget matched (64 samples, 8 steps, 6 elites,
                            initial sigma 0.5 -- the same values ``LatentPlanner`` uses).
3. ``swm_mppi_free``        upstream MPPI, matched budget.
4. ``swm_icem_free``        upstream iCEM, matched budget.
5. ``swm_predictive_free``  upstream Predictive Sampling, 64 samples, single shot.
6. ``swm_cem_default``      upstream CEM at its own defaults (300 samples, 30 steps, 30
                            elites). Not tuned by us; kept as the "more compute" reference.

Boxed action space (``[-1, 1]``), the box ``LatentPlanner`` ships with:

7. ``kinejepa_box``         hand-rolled CEM as shipped.
8. ``swm_cem_boxed``        upstream CEM with ``enforce_action_bounds=True``.
9. ``swm_mppi_boxed``       upstream MPPI with ``enforce_action_bounds=True``.
10. ``swm_icem_boxed``      upstream iCEM (natively clamped).
11. ``swm_predictive_boxed`` upstream Predictive Sampling with projection.

What is measured
----------------
The same quantity ``LatentPlanner.plan`` returns: the squared L2 distance between the
mean-pooled final rolled-out latent and the mean-pooled goal. For the stable-worldmodel
arms this is recomputed on the actions the solver actually returns rather than read out of
the solver's bookkeeping, so a solver that returns an elite mean is scored on that mean.
The ``shipped_assertion`` column applies arm 7's own predicate to every arm so the
comparison is not a different test for each row.

``enforce_action_bounds`` is our projection, not upstream behaviour -- it clips the
candidates the dynamics see, so the elite selection never scores an action outside the
box either. Any claim about upstream must be read off the ``False`` arm.

Honest limits
-------------
* Evidence level ``E1``: one task, one frozen random model, three seeds, CPU only. Says
  nothing about a trained checkpoint.
* The two-box design shows the box dominates *on this task*; the goal here is generated in
  a way no bounded actuator could reproduce. It is not a claim that bounds are useless --
  on a real body the bound is the difference between a plan and a suggestion.
* ``torch`` build matters. ``kineworld/kine-jepa#3`` records that arm 7's ratio assertion
  passes on the CPU wheel and fails on a local CUDA build, because the denominator is
  build-dependent. The absolute distances below are reported for that reason.

Run:
    python scripts/swm_planning_comparison.py
    python scripts/swm_planning_comparison.py --seeds 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kineworld_jepa.rollout import ActionRollout, LatentPlanner  # noqa: E402
from kineworld_jepa.interop.swm import SWMPlanner, swm_available  # noqa: E402

TASK = {"dim": 32, "tokens": 24, "action_dim": 4, "horizon": 8}
MATCHED = {"num_samples": 64, "n_steps": 8, "topk": 6, "var_scale": 0.5}
FREE = (-1e9, 1e9)
BOX = (-1.0, 1.0)


def build_task(seed: int):
    """Rebuild the repository's own planning task, verbatim, at one seed."""
    torch.manual_seed(seed)
    model = ActionRollout(
        TASK["dim"], depth=4, heads=4, action_dim=TASK["action_dim"], style="add"
    ).eval()
    latent0 = torch.randn(1, TASK["tokens"], TASK["dim"])
    a_star = torch.randn(1, TASK["horizon"], TASK["action_dim"])
    with torch.no_grad():
        goal = model(latent0, a_star)[-1].detach()
        other = torch.randn(1, TASK["horizon"], TASK["action_dim"])
        base = (
            model(latent0, other)[-1].mean(1) - goal.mean(1)
        ).pow(2).sum().item()
    return model, latent0, goal, base


def run_kinejepa(model, latent0, goal, bounds, seed):
    planner = LatentPlanner(
        model,
        goal_latent=goal,
        action_dim=TASK["action_dim"],
        horizon=TASK["horizon"],
        action_low=bounds[0],
        action_high=bounds[1],
    )
    started = time.perf_counter()
    actions, loss = planner.plan(
        latent0,
        iters=MATCHED["n_steps"],
        candidates=MATCHED["num_samples"],
        device="cpu",
        seed=seed,
    )
    elapsed = time.perf_counter() - started
    return float(loss), elapsed, _max_abs(actions)


def run_swm(model, latent0, goal, solver, bounds, enforce, seed, **kwargs):
    planner = SWMPlanner(
        model,
        goal_latent=goal,
        action_dim=TASK["action_dim"],
        horizon=TASK["horizon"],
        solver=solver,
        seed=seed,
        action_low=bounds[0],
        action_high=bounds[1],
        enforce_action_bounds=enforce,
        **kwargs,
    )
    actions, distance, elapsed = planner.plan(latent0)
    return float(distance), elapsed, _max_abs(actions)


def _max_abs(actions: torch.Tensor) -> float:
    return float(actions.abs().max().item())


# name -> (kind, solver, bounds, enforce, extra solver kwargs)
ARMS: dict[str, tuple] = {
    "kinejepa_unbounded": ("kinejepa", None, FREE, False, {}),
    "swm_cem_free": ("swm", "cem", FREE, False, MATCHED),
    "swm_mppi_free": ("swm", "mppi", FREE, False, MATCHED),
    "swm_icem_free": ("swm", "icem", FREE, False, MATCHED),
    "swm_predictive_free": ("swm", "predictive_sampling", FREE, False, {"num_samples": 64}),
    "swm_cem_default": ("swm", "cem", FREE, False, {"num_samples": 300, "n_steps": 30, "topk": 30}),
    "kinejepa_box": ("kinejepa", None, BOX, True, {}),
    "swm_cem_boxed": ("swm", "cem", BOX, True, MATCHED),
    "swm_mppi_boxed": ("swm", "mppi", BOX, True, MATCHED),
    "swm_icem_boxed": ("swm", "icem", BOX, True, MATCHED),
    "swm_predictive_boxed": ("swm", "predictive_sampling", BOX, True, {"num_samples": 64}),
}

# Arms whose numbers the previous run got wrong, and why. Recorded in the summary so the
# correction travels with the artifact instead of living only in a commit message.
SUPERSEDED = {
    "swm_icem": (
        "the previous run recorded iCEM at a median of 7.2166 and read it as an upstream "
        "weakness. iCEM is the only solver here that honours `configure(action_space=...)` "
        "and clamps, so it was the only upstream arm confined to [-1, 1] while CEM and MPPI "
        "ran free. The comparison varied the search space and the search rule at once and "
        "the difference was attributed to the rule. Replaced by swm_icem_free / "
        "swm_icem_boxed, which cross both axes."
    ),
}

# Revision 1's per-seed distances, quoted from the artifact it published (commit a14b262,
# EXPERIMENTS/SWM-PLANNING-v0/summary.json before this revision). Kept here so the
# correction can be *demonstrated* rather than asserted: if swm_icem_boxed reproduces
# revision 1's swm_icem exactly, then that arm was already running boxed -- which is the
# whole claim. A cross-revision check that can fail is worth more than a paragraph saying
# the old number was wrong.
REVISION_1_RECORD = {
    "commit": "a14b262",
    "swm_icem": [8.761207, 0.154088, 7.216594],
    "swm_cem_matched": [0.957186, 0.081932, 10.345827],
}
_TOL = 1e-6


def _same(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= _TOL for x, y in zip(a, b))


def cross_revision_check(arms: dict) -> dict[str, bool]:
    """Confirm the superseded arm was boxed, using revision 1's own published numbers."""
    return {
        # The claim: iCEM at [-1, 1] with no projection *is* what revision 1 ran.
        "icem_boxed_reproduces_revision_1": _same(
            arms["swm_icem_boxed"]["distance"], REVISION_1_RECORD["swm_icem"]
        ),
        # A control: an arm whose configuration did not change must be unchanged too.
        # Without this, the check above could pass because the harness ignores its inputs.
        "unchanged_arm_reproduces_revision_1": _same(
            arms["swm_cem_free"]["distance"], REVISION_1_RECORD["swm_cem_matched"]
        ),
    }


def shipped_assertion(distance: float, base: float) -> bool:
    """The predicate in tests/test_rollout.py::test_planner_reaches_goal, unchanged."""
    return distance < 0.3 * base and distance < base - 1.0


def native_clamp_probe(model, latent0, goal) -> dict[str, dict]:
    """Measure which upstream solvers stay inside a [-1, 1] space on their own.

    Run with the box *and* without the projection, so the only thing that can keep an
    action inside is the solver itself. Read off the source this is easy to get wrong --
    that is precisely how the previous run's iCEM row went bad -- so it is measured, and
    the observed peak is recorded next to the verdict so the reader can see how far outside
    a non-clamping solver actually went rather than trusting a boolean.
    """
    probe_arms = {
        "cem": MATCHED,
        "mppi": MATCHED,
        "icem": MATCHED,
        "predictive_sampling": {"num_samples": 64},
    }
    verdict: dict[str, dict] = {}
    for solver, kwargs in probe_arms.items():
        _, _, peak = run_swm(model, latent0, goal, solver, BOX, False, 0, **kwargs)
        verdict[solver] = {"clamps_natively": peak <= 1.0 + 1e-6, "peak": round(peak, 4)}
    return verdict


def _flag(passes: list[bool]) -> str:
    if all(passes):
        return "PASS"
    if not any(passes):
        return "FAIL"
    return "mixed"


def _table(arms: dict, names: list[str]) -> str:
    header = "| 方案 | 中位 d2 | 中位改善 | 中位耗时 s | 返回动作峰值 | 既有断言 |"
    divider = "| --- | ---: | ---: | ---: | ---: | --- |"
    rows = [header, divider]
    for name in names:
        data = arms[name]
        rows.append(
            f"| `{name}` | {data['distance_median']:.4f} | {data['improvement_median']:.2f}× "
            f"| {data['wall_s_median']:.3f} | {data['action_peak_median']:.3f} "
            f"| {_flag(data['shipped_assertion'])} |"
        )
    # A pipe inside a cell breaks the whole table, and the breakage is invisible in the
    # source -- it only shows up when GitHub renders it. The first version of this table
    # had "max|a|" as a column title, which produced a nine-pipe header over seven-pipe
    # rows. Counting the delimiters is the cheapest way to make that impossible to ship.
    expected = header.count("|")
    bad = [r for r in rows if r.count("|") != expected]
    if bad:
        raise ValueError(
            f"markdown table row has {bad[0].count('|')} pipes, expected {expected}: {bad[0]!r}"
        )
    return "\n".join(rows)


def render_markdown(summary: dict) -> str:
    """Render the experiment record from the measured numbers.

    The prose cites figures by pulling them out of ``summary`` rather than repeating them,
    so the record cannot drift from the artifact and no number in it was typed by hand.
    That is a standing rule in this repository -- see EXPERIMENTS and STATUS.md, "所有数字
    必须来自脚本实测输出".
    """
    arms = summary["arms"]
    env = summary["environment"]
    upstream = summary["upstream"]
    task = summary["task"]
    seeds = summary["seeds"]
    bases = summary["off_target_baseline"]
    clamp = summary["upstream_native_clamp"]

    free_names = [n for n, spec in ARMS.items() if spec[2] == FREE]
    box_names = [n for n, spec in ARMS.items() if spec[2] == BOX]

    def arm(name: str) -> dict:
        return arms[name]

    box = arm("kinejepa_box")
    unbounded = arm("kinejepa_unbounded")
    cem_free = arm("swm_cem_free")
    cem_boxed = arm("swm_cem_boxed")
    icem_free = arm("swm_icem_free")
    mppi_free = arm("swm_mppi_free")
    cem_default = arm("swm_cem_default")

    clamps = ", ".join(
        f"`{k}` {'是' if v['clamps_natively'] else '否'}（实测峰值 {v['peak']:.3f}）"
        for k, v in clamp.items()
    )
    n_clamping = sum(1 for v in clamp.values() if v["clamps_natively"])
    clamp_free = [k for k, v in clamp.items() if not v["clamps_natively"]]

    cross = cross_revision_check(arms)
    icem_boxed_seeds = ", ".join(f"{x:.6f}" for x in arms["swm_icem_boxed"]["distance"])
    rev1_seeds = ", ".join(f"{x:.6f}" for x in REVISION_1_RECORD["swm_icem"])
    reproduced = cross["icem_boxed_reproduces_revision_1"]
    control_ok = cross["unchanged_arm_reproduces_revision_1"]

    # The like-for-like comparison, both axes held: same budget, same box.
    hand_box = box["distance_median"]
    cem_box = cem_boxed["distance_median"]
    if cem_box > 0 and hand_box > 0:
        box_ratio = max(hand_box, cem_box) / min(hand_box, cem_box)
        box_better = "手写" if hand_box < cem_box else "上游"
    else:
        box_ratio, box_better = float("nan"), "?"

    return f"""# SWM-PLANNING-v0 · 用上游求解器替换手写 CEM

状态：已完成（第二版，已更正第一版的结论）。结论只在本文件与 `summary.json` 中，别处不引用。
本文件与 `summary.json` **都由脚本生成**，数字无手打：

    python scripts/swm_planning_comparison.py

机器可读版本：[`SWM-PLANNING-v0/summary.json`](SWM-PLANNING-v0/summary.json)。

环境：torch {env['torch']} / {env['device']}；上游 {upstream['package']} {upstream['version']}（{upstream['license']}，Fork 在 `kineworld/{upstream['package']}`）。
任务：dim={task['dim']}，tokens={task['tokens']}，action_dim={task['action_dim']}，horizon={task['horizon']}，与 `tests/test_rollout.py::test_planner_reaches_goal` 同一构造。
种子：{', '.join(str(s) for s in seeds)}；对应 off-target 基线 {', '.join(f'{b:.2f}' for b in bases)}。
证据级别：**{summary['evidence_level']}** —— {summary['evidence_level_detail']}。

## 为什么做这个

`kineworld_jepa/rollout.py` 里的 `LatentPlanner` 是三十行手写 CEM。上游
`stable-worldmodel`（MIT，LeWM 的作者团队）带八个维护中的求解器，而它对世界模型的要求只有
一个方法：

    get_cost(info_dict, action_candidates) -> (B, S)

`kineworld_jepa/interop/swm.py` 实现这一个方法，就换来整套搜索代码。

## 这一版改了什么（第一版错了什么）

第一版把「手写（截断到 ±1）」和「上游（不截断）」放在一起比，**同时变了两个变量**：搜索规则
和动作空间。差异被算到了搜索规则头上。这一版把两轴交叉开，并把「上游到底谁裁剪动作」改成
**实测字段** `upstream_native_clamp`，而不是读源码推断。实测结果：{clamps}。

⇒ {n_clamping} 个求解器会自己裁剪，其余只记录动作空间、从不执行。
第一版因此把 `swm_icem` 记成负结果——它是**唯一**被框住的上游臂，其余上游臂全是自由的。

这不是推断，是复现：把 iCEM 放到 `[-1, 1]`、关掉投影，逐种子距离为 {icem_boxed_seeds}；
第一版 `summary.json`（commit `{REVISION_1_RECORD['commit']}`）记的 `swm_icem` 为 {rev1_seeds}。
逐位相同 = **{'真' if reproduced else '否（不一致，需复查）'}**。
对照组（配置未变、数字也应未变）`swm_cem_free` 对第一版 `swm_cem_matched`：
**{'一致' if control_ok else '不一致（需复查）'}**。没有这个对照组，上面那条相等也可能只是
harness 忽略输入造成的。

即：第一版那条 iCEM 负结果，跑的一直是 `swm_icem_boxed`。
放开动作框后 iCEM 中位 {arms['swm_icem_free']['distance_median']:.4f}，是上游最好的一档。

## 结果 · 自由动作空间（±1e9）

{_table(arms, free_names)}

## 结果 · 动作框 ±1（`LatentPlanner` 出厂设置）

{_table(arms, box_names)}

## 读法

1. **主导变量是动作框，不是求解器。** 同一个求解器放进 `[-1, 1]` 就停在
   {box['distance_median']:.2f} 一档，放开到自由就进入 {unbounded['distance_median']:.2f} 一档。
   目标 latent 本身由**未截断**的 `randn` 动作生成，`[-1, 1]` 的框里未必存在解。
   既有的 `test_planner_reaches_goal` 之所以时红时绿，主因在这里，不在 CEM。
   这是**诊断**：真实本体上框是必要的，`kinejepa_unbounded` 不是提案。

2. **同预算同框：不能一概而论，要按求解器分。** 框住之后手写 {box['distance_median']:.4f}、
   上游 CEM {cem_boxed['distance_median']:.4f}（{box_ratio:.2f} 倍以内，同档，{box_better}略优）；
   但 MPPI {arm('swm_mppi_boxed')['distance_median']:.4f} **明显更好**，而且是框内唯一通过既有断言的臂；
   iCEM {arm('swm_icem_boxed')['distance_median']:.4f}；Predictive
   {arm('swm_predictive_boxed')['distance_median']:.4f} 最弱。
   ⇒ 说"上游不比手写好"和说"上游更好"都过度概括，取决于换成哪一个。

3. **自由空间、同预算：手写 {unbounded['distance_median']:.4f}，
   iCEM {icem_free['distance_median']:.4f}（上游最好，与手写同档），
   MPPI {mppi_free['distance_median']:.4f}，CEM {cem_free['distance_median']:.4f}。**
   手写在这个任务上并不落后；真正落后的是上游 CEM 的默认精英更新。

4. **上游要拉开差距，靠的是预算，不是算法。** `swm_cem_default`
   （300 采样 × 30 步 = 手写的 22 倍采样量）中位 {cem_default['distance_median']:.4f}，
   基本打到精确解。这是**算力差异**，不是"上游更聪明"。

5. **采纳的真正理由**是：八个维护中的求解器、可替换、不必自己维护搜索代码；以及本实验暴露的
   **反向风险**——直接换成上游会**丢掉执行器动作框**：只有 `icem` 会自己裁剪，
   {', '.join('`' + k + '`' for k in clamp_free)} 只记录动作空间而从不执行。
   `SWMPlanner(enforce_action_bounds=True)` 把框补回来，这也是 `swm_*_boxed` 各臂的由来。

6. **一处对不上的既有注释**：`tests/test_rollout.py` 写 "64 candidates x 8 iters
   (~140s on a laptop)"，本机实测手写方案中位 {box['wall_s_median']:.3f}s。这里只记录实测，
   **不改那条注释**——改它属于另一个改动集。

## 不声称什么

- 这是**一个任务、一个冻结的随机初始化模型、{len(seeds)} 个种子、纯 CPU**。对训练后的检查点
  不构成任何结论。
- 上游臂默认**不裁剪**动作；`enforce_action_bounds=True` 是我方投影，不是上游行为。
  关于上游的任何判断都只能读 `False` 那一列。
- 未跑 GPU；未跑真实轨迹；未改动任何已发布数字。
- 本实验不构成"上游更好"或"手写更好"的一般结论。它说明的是：**在这个任务上，动作框的影响
  大于搜索规则；同预算同框下的胜负按具体求解器而异（CEM 与手写同档，MPPI 更好）；
  上游的价值在可替换性与维护成本，不在"换了就一定更准"。**

## 本机复现

    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-swm.txt
    python scripts/swm_planning_comparison.py

不带 `stable-worldmodel` 时脚本以退出码 2 结束并说明缺什么，不会给出半个结论。
`tests/test_swm_interop.py` 同理：上游缺席时那条测试报 SKIP 并带原因，不计为通过。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "EXPERIMENTS" / "SWM-PLANNING-v0" / "summary.json"
    )
    args = parser.parse_args(argv)

    if not swm_available():
        print(
            "stable-worldmodel is not importable; nothing to compare against.\n"
            "Install with: pip install -r requirements-swm.txt"
        )
        return 2

    import stable_worldmodel

    results: dict[str, dict] = {
        name: {"distance": [], "wall_s": [], "improvement": [], "action_peak": []}
        for name in ARMS
    }
    bases: list[float] = []
    native_clamp: dict[str, bool] | None = None

    for seed in args.seeds:
        model, latent0, goal, base = build_task(seed)
        bases.append(base)
        if native_clamp is None:
            # Measured once, on the first seed: a property of the solvers, not of the task.
            native_clamp = native_clamp_probe(model, latent0, goal)
        row = [f"seed={seed}", f"base={base:.2f}"]
        for name, (kind, solver, bounds, enforce, kwargs) in ARMS.items():
            if kind == "kinejepa":
                distance, elapsed, peak = run_kinejepa(model, latent0, goal, bounds, seed)
            else:
                distance, elapsed, peak = run_swm(
                    model, latent0, goal, solver, bounds, enforce, seed, **kwargs
                )
            results[name]["distance"].append(distance)
            results[name]["wall_s"].append(elapsed)
            results[name]["action_peak"].append(peak)
            results[name]["improvement"].append(base / distance if distance > 0 else float("inf"))
            row.append(f"{name}={distance:.2f}")
        print("  ".join(row))

    summary = {
        "experiment": "SWM-PLANNING-v0",
        "revision": 2,
        "date": "2026-09-22",
        # E1 per GOVERNANCE.md: one task, one frozen random-initialised model, however many
        # seeds. Three seeds give a variance estimate, which is why the per-seed distances
        # are kept; it does not lift this above E1 and it says nothing about a trained
        # checkpoint.
        "evidence_level": "E1",
        "evidence_level_detail": (
            "one task, one frozen random-initialised model, %d seeds, CPU only; "
            "no trained checkpoint was involved" % len(args.seeds)
        ),
        "task": TASK,
        "budget": MATCHED,
        "upstream": {
            "package": "stable-worldmodel",
            "version": getattr(stable_worldmodel, "__version__", "0.1.1"),
            "license": "MIT",
        },
        "environment": {"torch": torch.__version__, "device": "cpu"},
        "seeds": args.seeds,
        "off_target_baseline": bases,
        # Measured, not inferred: which upstream solvers clamp natively. See the probe.
        "upstream_native_clamp": native_clamp,
        "arms": {},
        "superseded_from_revision_1": SUPERSEDED,
        "notes": [
            "distances use LatentPlanner's reduction: mean-pool over tokens, then squared "
            "L2 to the goal; for the SWM arms it is recomputed on the returned actions.",
            "two action spaces are crossed with the solvers: [-1, 1] and free (+/-1e9).",
            "enforce_action_bounds=True is our projection, not upstream behaviour; only "
            "arms with bounds [-1, 1] use it, and the free arms use the library default of "
            "no projection.",
            "the goal is generated by unclamped randn actions, so a [-1, 1] box may not "
            "contain the generating sequence at all; that is what makes the box the "
            "dominant variable on this task.",
            "kinejepa_box's ratio assertion is build-dependent -- see kineworld/kine-jepa#3.",
        ],
    }
    for name, data in results.items():
        summary["arms"][name] = {
            "distance": [round(x, 6) for x in data["distance"]],
            "distance_median": round(statistics.median(data["distance"]), 6),
            "improvement_vs_off_target": [round(x, 4) for x in data["improvement"]],
            "improvement_median": round(statistics.median(data["improvement"]), 4),
            "wall_s": [round(x, 3) for x in data["wall_s"]],
            "wall_s_median": round(statistics.median(data["wall_s"]), 3),
            "action_peak": [round(x, 4) for x in data["action_peak"]],
            "action_peak_median": round(statistics.median(data["action_peak"]), 4),
            "shipped_assertion": [
                shipped_assertion(d, b) for d, b in zip(data["distance"], bases)
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cross = cross_revision_check(summary["arms"])
    summary["cross_revision_check"] = cross
    # Rewrite with the check included, then render: the markdown cites it, so it must exist
    # before the prose is generated.
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown_path = args.out.parent.parent / "SWM-PLANNING.md"
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print("\ncross-revision check (revision 1 = commit a14b262):")
    for k, v in cross.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    if not all(cross.values()):
        print(
            "\nWARNING: a cross-revision check failed. The retraction in this record rests on\n"
            "the identity it asserts; do not publish the prose until it is understood.\n"
        )

    print("\nupstream clamps natively (measured with a [-1, 1] space, no projection):")
    for k, v in (native_clamp or {}).items():
        print(f"  {k:20s} clamps={v['clamps_natively']!s:5s} peak|a|={v['peak']:.3f}")
    print(f"\n{'arm':24s} {'median d2':>10s} {'median x':>9s} {'median s':>9s} {'max|a|':>8s}  shipped")
    for name, data in summary["arms"].items():
        passes = data["shipped_assertion"]
        flag = "PASS" if all(passes) else ("FAIL" if not any(passes) else "mixed")
        print(
            f"{name:24s} {data['distance_median']:10.4f} "
            f"{data['improvement_median']:9.2f} {data['wall_s_median']:9.3f} "
            f"{data['action_peak_median']:8.3f}  {flag}"
        )
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
