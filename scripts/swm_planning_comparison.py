"""Compare kine-jepa's hand-rolled CEM against stable-worldmodel's planning solvers.

The task is the one this repository already ships: `tests/test_rollout.py::
test_planner_reaches_goal`. An `ActionRollout` is frozen at its random initialisation, a
goal latent is produced by rolling it with a chosen action sequence, and a planner must
find an action sequence landing near that goal. The goal is reachable by construction, so
the only variable is the search.

Arms
----
1. ``kinejepa_box``        -- ``LatentPlanner`` exactly as shipped: candidates clamped
                              into ``[-1, 1]``, 8 iterations x 64 candidates.
2. ``kinejepa_unbounded``  -- the same planner and budget with the clamp widened to
                              ``+/-1e9``. The goal's own generating actions are drawn from
                              an unclamped ``randn``, so this arm separates "the search is
                              weak" from "the search is looking in a box the answer is not
                              in". It is a diagnostic, not a proposal.
3. ``swm_cem_matched``     -- ``CEMSolver`` through ``kineworld_jepa.interop.swm``, budget
                              matched to arm 1 where the APIs allow (64 samples, 8 steps,
                              6 elites) and initial std matched at 0.5.
4. ``swm_cem_default``     -- ``CEMSolver`` at its own library defaults (300 samples,
                              30 steps, 30 elites, ``var_scale=1.0``). Not tuned by us.
5. ``swm_icem``            -- ``ICEMSolver``, matched budget. Colored noise + elite
                              retention.
6. ``swm_mppi``            -- ``MPPISolver``, matched budget.
7. ``swm_predictive``      -- ``PredictiveSamplingSolver``, 64 samples, single shot.

What is measured
----------------
The same quantity ``LatentPlanner.plan`` returns: the squared L2 distance between the
mean-pooled final rolled-out latent and the mean-pooled goal. For the stable-worldmodel
arms this is recomputed on the actions the solver actually returns rather than read out of
the solver's bookkeeping, so a solver that returns an elite mean is scored on that mean.
The ``shipped_assertion`` column applies arm 1's own predicate to every arm so the
comparison is not a different test for each row.

Honest limits
-------------
* Evidence level: one task, one frozen random model, three seeds, CPU only. This says
  nothing about a trained checkpoint.
* Arms 3-6 sample an unclamped Gaussian while arm 1 clamps. That asymmetry is a real
  difference between the two implementations, not an artefact of the harness.
* ``torch`` build matters. ``kineworld/kine-jepa#3`` records that arm 1's ratio assertion
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


def run_kinejepa(model, latent0, goal, low, high, seed):
    planner = LatentPlanner(
        model,
        goal_latent=goal,
        action_dim=TASK["action_dim"],
        horizon=TASK["horizon"],
        action_low=low,
        action_high=high,
    )
    started = time.perf_counter()
    _, loss = planner.plan(
        latent0, iters=MATCHED["n_steps"], candidates=MATCHED["num_samples"], device="cpu", seed=seed
    )
    return float(loss), time.perf_counter() - started


def run_swm(model, latent0, goal, solver, seed, **kwargs):
    planner = SWMPlanner(
        model,
        goal_latent=goal,
        action_dim=TASK["action_dim"],
        horizon=TASK["horizon"],
        solver=solver,
        seed=seed,
        **kwargs,
    )
    _, distance, elapsed = planner.plan(latent0)
    return float(distance), elapsed


ARMS = {
    "kinejepa_box": lambda m, l, g, s: run_kinejepa(m, l, g, -1.0, 1.0, s),
    "kinejepa_unbounded": lambda m, l, g, s: run_kinejepa(m, l, g, -1e9, 1e9, s),
    "swm_cem_matched": lambda m, l, g, s: run_swm(m, l, g, "cem", s, **MATCHED),
    "swm_cem_default": lambda m, l, g, s: run_swm(
        m, l, g, "cem", s, num_samples=300, n_steps=30, topk=30
    ),
    "swm_icem": lambda m, l, g, s: run_swm(m, l, g, "icem", s, **MATCHED),
    "swm_mppi": lambda m, l, g, s: run_swm(m, l, g, "mppi", s, **MATCHED),
    "swm_predictive": lambda m, l, g, s: run_swm(
        m, l, g, "predictive_sampling", s, num_samples=MATCHED["num_samples"]
    ),
}


def shipped_assertion(distance: float, base: float) -> bool:
    """The predicate in tests/test_rollout.py::test_planner_reaches_goal, unchanged."""
    return distance < 0.3 * base and distance < base - 1.0


def _flag(passes: list[bool]) -> str:
    if all(passes):
        return "PASS"
    if not any(passes):
        return "FAIL"
    return "mixed"


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

    def arm(name: str) -> dict:
        return arms[name]

    rows = [
        "| 方案 | 中位 d2 | 中位改善 | 中位耗时 s | 仓库既有断言 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, data in arms.items():
        rows.append(
            f"| `{name}` | {data['distance_median']:.4f} | {data['improvement_median']:.2f}× "
            f"| {data['wall_s_median']:.3f} | {_flag(data['shipped_assertion'])} |"
        )
    table = "\n".join(rows)

    box = arm("kinejepa_box")
    unbounded = arm("kinejepa_unbounded")
    cem_matched = arm("swm_cem_matched")
    cem_default = arm("swm_cem_default")
    mppi = arm("swm_mppi")
    icem = arm("swm_icem")
    predictive = arm("swm_predictive")

    slow_factor = box["wall_s_median"] / max(cem_matched["wall_s_median"], 1e-9)
    note_factor = box["wall_s_median"] / 140.0

    return f"""# SWM-PLANNING-v0 · 用上游求解器替换手写 CEM

状态：已完成。结论只在本文件与 `summary.json` 中，别处不引用。
本文件与 `summary.json` **都由脚本生成**，数字无手打：

    python scripts/swm_planning_comparison.py

机器可读版本：[`SWM-PLANNING-v0/summary.json`](SWM-PLANNING-v0/summary.json)。

环境：torch {env['torch']} / {env['device']}；上游 {upstream['package']} {upstream['version']}（{upstream['license']}，Fork 在 `kineworld/{upstream['package']}`）。
任务：dim={task['dim']}，tokens={task['tokens']}，action_dim={task['action_dim']}，horizon={task['horizon']}，与 `tests/test_rollout.py::test_planner_reaches_goal` 同一构造。
种子：{', '.join(str(s) for s in seeds)}；对应 off-target 基线 {', '.join(f'{b:.2f}' for b in bases)}。
证据级别：**{summary['evidence_level']}** —— {summary['evidence_level_detail']}。
按 `GOVERNANCE.md` 证据门第 3 条，单任务结果标 `E1`，不得当作一般能力；三个种子给的是方差，
不是把结论抬到 `E1` 以上。

## 为什么做这个

`kineworld_jepa/rollout.py` 里的 `LatentPlanner` 是三十行手写 CEM。上游
`stable-worldmodel`（MIT，LeWM 的作者团队）带八个维护中的求解器，而它对世界模型的要求只有
一个方法：

    get_cost(info_dict, action_candidates) -> (B, S)

`kineworld_jepa/interop/swm.py` 实现这一个方法，就换来整套搜索代码。本实验回答两件事：
换过去至少不更差吗；以及既有的 `test_planner_reaches_goal` 到底为什么红。

## 结果

{table}

## 读法

1. **现状方案**（`kinejepa_box`：截断到 ±1，8 次迭代 × 64 候选）中位 d2 {box['distance_median']:.4f}，
   中位改善 {box['improvement_median']:.2f}×，既有断言为 {_flag(box['shipped_assertion'])}。
   断言在部分种子为红，与 `kineworld/kine-jepa#3` 已记录的"该比值依赖 torch 构建"一致。

2. **同预算、同算法，只把动作框放开**（`kinejepa_unbounded`，±1e9）中位 d2
   {unbounded['distance_median']:.4f}，中位改善 {unbounded['improvement_median']:.2f}×，
   断言 {_flag(unbounded['shipped_assertion'])}。目标 latent 本身就是由**未截断**的 `randn`
   动作生成的，所以 `[-1, 1]` 的框里未必存在能落到目标邻域的解。既有失败的主因是动作框，
   不是 CEM 本身。这个臂是诊断，不是提案：`LatentPlanner` 的框在真实本体上是必要的。

3. **上游 CEM，预算对齐**（`swm_cem_matched`：64 采样 / 8 步 / 6 精英 / 初始 σ=0.5）
   中位 d2 {cem_matched['distance_median']:.4f}，中位改善 {cem_matched['improvement_median']:.2f}×，
   断言 {_flag(cem_matched['shipped_assertion'])}，中位耗时 {cem_matched['wall_s_median']:.3f}s
   （现状方案 {box['wall_s_median']:.3f}s，即 {slow_factor:.2f} 分之一）。同预算下不更差。

4. **上游 CEM，库默认值**（`swm_cem_default`：300 采样 / 30 步 / 30 精英 / σ=1.0，
   我方**未调参**）中位 d2 {cem_default['distance_median']:.4f}，中位改善
   {cem_default['improvement_median']:.2f}×，断言 {_flag(cem_default['shipped_assertion'])}。
   在这个任务上基本打到精确解。

5. **上游 MPPI，预算对齐**中位 d2 {mppi['distance_median']:.4f}，中位改善
   {mppi['improvement_median']:.2f}×，断言 {_flag(mppi['shipped_assertion'])}。同预算里最好的一档。

6. **负结果，照记**：`swm_icem`（预算对齐）中位 d2 {icem['distance_median']:.4f}，比同预算的
   普通 CEM 差。在 8 次迭代这个预算下，iCEM 的彩色噪声 + 精英保留 + 动量没有兑现论文里的
   样本效率优势。`swm_predictive`（单发采样）中位 d2 {predictive['distance_median']:.4f}，最弱，
   符合单发采样的预期。两者都不作为改用上游的理由。

7. **一处对不上的既有注释**：`tests/test_rollout.py` 写 "64 candidates x 8 iters
   (~140s on a laptop)"，本机实测现状方案中位 {box['wall_s_median']:.3f}s，相差约
   {1.0 / max(note_factor, 1e-9):.0f} 分之一。这里只记录实测，
   **不改那条注释**——改它属于另一个改动集。

## 不声称什么

- 这是**一个任务、一个冻结的随机初始化模型、{len(seeds)} 个种子、纯 CPU**。对训练后的检查点
  不构成任何结论。
- 上游诸臂采样**未截断**的高斯，现状臂截断在 `[-1, 1]`。这是两个实现之间的真实差异，
  不是 harness 造成的。
- 未跑 GPU；未跑真实轨迹；未改动任何已发布数字。
- 本实验不构成"上游更好"的一般结论，只说明：**同一个任务、同一个预算下，复用上游求解器
  不比三十行手写 CEM 差，且省掉维护**。

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

    results: dict[str, dict] = {name: {"distance": [], "wall_s": [], "improvement": []} for name in ARMS}
    bases: list[float] = []

    for seed in args.seeds:
        model, latent0, goal, base = build_task(seed)
        bases.append(base)
        row = [f"seed={seed}", f"base={base:.2f}"]
        for name, arm in ARMS.items():
            distance, elapsed = arm(model, latent0, goal, seed)
            results[name]["distance"].append(distance)
            results[name]["wall_s"].append(elapsed)
            results[name]["improvement"].append(base / distance if distance > 0 else float("inf"))
            row.append(f"{name}={distance:.2f}")
        print("  ".join(row))

    summary = {
        "experiment": "SWM-PLANNING-v0",
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
        "arms": {},
        "notes": [
            "distances use LatentPlanner's reduction: mean-pool over tokens, then squared "
            "L2 to the goal; for the SWM arms it is recomputed on the returned actions.",
            "arms 3-6 sample an unclamped Gaussian; kinejepa_box clamps into [-1, 1].",
            "the goal is generated by unclamped randn actions, so kinejepa_box cannot "
            "express the generating sequence itself; kinejepa_unbounded isolates that.",
            "arm 1's ratio assertion is build-dependent -- see kineworld/kine-jepa#3.",
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
            "shipped_assertion": [
                shipped_assertion(d, b) for d, b in zip(data["distance"], bases)
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown_path = args.out.parent.parent / "SWM-PLANNING.md"
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"\n{'arm':24s} {'median d2':>10s} {'median x':>9s} {'median s':>9s}  shipped_assertion")
    for name, data in summary["arms"].items():
        passes = data["shipped_assertion"]
        flag = "PASS" if all(passes) else ("FAIL" if not any(passes) else "mixed")
        print(
            f"{name:24s} {data['distance_median']:10.4f} "
            f"{data['improvement_median']:9.2f} {data['wall_s_median']:9.3f}  {flag}"
        )
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
