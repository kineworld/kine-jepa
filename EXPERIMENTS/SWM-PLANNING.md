# SWM-PLANNING-v0 · 用上游求解器替换手写 CEM

状态：已完成。结论只在本文件与 `summary.json` 中，别处不引用。
本文件与 `summary.json` **都由脚本生成**，数字无手打：

    python scripts/swm_planning_comparison.py

机器可读版本：[`SWM-PLANNING-v0/summary.json`](SWM-PLANNING-v0/summary.json)。

环境：torch 2.11.0+cu128 / cpu；上游 stable-worldmodel 0.1.1（MIT，Fork 在 `kineworld/stable-worldmodel`）。
任务：dim=32，tokens=24，action_dim=4，horizon=8，与 `tests/test_rollout.py::test_planner_reaches_goal` 同一构造。
种子：2, 3, 4；对应 off-target 基线 25.88, 12.20, 47.66。
证据级别：one task, one frozen random model, 3 seeds, CPU。

## 为什么做这个

`kineworld_jepa/rollout.py` 里的 `LatentPlanner` 是三十行手写 CEM。上游
`stable-worldmodel`（MIT，LeWM 的作者团队）带八个维护中的求解器，而它对世界模型的要求只有
一个方法：

    get_cost(info_dict, action_candidates) -> (B, S)

`kineworld_jepa/interop/swm.py` 实现这一个方法，就换来整套搜索代码。本实验回答两件事：
换过去至少不更差吗；以及既有的 `test_planner_reaches_goal` 到底为什么红。

## 结果

| 方案 | 中位 d2 | 中位改善 | 中位耗时 s | 仓库既有断言 |
| --- | ---: | ---: | ---: | --- |
| `kinejepa_box` | 8.2277 | 4.88× | 0.878 | mixed |
| `kinejepa_unbounded` | 0.2121 | 87.61× | 0.852 | PASS |
| `swm_cem_matched` | 0.9572 | 27.04× | 0.541 | PASS |
| `swm_cem_default` | 0.0061 | 1990.28× | 3.993 | PASS |
| `swm_icem` | 7.2166 | 6.60× | 0.479 | mixed |
| `swm_mppi` | 0.5268 | 57.88× | 0.476 | PASS |
| `swm_predictive` | 6.5560 | 3.95× | 0.061 | mixed |

## 读法

1. **现状方案**（`kinejepa_box`：截断到 ±1，8 次迭代 × 64 候选）中位 d2 8.2277，
   中位改善 4.88×，既有断言为 mixed。
   断言在部分种子为红，与 `kineworld/kine-jepa#3` 已记录的"该比值依赖 torch 构建"一致。

2. **同预算、同算法，只把动作框放开**（`kinejepa_unbounded`，±1e9）中位 d2
   0.2121，中位改善 87.61×，
   断言 PASS。目标 latent 本身就是由**未截断**的 `randn`
   动作生成的，所以 `[-1, 1]` 的框里未必存在能落到目标邻域的解。既有失败的主因是动作框，
   不是 CEM 本身。这个臂是诊断，不是提案：`LatentPlanner` 的框在真实本体上是必要的。

3. **上游 CEM，预算对齐**（`swm_cem_matched`：64 采样 / 8 步 / 6 精英 / 初始 σ=0.5）
   中位 d2 0.9572，中位改善 27.04×，
   断言 PASS，中位耗时 0.541s
   （现状方案 0.878s，即 1.62 分之一）。同预算下不更差。

4. **上游 CEM，库默认值**（`swm_cem_default`：300 采样 / 30 步 / 30 精英 / σ=1.0，
   我方**未调参**）中位 d2 0.0061，中位改善
   1990.28×，断言 PASS。
   在这个任务上基本打到精确解。

5. **上游 MPPI，预算对齐**中位 d2 0.5268，中位改善
   57.88×，断言 PASS。同预算里最好的一档。

6. **负结果，照记**：`swm_icem`（预算对齐）中位 d2 7.2166，比同预算的
   普通 CEM 差。在 8 次迭代这个预算下，iCEM 的彩色噪声 + 精英保留 + 动量没有兑现论文里的
   样本效率优势。`swm_predictive`（单发采样）中位 d2 6.5560，最弱，
   符合单发采样的预期。两者都不作为改用上游的理由。

7. **一处对不上的既有注释**：`tests/test_rollout.py` 写 "64 candidates x 8 iters
   (~140s on a laptop)"，本机实测现状方案中位 0.878s，相差约
   159 分之一。这里只记录实测，
   **不改那条注释**——改它属于另一个改动集。

## 不声称什么

- 这是**一个任务、一个冻结的随机初始化模型、3 个种子、纯 CPU**。对训练后的检查点
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
