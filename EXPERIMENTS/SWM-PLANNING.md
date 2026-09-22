# SWM-PLANNING-v0 · 用上游求解器替换手写 CEM

状态：已完成（第二版，已更正第一版的结论）。结论只在本文件与 `summary.json` 中，别处不引用。
本文件与 `summary.json` **都由脚本生成**，数字无手打：

    python scripts/swm_planning_comparison.py

机器可读版本：[`SWM-PLANNING-v0/summary.json`](SWM-PLANNING-v0/summary.json)。

环境：torch 2.11.0+cu128 / cpu；上游 stable-worldmodel 0.1.1（MIT，Fork 在 `kineworld/stable-worldmodel`）。
任务：dim=32，tokens=24，action_dim=4，horizon=8，与 `tests/test_rollout.py::test_planner_reaches_goal` 同一构造。
种子：2, 3, 4；对应 off-target 基线 25.88, 12.20, 47.66。
证据级别：**E1** —— one task, one frozen random-initialised model, 3 seeds, CPU only; no trained checkpoint was involved。

## 为什么做这个

`kineworld_jepa/rollout.py` 里的 `LatentPlanner` 是三十行手写 CEM。上游
`stable-worldmodel`（MIT，LeWM 的作者团队）带八个维护中的求解器，而它对世界模型的要求只有
一个方法：

    get_cost(info_dict, action_candidates) -> (B, S)

`kineworld_jepa/interop/swm.py` 实现这一个方法，就换来整套搜索代码。

## 这一版改了什么（第一版错了什么）

第一版把「手写（截断到 ±1）」和「上游（不截断）」放在一起比，**同时变了两个变量**：搜索规则
和动作空间。差异被算到了搜索规则头上。这一版把两轴交叉开，并把「上游到底谁裁剪动作」改成
**实测字段** `upstream_native_clamp`，而不是读源码推断。实测结果：`cem` 否（实测峰值 5.376）, `mppi` 否（实测峰值 2.550）, `icem` 是（实测峰值 0.999）, `predictive_sampling` 否（实测峰值 1.686）。

⇒ 1 个求解器会自己裁剪，其余只记录动作空间、从不执行。
第一版因此把 `swm_icem` 记成负结果——它是**唯一**被框住的上游臂，其余上游臂全是自由的。

这不是推断，是复现：把 iCEM 放到 `[-1, 1]`、关掉投影，逐种子距离为 8.761207, 0.154088, 7.216594；
第一版 `summary.json`（commit `a14b262`）记的 `swm_icem` 为 8.761207, 0.154088, 7.216594。
逐位相同 = **真**。
对照组（配置未变、数字也应未变）`swm_cem_free` 对第一版 `swm_cem_matched`：
**一致**。没有这个对照组，上面那条相等也可能只是
harness 忽略输入造成的。

即：第一版那条 iCEM 负结果，跑的一直是 `swm_icem_boxed`。
放开动作框后 iCEM 中位 0.2008，是上游最好的一档。

## 结果 · 自由动作空间（±1e9）

| 方案 | 中位 d2 | 中位改善 | 中位耗时 s | 返回动作 max|a| | 既有断言 |
| --- | ---: | ---: | ---: | ---: | --- |
| `kinejepa_unbounded` | 0.2121 | 87.61× | 1.027 | 2.333 | PASS |
| `swm_cem_free` | 0.9572 | 27.04× | 0.741 | 1.895 | PASS |
| `swm_mppi_free` | 0.5268 | 57.88× | 0.700 | 1.840 | PASS |
| `swm_icem_free` | 0.2008 | 118.60× | 0.726 | 3.716 | PASS |
| `swm_predictive_free` | 6.5560 | 3.95× | 0.090 | 2.428 | mixed |
| `swm_cem_default` | 0.0061 | 1990.28× | 5.495 | 2.999 | PASS |

## 结果 · 动作框 ±1（`LatentPlanner` 出厂设置）

| 方案 | 中位 d2 | 中位改善 | 中位耗时 s | 返回动作 max|a| | 既有断言 |
| --- | ---: | ---: | ---: | ---: | --- |
| `kinejepa_box` | 8.2277 | 4.88× | 1.057 | 1.000 | mixed |
| `swm_cem_boxed` | 8.4617 | 4.24× | 0.725 | 1.000 | mixed |
| `swm_mppi_boxed` | 6.4978 | 7.33× | 0.701 | 1.000 | PASS |
| `swm_icem_boxed` | 7.2166 | 6.60× | 0.696 | 0.997 | mixed |
| `swm_predictive_boxed` | 11.7091 | 2.21× | 0.090 | 1.000 | mixed |

## 读法

1. **主导变量是动作框，不是求解器。** 同一个求解器放进 `[-1, 1]` 就停在
   8.23 一档，放开到自由就进入 0.21 一档。
   目标 latent 本身由**未截断**的 `randn` 动作生成，`[-1, 1]` 的框里未必存在解。
   既有的 `test_planner_reaches_goal` 之所以时红时绿，主因在这里，不在 CEM。
   这是**诊断**：真实本体上框是必要的，`kinejepa_unbounded` 不是提案。

2. **同预算同框：不能一概而论，要按求解器分。** 框住之后手写 8.2277、
   上游 CEM 8.4617（1.03 倍以内，同档，手写略优）；
   但 MPPI 6.4978 **明显更好**，而且是框内唯一通过既有断言的臂；
   iCEM 7.2166；Predictive
   11.7091 最弱。
   ⇒ 说"上游不比手写好"和说"上游更好"都过度概括，取决于换成哪一个。

3. **自由空间、同预算：手写 0.2121，
   iCEM 0.2008（上游最好，与手写同档），
   MPPI 0.5268，CEM 0.9572。**
   手写在这个任务上并不落后；真正落后的是上游 CEM 的默认精英更新。

4. **上游要拉开差距，靠的是预算，不是算法。** `swm_cem_default`
   （300 采样 × 30 步 = 手写的 22 倍采样量）中位 0.0061，
   基本打到精确解。这是**算力差异**，不是"上游更聪明"。

5. **采纳的真正理由**是：八个维护中的求解器、可替换、不必自己维护搜索代码；以及本实验暴露的
   **反向风险**——直接换成上游会**丢掉执行器动作框**：只有 `icem` 会自己裁剪，
   `cem`, `mppi`, `predictive_sampling` 只记录动作空间而从不执行。
   `SWMPlanner(enforce_action_bounds=True)` 把框补回来，这也是 `swm_*_boxed` 各臂的由来。

6. **一处对不上的既有注释**：`tests/test_rollout.py` 写 "64 candidates x 8 iters
   (~140s on a laptop)"，本机实测手写方案中位 1.057s。这里只记录实测，
   **不改那条注释**——改它属于另一个改动集。

## 不声称什么

- 这是**一个任务、一个冻结的随机初始化模型、3 个种子、纯 CPU**。对训练后的检查点
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
