# Grant Loop v0

observe → rollout×3 → score → request_grant → ALLOW|DENY → act? → receipt

缺票必须 DENY。pred_risk > 0.4 必须 DENY。
本周 HMAC，不可作为真机唯一安全层。

跑：`python scripts/grant_loop_smoke.py`

输出：`EXPERIMENTS/GRANT-LOOP-v0/`（`receipts.jsonl` + `summary.json`，均由脚本生成，
大写 `EXPERIMENTS/`，与本文档同目录）。是否把这两个产物提交进仓库见
`kineworld/kine-jepa#7`。

## 这个 smoke 证明什么，不证明什么

证明：缺票必 DENY；`pred_risk > 0.4` 必 DENY；20 轮 × 2 条的 receipt 形状稳定。

**不证明**签名或意图绑定有效。本脚本是**自包含的平行实现**：它自己的 `decide()` 只看
「有没有票 / `pred_risk` / 是否过期」，`ticket["sig"]` 算出来之后**既不写进 receipt、
也不做任何校验**，全程不调用 `kineworld_jepa/grant.py`。

真正的校验（`hmac.compare_digest` 验签 + `agent`/`target`/`action`/`purpose` 逐项绑定）
在 `kineworld_jepa/grant.py::verify`，`tests/test_grant.py` 覆盖的是那一份。

⇒ **跑通这个 smoke 不等于授权层安全**；它是流程形状的检查，不是安全证明。
