# Grant Loop v0

observe → rollout×3 → score → request_grant → ALLOW|DENY → act? → receipt

缺票必须 DENY。pred_risk > 0.4 必须 DENY。
本周 HMAC，不可作为真机唯一安全层。

跑：`python scripts/grant_loop_smoke.py`
