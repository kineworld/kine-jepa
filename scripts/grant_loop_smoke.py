#!/usr/bin/env python3
"""Grant Loop v0 smoke. Simulation only. Not a safety control."""
from __future__ import annotations
import hashlib, hmac, json, secrets, time
from pathlib import Path
SECRET = b"kineworld-dev-hmac-not-for-hardware"
OUT = Path("experiments/GRANT-LOOP-v0")
RISK_DENY = 0.4
def sign(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(SECRET, raw, hashlib.sha256).hexdigest()
def score_rollouts(obs):
    drop_near = obs.get("drop_near", False)
    return [
        {"id": "r0", "action": "idle", "pred_risk": 0.05, "purpose": "hold", "task": False},
        {"id": "r1", "action": "close_gripper", "pred_risk": 0.62 if drop_near else 0.11, "purpose": "pick_block_red", "task": True},
        {"id": "r2", "action": "push", "pred_risk": 0.28, "purpose": "clear_path", "task": False},
    ]
def decide(best, ticket):
    if ticket is None: return "DENY"
    if best["pred_risk"] > RISK_DENY: return "DENY"
    if time.time() > ticket["exp"]: return "DENY"
    return "ALLOW"
def loop(obs, with_ticket):
    rolls = score_rollouts(obs)
    best = next(r for r in rolls if r["task"])
    intent = {"agent": "kineone-wm-sim-0", "target": "sim.arm.gripper", "action": best["action"], "purpose": best["purpose"], "horizon_s": 30, "rollout_id": best["id"], "pred_risk": best["pred_risk"]}
    ticket = None
    if with_ticket:
        ticket = {"id": "cap_" + secrets.token_hex(4), "binds": ["agent", "target", "action", "purpose", "horizon_s"], "exp": time.time() + 30, "nonce": secrets.token_hex(8)}
        ticket["sig"] = sign(ticket)
    decision = decide(best, ticket)
    return {"cap_id": None if ticket is None else ticket["id"], "decision": decision, "predicted": {"contact": True, "drop": best["pred_risk"] > RISK_DENY}, "observed": {"contact": decision == "ALLOW", "drop": False}, "mismatch": False, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "intent": intent}
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    deny_missing = deny_risk = allow = 0
    for i in range(20):
        obs = {"drop_near": i == 3}
        r = loop(obs, True)
        receipts.append(r)
        if r["decision"] == "ALLOW": allow += 1
        elif r["predicted"]["drop"]: deny_risk += 1
        r2 = loop(obs, False)
        receipts.append(r2)
        assert r2["decision"] == "DENY"
        deny_missing += 1
    summary = {"cycles": 20, "allow": allow, "deny_missing_ticket": deny_missing, "deny_risk": deny_risk, "ok_missing_ticket": deny_missing == 20, "ok_risk_at_least_one": deny_risk >= 1}
    (OUT / "receipts.jsonl").write_text("\n".join(json.dumps(x) for x in receipts) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["ok_missing_ticket"] or not summary["ok_risk_at_least_one"]:
        raise SystemExit(1)
if __name__ == "__main__":
    main()
