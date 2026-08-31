"""KineGrant v0: capability tickets. Simulation only. Not a safety control."""
from __future__ import annotations
import hashlib, hmac, json, secrets, time

SECRET = b"kineworld-dev-hmac-not-for-hardware"
RISK_DENY = 0.4


def sign(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(SECRET, raw, hashlib.sha256).hexdigest()


def issue(agent, target, action, purpose, horizon_s=30) -> dict:
    ticket = {
        "id": "cap_" + secrets.token_hex(4),
        "agent": agent,
        "target": target,
        "action": action,
        "purpose": purpose,
        "horizon_s": horizon_s,
        "exp": time.time() + horizon_s,
        "nonce": secrets.token_hex(8),
    }
    ticket["sig"] = sign({k: ticket[k] for k in ticket if k != "sig"})
    return ticket


def verify(ticket, intent) -> str:
    if ticket is None:
        return "DENY"
    if time.time() > ticket.get("exp", 0):
        return "DENY"
    expected = sign({k: ticket[k] for k in ticket if k != "sig"})
    if not hmac.compare_digest(expected, ticket.get("sig", "")):
        return "DENY"
    for key in ("agent", "target", "action", "purpose"):
        if intent.get(key) != ticket.get(key) and key in ticket:
            return "DENY"
    return "ALLOW"


def decide(best, ticket) -> str:
    if best.get("pred_risk", 0) > RISK_DENY:
        return "DENY"
    intent = {
        "agent": "kineone-wm-sim-0",
        "target": "sim.arm.gripper",
        "action": best["action"],
        "purpose": best["purpose"],
    }
    return verify(ticket, intent)
