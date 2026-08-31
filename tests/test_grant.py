import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kineworld_jepa.grant import issue, verify, decide

def test_missing_ticket_denies():
    assert verify(None, {"action": "close_gripper"}) == "DENY"

def test_bound_ticket_allows():
    t = issue("a", "sim.arm.gripper", "close_gripper", "pick")
    assert verify(t, {"agent": "a", "target": "sim.arm.gripper", "action": "close_gripper", "purpose": "pick"}) == "ALLOW"

def test_wrong_action_denies():
    t = issue("a", "sim.arm.gripper", "close_gripper", "pick")
    assert verify(t, {"agent": "a", "target": "sim.arm.gripper", "action": "open_gripper", "purpose": "pick"}) == "DENY"

def test_risk_overrides_ticket():
    t = issue("kineone-wm-sim-0", "sim.arm.gripper", "close_gripper", "pick")
    assert decide({"action": "close_gripper", "purpose": "pick", "pred_risk": 0.9}, t) == "DENY"

if __name__ == "__main__":
    for fn in (test_missing_ticket_denies, test_bound_ticket_allows, test_wrong_action_denies, test_risk_overrides_ticket):
        fn(); print("PASS", fn.__name__)
