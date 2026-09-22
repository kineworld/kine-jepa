import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kineworld_jepa.grant import issue, verify, decide

class TestSuite(unittest.TestCase):
    def test_missing_ticket_denies(self):
        assert verify(None, {"action": "close_gripper"}) == "DENY"

    def test_bound_ticket_allows(self):
        t = issue("a", "sim.arm.gripper", "close_gripper", "pick")
        assert verify(t, {"agent": "a", "target": "sim.arm.gripper", "action": "close_gripper", "purpose": "pick"}) == "ALLOW"

    def test_wrong_action_denies(self):
        t = issue("a", "sim.arm.gripper", "close_gripper", "pick")
        assert verify(t, {"agent": "a", "target": "sim.arm.gripper", "action": "open_gripper", "purpose": "pick"}) == "DENY"

    def test_risk_overrides_ticket(self):
        t = issue("kineone-wm-sim-0", "sim.arm.gripper", "close_gripper", "pick")
        assert decide({"action": "close_gripper", "purpose": "pick", "pred_risk": 0.9}, t) == "DENY"

if __name__ == "__main__":
    unittest.main()