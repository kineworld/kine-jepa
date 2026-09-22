import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kineworld_jepa.grant import issue, verify, decide

ROOT = Path(__file__).resolve().parent.parent

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

def test_smoke_writes_into_the_documented_experiment_directory():
    """The smoke script's output directory must be the one the repository tracks.

    It read `experiments/GRANT-LOOP-v0` -- lowercase -- while the record is
    `EXPERIMENTS/GRANT-LOOP.md` and `.gitignore` carries an `experiments/` scratch rule
    (negated for `EXPERIMENTS/`). On a case-insensitive filesystem the two spellings are
    one directory, so it looked correct; on Linux the script would create a *second*
    directory that the ignore rule does cover, and that run's outputs could never be
    published. Comparing the first path component case-sensitively is what fails on a
    regression, including on Windows, where `Path` comparison alone would not.
    """
    spec = importlib.util.spec_from_file_location(
        "grant_loop_smoke", ROOT / "scripts" / "grant_loop_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the __main__ guard keeps main() from running

    out = Path(module.OUT)
    assert out.parts[0] == "EXPERIMENTS", f"smoke output starts with {out.parts[0]!r}, expected 'EXPERIMENTS'"
    assert out.parent == Path("EXPERIMENTS"), f"smoke output parent is {out.parent}"
    assert (ROOT / "EXPERIMENTS" / "GRANT-LOOP.md").is_file(), "the record the output sits beside is missing"

if __name__ == "__main__":
    for fn in (test_missing_ticket_denies, test_bound_ticket_allows, test_wrong_action_denies,
               test_risk_overrides_ticket, test_smoke_writes_into_the_documented_experiment_directory):
        fn(); print("PASS", fn.__name__)
