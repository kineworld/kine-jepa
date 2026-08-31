#!/usr/bin/env python3
"""Fake 1-DoF gripper. Moves only if Grant Loop returns ALLOW. Not safety-rated."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "GRANT-LOOP-v0"


def main():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "grant_loop_smoke.py")], cwd=ROOT)
    receipts = [json.loads(l) for l in (OUT / "receipts.jsonl").read_text().splitlines() if l.strip()]
    traj = []
    pos = 0.0
    for r in receipts:
        if r["decision"] == "ALLOW" and r["intent"]["action"] == "close_gripper":
            pos = min(1.0, pos + 0.2)
            moved = True
        else:
            moved = False
        traj.append({"t": r["ts"], "pos": round(pos, 3), "moved": moved, "decision": r["decision"], "cap_id": r["cap_id"]})
    (OUT / "trajectory.jsonl").write_text("\n".join(json.dumps(x) for x in traj) + "\n")
    moved_n = sum(1 for x in traj if x["moved"])
    denied_n = sum(1 for x in traj if x["decision"] == "DENY")
    summary = {"steps": len(traj), "moved": moved_n, "denied": denied_n, "final_pos": traj[-1]["pos"]}
    (OUT / "trajectory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if denied_n == 0:
        raise SystemExit("actuator moved without any DENY — loop broken")


if __name__ == "__main__":
    main()
