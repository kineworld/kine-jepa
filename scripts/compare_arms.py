#!/usr/bin/env python3
"""Build the EXP-002 public table from three kinebench JSON files."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

NEED = ("KINE-EVT-1", "KINE-CAU-1", "KINE-FUT-1")

def score(task):
    if not isinstance(task, dict):
        return None
    for k in ("auc", "score", "cosine", "top1"):
        if k in task and task[k] is not None:
            return float(task[k])
    vals = [v for v in task.values() if isinstance(v, (int, float))]
    return float(vals[0]) if vals else None

def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    return {name: score(tasks.get(name, {})) for name in NEED}

def decide(rows):
    a, b, c = rows["A"], rows["B"], rows["C"]
    notes = []
    ok = True
    if c["KINE-EVT-1"] is None or a["KINE-EVT-1"] is None:
        return False, ["missing EVT"]
    if not (c["KINE-EVT-1"] >= 0.58 and c["KINE-EVT-1"] > a["KINE-EVT-1"]):
        ok = False; notes.append("EVT gate fail")
    if c["KINE-CAU-1"] is None or a["KINE-CAU-1"] is None:
        ok = False; notes.append("missing CAU")
    elif c["KINE-CAU-1"] - a["KINE-CAU-1"] < 0.08:
        ok = False; notes.append("CAU delta < 0.08")
    if c["KINE-FUT-1"] is not None and a["KINE-FUT-1"] is not None:
        if a["KINE-FUT-1"] - c["KINE-FUT-1"] > 0.03:
            ok = False; notes.append("FUT drop > 0.03")
    if b["KINE-CAU-1"] is not None and c["KINE-CAU-1"] is not None:
        if b["KINE-CAU-1"] >= c["KINE-CAU-1"]:
            ok = False; notes.append("B not worse than C")
    if ok:
        notes.append("PASS")
    return ok, notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--C", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = {arm: load(getattr(args, arm)) for arm in "ABC"}
    ok, notes = decide(rows)
    print("| arm | EVT-1 | CAU-1 | FUT-1 |")
    print("|---|---|---|---|")
    for arm in "ABC":
        r = rows[arm]
        print(f"| {arm} | {r['KINE-EVT-1']} | {r['KINE-CAU-1']} | {r['KINE-FUT-1']} |")
    print("verdict:", "PASS" if ok else "FAIL", ";".join(notes))
    payload = {"arms": rows, "pass": ok, "notes": notes}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
