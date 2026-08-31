#!/usr/bin/env python3
"""Strip CausalKineJEPA 'base.' prefix so kine-bench load_model can ingest EXP-002 ckpts."""
from __future__ import annotations
import argparse, torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    if any(k.startswith("base.") for k in state):
        state = {k[5:]: v for k, v in state.items() if k.startswith("base.")}
    out = {"model": state, "config": ckpt.get("config", {}), "arm": ckpt.get("arm"), "step": ckpt.get("step")}
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"wrote {args.dst} keys={len(state)}")

if __name__ == "__main__":
    main()
