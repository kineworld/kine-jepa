# One-machine runbook (5070 Ti)

## Today

```
python tests/test_core.py
python tests/test_causal.py
python scripts/grant_loop_smoke.py
```

## EXP-002 on the laptop GPU (after EXP-001 10k ckpt exists)

```
python -m kineworld_jepa.train_exp002 --resume <ckpt-step10000.pt> --arm A --steps 2000 --batch-size 8 --seed 42
python -m kineworld_jepa.train_exp002 --resume <ckpt-step10000.pt> --arm B --steps 2000 --batch-size 8 --seed 42
python -m kineworld_jepa.train_exp002 --resume <ckpt-step10000.pt> --arm C --steps 2000 --batch-size 8 --seed 42
```

Then evaluate with kine-bench KINE-CAU-1 + existing EVT-1. Publish the table even if C fails.

Do not resume 25k-step loss-only training as the company main line.
