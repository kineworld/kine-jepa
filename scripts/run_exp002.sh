#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:?usage: run_exp002.sh path/to/exp001-ckpt.pt}
STEPS=${STEPS:-2000}
BS=${BS:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
python tests/test_causal.py
python tests/test_pairs.py
python scripts/grant_loop_smoke.py
for ARM in A B C; do
  python -m kineworld_jepa.train_exp002 --resume "$CKPT" --arm "$ARM" --steps "$STEPS" --batch-size "$BS" --img-size 64 --num-frames 16
  LAST=$(ls -dt experiments/KINE-EXP-002/${ARM}-* | head -1)
  python scripts/extract_base_ckpt.py "$LAST/ckpt-final.pt" "$LAST/ckpt-base.pt"
  echo "ARM $ARM -> $LAST"
done
echo "next: python -m kinebench run --ckpt experiments/KINE-EXP-002/<arm>/ckpt-base.pt --out results/EXP-002-<arm>.json"
