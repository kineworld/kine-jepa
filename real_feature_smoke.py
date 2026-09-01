#!/usr/bin/env python
# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT); all code original.
"""Real-feature end-to-end smoke: KineOne-WM's counterfactual engine running
DIRECTLY on Meta V-JEPA 2 encoder features (not synthetic latents).

This is the credibility bridge: it proves the world model ingests the same
SOTA features KINE-Bench scores, in the same 1024-d space, and can roll them
forward / counterfactually imagine on them. The predictor is untrained (random
init) so the *rollout values* are not physical yet -- that is the moat recipe
(post-training on real trajectories, private repo). The integration is real.

CPU-feasible by encoding only 8 frames (override facade.num_frames) so the
token grid is 4x16x16 = 1024, then projecting to 256 tokens for the rollout.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "kine-bench"))
from kinebench.adapters.vjepa2 import VJEPA2Adapter
from kineworld_jepa.counterfactual import CounterfactualRollout
from kineworld_jepa.rollout import VJEPA2Projector

LOCAL = "C:/Users/zoah/AppData/Local/Temp/vjepa2"
torch.manual_seed(0)

t0 = time.time()
adapter = VJEPA2Adapter(alias="vjepa2-vitl-256", local_dir=LOCAL, num_frames=8, batch_size=1)
fac = adapter.build(device="cpu")
fac.num_frames = 8  # encode only 8 frames -> 1024 tokens, CPU-feasible
print(f"[1] build {time.time()-t0:.1f}s  grid={fac.grid}  device=cpu", flush=True)

# synthetic clip: a bright square drifting across a dark frame, (1,3,8,256,256)
clip = torch.zeros(1, 3, 8, 256, 256)
for t in range(8):
    x = 70 + t * 14
    clip[0, :, t, x:x + 64, x:x + 64] = 1.0
clip = clip.clamp(0, 1)

t1 = time.time()
z = fac.encoder(clip)  # (1, N, 1024) -- REAL V-JEPA 2 features
print(f"[2] encode {time.time()-t1:.1f}s  shape={tuple(z.shape)}", flush=True)

proj = VJEPA2Projector(in_tokens=z.shape[1], out_tokens=256, dim=1024)
z0 = proj(z)  # (1, 256, 1024)
print(f"[3] project -> {tuple(z0.shape)}", flush=True)

cf = CounterfactualRollout(dim=1024, action_dim=8, depth=3, heads=8, latent_clip=20.0)
cf.eval()
arm = torch.randn(1, 5, 8) * 0.2
t2 = time.time()
base = cf(z0, arm, torch.zeros(1, 5, dtype=torch.long))
alt = cf(z0, arm, torch.ones(1, 5, dtype=torch.long))
div = (base[-1] - alt[-1]).pow(2).mean().sqrt().item()
print(f"[4] counterfactual on REAL V-JEPA 2 features: div(no-op vs remove_support)={div:.4f}  "
      f"rollout {time.time()-t2:.1f}s", flush=True)
print(f"[done] total {time.time()-t0:.1f}s", flush=True)
