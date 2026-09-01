# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT) and
# KINE-EXP-002 causal.py (InterventionHead); all code original.
"""End-to-end counterfactual latent rollout.

Wires `causal.InterventionHead` (discrete do(x) scene-conditioner) into the
action-conditioned world model (`rollout.ActionRollout` + `MultiActionEmbedder`)
so the model answers *what-if* questions in latent space:

    "Given the same initial scene and the same continuous arm commands, what
     would the future look like if I had (not) removed the support?"

This is the capability that turns KineOne-WM from a *forecaster* into a *world
model*: it can imagine alternatives, not just predict the default future. It is
the core differentiator vs. a pure encoder/benchmark like Baize.

The factorization (no double counting)
--------------------------------------
- discrete do(x)  -> InterventionHead, applied to the *current latent* every
  step. It is a *scene conditioner* -- it changes the state the world evolves
  from -- mirroring CausalKineJEPA's usage. This is the counterfactual lever.
- continuous arm  -> MultiActionEmbedder, added as an action token (the agent's
  motor command the world model must respect).

Both live in the same `dim` space as the frozen encoder, so a single world
model can score (KINE-Bench), plan (LatentPlanner), and counterfactually
imagine in one space.

Open vs moat
------------
This interface and recipe are open. The trained weights for a specific
embodiment, the trajectory action-labels, and the post-training recipe
(curriculum / data mix / ViT-g teacher distillation) are the moat and live in a
private repo -- never pushed to this public one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rollout import ActionRollout, MultiActionEmbedder
from .causal import InterventionHead, NUM_INTERVENTIONS


class CounterfactualRollout(ActionRollout):
    """Action-conditioned world model where a discrete intervention is a
    *counterfactual lever* applied to the latent every step.

    Inputs to `forward`:
      latent0    : (B, V, D)        initial visible latent
      arm_actions: (B, H, action_dim)  continuous motor command per step
      do_ids     : (B, H)           discrete intervention id per step (0..3)
    Returns a list[(B, V, D)] of H future latents.

    The discrete `do(x)` is handled by the causal InterventionHead (a scene
    conditioner), the continuous arm command by the MultiActionEmbedder (an
    action token) -- clean separation, no duplicated conditioning.
    """

    def __init__(self, dim: int, action_dim: int, depth: int = 6, heads: int = 12,
                 latent_clip: float | None = None):
        super().__init__(dim, depth=depth, heads=heads, action_dim=action_dim,
                         style="add",
                         action_embed=MultiActionEmbedder(
                             {"arm": ("continuous", action_dim)}, dim),
                         latent_clip=latent_clip)
        # discrete do(x) scene-conditioner, reused from KINE-EXP-002
        self.intervention_head = InterventionHead(dim, NUM_INTERVENTIONS)

    def step(self, latent: torch.Tensor, action: dict) -> torch.Tensor:
        # action = {"arm": (B, action_dim), "do": (B,) or (B, 1)}
        do_id = action["do"]
        if do_id.dim() == 2:
            do_id = do_id.squeeze(-1)
        # 1) counterfactual lever: re-condition the current latent on do(x)
        z = self.intervention_head(latent, do_id.long())
        # 2) arm-conditioned rollout on the do-conditioned latent
        return super().step(z, {"arm": action["arm"]})

    def forward(self, latent0: torch.Tensor, arm_actions: torch.Tensor,
                do_ids: torch.Tensor, horizon: int | None = None) -> list:
        H = horizon or arm_actions.shape[1]
        out, cur = [], latent0
        for t in range(H):
            cur = self.step(cur, {"arm": arm_actions[:, t], "do": do_ids[:, t]})
            out.append(cur)
        return out

    @torch.no_grad()
    def counterfactual(self, latent0: torch.Tensor, arm_actions: torch.Tensor,
                       base_id: int, alt_id: int) -> tuple:
        """Roll out the *same* continuous arm actions twice, swapping only the
        discrete intervention id (base_id vs alt_id) at every step.

        Returns (base_futures, alt_futures, divergence) where `divergence` is the
        L2 gap between the two final latents (mean over batch & tokens) -- the
        magnitude of the counterfactual effect. Two identical ids give 0.
        """
        B, H = arm_actions.shape[0], arm_actions.shape[1]
        base_ids = torch.full((B, H), base_id, dtype=torch.long, device=arm_actions.device)
        alt_ids = torch.full_like(base_ids, alt_id)
        base = self(latent0, arm_actions, base_ids)
        alt = self(latent0, arm_actions, alt_ids)
        div = (base[-1] - alt[-1]).pow(2).mean().sqrt().item()
        return base, alt, div
