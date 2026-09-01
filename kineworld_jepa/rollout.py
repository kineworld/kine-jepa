# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT) and the
# action-conditioned world-model literature; all code original.
"""Action-conditioned latent rollout + goal-conditioned planning.

Why this module exists
-----------------------
KINE-EXP-001 / KineOne-WM-Latent is an *encoder + masked-token predictor*: it
learns a physical latent space but cannot *act* in it. That is a world MODEL
in name only -- it cannot propose what happens next given an action. This
module closes that gap, mirroring Meta's V-JEPA 2-AC recipe:

    freeze the pretrained encoder, then post-train a small predictor that,
    given the current latent and an action chunk, imagines the next latent.

Because the encoder is frozen and the predictor is tiny (and post-trained from
a small amount of real robot / simulator trajectories), this runs on the same
single 12GB laptop the rest of KINEWORLD uses.

What is open vs. what is the moat
---------------------------------
- This *interface and architecture* (ActionRollout, LatentPlanner) are open:
  they are the scaffolding anyone needs, and openness makes the KINE-Bench
  planner track reproducible.
- The *trained weights* for a specific embodiment, the *action-labeling* of
  your trajectory data, and the *post-training recipe* (curriculum, data mix,
  distillation from a ViT-g teacher) are the company moat and live in a
  private repo -- never pushed to this public one.

Run on CPU for the smoke test; the planner is gradient-free (CEM), so it works
without a GPU on the laptop for short horizons.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEmbedder(nn.Module):
    """Discretized / continuous action -> latent-space conditioning vector.

    `dim` should match the token dimension of the frozen encoder so the action
    can be added to (or cross-attended over) the latent tokens, exactly like
    KineOne-WM's InterventionHead adds a do(x) token.
    """

    def __init__(self, action_dim: int, dim: int, hidden: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.trunc_normal_(self.proj[-1].weight, std=0.02)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # action: (B, action_dim) -> (B, 1, dim) broadcastable over tokens
        return self.proj(action).unsqueeze(1)


class ActionRollout(nn.Module):
    """Predicts the next latent given the current latent and an action.

    Two conditioning styles are supported:
      - "add":    action vector is added to every visible token (cheap, like
                 the intervention head)
      - "cross":  action is a cross-attention token in each transformer block
                 (more expressive, more params)

    Input latents are the *visible* encoder tokens (B, V, D); the predictor
    rolls them forward in latent space for `horizon` steps.
    """

    def __init__(self, dim: int, depth: int = 6, heads: int = 12,
                 action_dim: int = 8, style: str = "add"):
        super().__init__()
        self.style = style
        self.action_embed = ActionEmbedder(action_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, 1, dim))
        blocks = [nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                              dim_feedforward=dim * 4,
                                              batch_first=True) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        if style == "cross":
            self.act_cross = nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True)
                for _ in range(depth)
            ])

    def step(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.action_embed(action)  # (B, 1, dim)
        x = latent + self.pos
        if self.style == "add":
            # inject the action as a broadcast bias over every token (cheap,
            # same shape as KineOne-WM's InterventionHead do(x) token)
            x = x + a
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.style == "cross":
                x = x + self.act_cross[i](x, a, a)[0]
        delta = self.norm(x)
        # residual update: predict the *change* in latent given the action.
        # This is what lets a planner move the trajectory toward a goal and
        # what keeps a frozen-initialization rollout stable (delta ~ 0 at init).
        return latent + delta

    def forward(self, latent0: torch.Tensor, actions: torch.Tensor,
                horizon: int | None = None) -> list:
        """Roll out `horizon` steps.

        latent0: (B, V, D)  initial visible latent
        actions: (B, horizon, action_dim) or (B, action_dim) applied each step
        returns: list[(B, V, D)] of length `horizon` (future latents)
        """
        if horizon is None:
            horizon = actions.shape[1]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).repeat(1, horizon, 1)
        out = []
        cur = latent0
        for t in range(horizon):
            cur = self.step(cur, actions[:, t])
            out.append(cur)
        return out


class LatentPlanner:
    """Goal-conditioned planning in latent space via Cross-Entropy Method.

    Given an initial visible latent and a *goal latent* (e.g. the encoder
    features of a target image), search for the action sequence whose rolled-
    out latent lands closest to the goal. This is how a frozen world model
    drives a robot: imagine, then pick the future you want.
    """

    def __init__(self, rollout: ActionRollout, goal_latent: torch.Tensor,
                 action_dim: int, horizon: int = 8, action_low: float = -1.0,
                 action_high: float = 1.0):
        self.rollout = rollout
        self.goal = goal_latent.detach()
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def _distance(self, latent: torch.Tensor) -> torch.Tensor:
        # mean over tokens, then L2 to the goal (both already in encoder space)
        z = latent.mean(dim=1)
        g = self.goal.mean(dim=1)
        return (z - g).pow(2).sum(dim=-1)  # (B,)

    def plan(self, latent0: torch.Tensor, iters: int = 12, candidates: int = 256,
              elite_frac: float = 0.1, lr: float = 0.6, device: str = "cpu", seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        best = None
        best_loss = None
        mean = torch.zeros(candidates, self.horizon, self.action_dim, device=device)
        # std is *per-distribution* (not per-elite): elite variance sets a step
        # size, not a frozen floor -- otherwise the search freezes immediately.
        std = torch.ones_like(mean) * 0.5
        for it in range(iters):
            acts = torch.clamp(
                mean + std * torch.randn(candidates, self.horizon, self.action_dim,
                                         generator=g, device=device),
                self.action_low, self.action_high,
            )
            futures = self.rollout(latent0.repeat(candidates, 1, 1), acts, self.horizon)
            loss = self._distance(futures[-1])
            k = max(1, int(elite_frac * candidates))
            idx = torch.argsort(loss)[:k]
            elite = acts[idx]
            mean = elite.mean(dim=0)                      # move toward elite mean
            std = (std * (1 - lr) + elite.std(dim=0) * lr).clamp_(
                1e-2, 1.0)                                 # shrink but never freeze
            cur = loss[idx[0]].item()
            if best_loss is None or cur < best_loss:
                best_loss = cur
                best = elite[0]
        return best.detach(), best_loss


def rollout_demo(latent0: torch.Tensor, actions: torch.Tensor, model: ActionRollout):
    return model(latent0, actions)
