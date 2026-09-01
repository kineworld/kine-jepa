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

Extensions (this file)
----------------------
1. Multi-action space: a heterogeneous action (continuous arm command + discrete
   do(x) intervention) is embedded and mixed into one conditioning token, so a
   single world model can drive a real embodiment. See `MultiActionEmbedder`.
2. Long-horizon stability + post-training: recursive rollouts fix the latent
   norm (`latent_clip`) so they do not drift over many steps, and
   `training_loss` gives a teacher-forced regression target for post-training
   the predictor on real trajectories (the moat recipe).
3. V-JEPA 2 alignment: `VJEPA2AlignedRollout` operates *directly* in V-JEPA 2's
   1024-d semantic space (its (B, 8192, 1024) encoder output is projected to a
   manageable token grid, never re-embedded), so the model both scores SOTA
   features (via KINE-Bench) and acts on them (plans) in the same space.

What is open vs. what is the moat
---------------------------------
- This interface and architecture (ActionRollout, LatentPlanner,
  MultiActionEmbedder, VJEPA2AlignedRollout) are open: they are the scaffolding
  anyone needs, and openness makes the KINE-Bench planner track reproducible.
- The trained weights for a specific embodiment, the action-labeling of your
  trajectory data, and the post-training recipe (curriculum, data mix,
  distillation from a ViT-g teacher) are the company moat and live in a private
  repo -- never pushed to this public one.

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

    The action embedder is injectable (`action_embed`): pass a
    `MultiActionEmbedder` to condition on a heterogeneous action space.
    """

    def __init__(self, dim: int, depth: int = 6, heads: int = 12,
                 action_dim: int = 8, style: str = "add",
                 action_embed: nn.Module | None = None,
                 latent_clip: float | None = None):
        super().__init__()
        self.style = style
        self.action_dim = action_dim
        self.latent_clip = latent_clip
        # Inject a custom embedder (e.g. MultiActionEmbedder) for heterogeneous
        # action spaces; default to a single continuous stream.
        self.action_embed = action_embed or ActionEmbedder(action_dim, dim)
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
        a = self.action_embed(action)  # (B, 1, dim)  (action may be a dict)
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
        nxt = latent + delta
        # Long-horizon stabilisation: fix the latent norm so recursive rollouts
        # do not drift / explode over many steps. Keeps direction, clips
        # magnitude. Disabled when latent_clip is None.
        if self.latent_clip is not None:
            n = nxt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            nxt = nxt / n * self.latent_clip
        return nxt

    def forward(self, latent0: torch.Tensor, actions: torch.Tensor,
                horizon: int | None = None) -> list:
        """Roll out `horizon` steps.

        latent0: (B, V, D)  initial visible latent
        actions: (B, horizon, action_dim) tensor, OR a dict of named streams
                each (B, horizon, *) for a heterogeneous action space
        returns: list[(B, V, D)] of length `horizon` (future latents)
        """
        if isinstance(actions, dict):
            first = next(iter(actions))
            H = actions[first].shape[1]
            horizon = horizon or H
            out, cur = [], latent0
            for t in range(horizon):
                cur = self.step(cur, {k: v[:, t] for k, v in actions.items()})
                out.append(cur)
            return out
        if horizon is None:
            horizon = actions.shape[1]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).repeat(1, horizon, 1)
        out, cur = [], latent0
        for t in range(horizon):
            cur = self.step(cur, actions[:, t])
            out.append(cur)
        return out

    @torch.no_grad()
    def training_loss(self, latent0: torch.Tensor, actions: torch.Tensor,
                      target_latents: list) -> torch.Tensor:
        """Teacher-forced next-latent regression for *post-training* the predictor
        on real trajectories (the moat recipe). Returns mean MSE over `horizon`
        steps between the rolled-out and the ground-truth future latents.

        latent0: (B, V, D)   initial visible latent
        actions: (B, H, action_dim)  (or a dict of streams, each (B, H, *))
        target_latents: list[(B, V, D)] length H -- encoder features of real future
        """
        preds = self(latent0, actions, horizon=len(target_latents))
        if len(preds) != len(target_latents):
            raise ValueError("horizon mismatch between actions and targets")
        return sum(F.mse_loss(p, t.detach()) for p, t in zip(preds, target_latents)) / len(preds)


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


# ---------------------------------------------------------------------------
# Extension 1: heterogeneous / multi-stream action space
# ---------------------------------------------------------------------------
class MultiActionEmbedder(nn.Module):
    """Map a *heterogeneous* action space into the conditioning token.

    `streams` is a dict name -> (kind, size), where:
      - ("continuous", d) -> a continuous action vector of dim d, MLP-embedded
      - ("discrete", n)   -> a discrete action of n classes, embedded via Embedding
    Each stream is embedded to `dim`, then summed (after a final linear mix) into
    one (B,1,dim) token -- so a world model can condition on e.g. a 7-DoF arm
    command *and* a discrete do(x) intervention (see causal.InterventionHead) at
    once. This is exactly the "multi-action space" a real embodiment needs.
    """

    def __init__(self, streams: dict, dim: int, hidden: int = 256):
        super().__init__()
        self.names = list(streams.keys())
        self.emb = nn.ModuleDict()
        for name, (kind, size) in streams.items():
            if kind == "discrete":
                self.emb[name] = nn.Embedding(int(size), dim)
            else:  # continuous
                d = int(size)
                self.emb[name] = nn.Sequential(
                    nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mix = nn.Linear(len(self.names) * dim, dim)
        nn.init.trunc_normal_(self.mix.weight, std=0.02)

    def forward(self, actions: dict) -> torch.Tensor:
        toks = []
        for name in self.names:
            a = actions[name]
            if isinstance(self.emb[name], nn.Embedding):
                a = a.long()
            toks.append(self.emb[name](a))
        return self.mix(torch.cat(toks, dim=-1)).unsqueeze(1)


# ---------------------------------------------------------------------------
# Extension 2: feature-space alignment with Meta V-JEPA 2 (the SOTA baseline)
# ---------------------------------------------------------------------------
class VJEPA2Projector(nn.Module):
    """Bridge V-JEPA 2's encoder output (B, 8192, 1024) into the token grid an
    ActionRollout can act on, *without* re-embedding.

    8192 = 32 x 16 x 16 context tokens of a ViT-L/16 at 64 frames @256^2; dim
    1024 is V-JEPA 2's token dimension. We learn a token-merge (8192 -> out)
    plus a per-token linear, so the rollout operates directly in V-JEPA 2's
    1024-d semantic space -- predictions stay comparable to the SOTA model's
    own features (the point of "aligning with V-JEPA 2").
    """

    def __init__(self, in_tokens: int = 8192, out_tokens: int = 1024, dim: int = 1024):
        super().__init__()
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.merge = nn.Parameter(torch.zeros(in_tokens, out_tokens))
        nn.init.trunc_normal_(self.merge, std=0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, in_tokens, dim) -> (B, out_tokens, dim)
        # einsum "bnd,no->bod": b=B, n=in_tokens, d=dim, o=out_tokens -> (B, o, d)
        merged = torch.einsum("bnd,no->bod", z, self.merge)  # (B, out_tokens, dim)
        return self.proj(merged)


class VJEPA2AlignedRollout(nn.Module):
    """Action-conditioned rollout that lives in V-JEPA 2's latent space.

    Drop in a V-JEPA 2 encoder's (B, 8192, 1024) features; this projects them to
    a manageable token grid and runs ActionRollout in 1024-d space. Means a
    single world model can both *score* SOTA encoder features (via KINE-Bench)
    and *act* on them (plan) -- closing the loop between benchmark and model.
    """

    def __init__(self, out_tokens: int = 1024, dim: int = 1024, depth: int = 6,
                 heads: int = 12, action_dim: int = 8, style: str = "add",
                 action_embed: nn.Module | None = None, latent_clip: float | None = None):
        super().__init__()
        self.projector = VJEPA2Projector(in_tokens=8192, out_tokens=out_tokens, dim=dim)
        self.rollout = ActionRollout(dim, depth=depth, heads=heads, action_dim=action_dim,
                                     style=style, action_embed=action_embed, latent_clip=latent_clip)

    def encode_project(self, vjepa2_latent: torch.Tensor) -> torch.Tensor:
        return self.projector(vjepa2_latent)

    def step(self, vjepa2_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.rollout.step(self.projector(vjepa2_latent), action)

    def forward(self, vjepa2_latent: torch.Tensor, actions: torch.Tensor,
                horizon: int | None = None) -> list:
        z = self.projector(vjepa2_latent)
        return self.rollout(z, actions, horizon=horizon)
