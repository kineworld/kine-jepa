"""Plan a kine-jepa latent model with ``stable-worldmodel``'s solvers.

Why this module exists
----------------------
:class:`kineworld_jepa.rollout.LatentPlanner` is a hand-rolled cross-entropy method:
about thirty lines, one hyper-parameter set, no warm-start, no elite retention, no
gradient path. ``stable-worldmodel`` (MIT, ``galilai-group``) is the upstream of LeWM and
ships eight maintained solvers behind a single structural contract::

    model.get_cost(info_dict, action_candidates) -> Tensor of shape (B, S)

That one method *is* the ``Costable`` protocol in ``stable_worldmodel/solver/solver.py``.
Everything else -- the sampling distribution, the elite update, the momentum, the
projection -- belongs to the solver. So the whole cost of reusing CEM, iCEM, MPPI,
Predictive Sampling, SGD, PGD and the augmented-Lagrangian solver is one adapter, and
kine-jepa does not own any of that search code.

What this adapter does not do
-----------------------------
It does not turn pixels into latents. The latents it plans over must already exist: kine-
jepa's ``ActionRollout`` operates on frozen encoder features, so ``info_dict`` carries
``latent`` directly. For the pixel -> latent leg see the V-JEPA 2 adapter in the
``kine-bench`` repository (``kinebench/adapters/vjepa2.py``). A model that needs both is
kine-bench for the encoding and this module for the planning.

Contract details, so the next reader does not have to re-derive them
------------------------------------------------------------------
* ``latent`` is ``(B, V, D)`` (one observation) or ``(B, S, V, D)`` (a solver's expanded
  candidate batch, where ``S`` is the number of samples). Both are accepted.
* ``predicted_emb`` is ``(B, S, 1 + T, D)``: entry 0 is the pooled context latent and the
  remaining ``T`` entries are the rolled-out futures, matching the convention used by
  ``stable_worldmodel.wm.lewm.LeWM.rollout``.
* The token axis ``V`` is mean-pooled. This is not an arbitrary choice: it is exactly the
  reduction ``LatentPlanner._distance`` already performs, so :meth:`SWMCostModel.criterion`
  and kine-jepa's own distance agree elementwise. ``tests/test_swm_interop.py`` asserts
  that equality, because a silent change to either convention would quietly change every
  number reported downstream.

Version note
------------
Written against the released ``stable-worldmodel==0.1.1``, whose solvers type against
``Costable``. Upstream ``main`` additionally documents a ``Dynamics`` + ``Objective``
composition surface (``ShootingCostEvaluator``) that the released wheel does not contain.
This module targets the released API; ``encode``/``rollout`` follow the naming of that
newer surface so that the adapter does not have to be rewritten when it lands.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from kineworld_jepa.rollout import _action_bounds


def swm_available() -> bool:
    """Return True when ``stable-worldmodel`` can be imported in this interpreter.

    Callers use this to skip honestly instead of reporting a pass they did not earn.
    """
    try:
        import stable_worldmodel  # noqa: F401
    except Exception:
        return False
    return True


def _pool(z: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis ``V``.

    ``(..., V, D) -> (..., D)``. See the module docstring: this matches the reduction
    ``LatentPlanner._distance`` already uses, which is what keeps the two cost functions
    numerically identical.
    """
    return z.mean(dim=-2)


def _as_sample_batch(z: torch.Tensor) -> torch.Tensor:
    """Normalise ``(B, V, D)`` to ``(B, 1, V, D)``; leave ``(B, S, V, D)`` alone."""
    return z.unsqueeze(1) if z.dim() == 3 else z


class SWMCostModel(torch.nn.Module):
    """Adapt an ``ActionRollout`` to ``stable-worldmodel``'s ``Costable`` protocol.

    ``stable-worldmodel``'s solvers need exactly one method from a world model::

        get_cost(info_dict, action_candidates) -> (B, S)

    ``info_dict`` must carry ``latent`` (see the module docstring) and ``goal_latent``.
    A cached ``goal_emb`` is reused across solver iterations, mirroring how
    ``LeWM.get_cost`` avoids re-encoding the goal on every call.

    Args:
        rollout: an ``ActionRollout`` (or ``VJEPA2AlignedRollout``, which shares the
            ``(N, T, A) -> list[(N, V, D)]`` calling convention).
        action_dim: flattened action dimension. Read from the rollout when omitted.
        action_low / action_high: scalar or per-axis limits. When given, every
            candidate is clipped into this box before the dynamics see it. Off by
            default because upstream solvers (except ``ICEMSolver``) do not clamp;
            see :meth:`_project`.

    Example:
        >>> from kineworld_jepa.rollout import ActionRollout
        >>> from kineworld_jepa.interop.swm import SWMCostModel
        >>> model = ActionRollout(32, depth=4, heads=4, action_dim=4).eval()
        >>> cost_model = SWMCostModel(model)
        >>> cost = cost_model.get_cost(
        ...     {"latent": torch.randn(1, 24, 32), "goal_latent": torch.randn(1, 24, 32)},
        ...     torch.randn(1, 8, 8, 4),   # (B, S, horizon, action_dim)
        ... )
        >>> tuple(cost.shape)
        (1, 8)
    """

    def __init__(
        self,
        rollout: torch.nn.Module,
        action_dim: int | None = None,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.dynamics = rollout
        resolved = action_dim
        if resolved is None:
            resolved = getattr(rollout, "action_dim", None)
        if resolved is None:
            raise ValueError(
                "action_dim is required when the rollout does not expose one"
            )
        self.action_dim = int(resolved)
        if (action_low is None) != (action_high is None):
            raise ValueError("pass both action_low and action_high, or neither")
        self.action_low, self.action_high = (
            _action_bounds(action_low, action_high, self.action_dim)
            if action_low is not None else (None, None)
        )

    def _project(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip ``actions`` into ``[action_low, action_high]`` when bounds are set.

        Opt-in, and off by default, because it is **not** upstream behaviour: of the eight
        solvers, only ``ICEMSolver`` reads ``configure(action_space=...)`` and clamps its
        samples. ``CEMSolver``, ``MPPISolver`` and ``PredictiveSamplingSolver`` record the
        space and never enforce it -- ``CEMSolver.configure`` merely warns when the space is
        not a ``Box``. So an unbounded upstream solver can propose actions a real actuator
        cannot execute, and kine-jepa's ``LatentPlanner`` is the one that clamps.

        Turning this on narrows the search to the same box ``LatentPlanner`` uses, which is
        what makes a like-for-like comparison possible. It is applied to the candidates the
        dynamics actually see, so the elite selection also never scores an out-of-box
        action -- clipping only the returned sequence would leave the search exploring
        outside the box and then reporting a plan it never evaluated.
        """
        if self.action_low is None:
            return actions
        low = self.action_low.to(device=actions.device, dtype=actions.dtype)
        high = self.action_high.to(device=actions.device, dtype=actions.dtype)
        return torch.maximum(torch.minimum(actions, high), low)

    # -- the newer `Dynamics` surface, implemented under its own names ----------------

    def encode(self, info: dict) -> dict:
        """Pass a pre-encoded latent through.

        Mirrors ``LeWM.encode`` in shape and intent, but there is no pixel encoder to run:
        kine-jepa's rollout consumes frozen encoder features. If ``latent`` is absent the
        call fails loudly rather than inventing an embedding, because a silently wrong
        latent would be indistinguishable from a bad plan.
        """
        latent = info.get("latent", info.get("emb"))
        if latent is None:
            raise KeyError(
                "info_dict needs 'latent' of shape (B, V, D) or (B, S, V, D). This adapter "
                "does not encode pixels; use kine-bench's V-JEPA 2 adapter for that leg."
            )
        info["latent"] = latent
        return info

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        """Roll candidate action sequences forward and store ``predicted_emb``.

        Args:
            info_dict: carries ``latent``.
            action_candidates: ``(B, S, horizon, action_dim)``.

        Returns:
            ``info_dict`` with ``predicted_emb`` of shape ``(B, S, 1 + horizon, D)``.
        """
        self.encode(info_dict)
        latent = _as_sample_batch(info_dict["latent"])
        B, S, V, D = latent.shape
        horizon = int(action_candidates.shape[2])

        if int(action_candidates.shape[0]) != B:
            raise ValueError(
                f"latent batch {B} does not match action batch "
                f"{int(action_candidates.shape[0])}"
            )
        # A solver that expands the info dict hands over one latent per sample. A direct
        # call may pass a single latent next to S>1 candidates; broadcast it rather than
        # failing on a shape the caller had no reason to expect.
        if latent.shape[1] == 1 and int(action_candidates.shape[1]) > 1:
            latent = latent.expand(B, int(action_candidates.shape[1]), V, D)
            S = int(action_candidates.shape[1])

        # Opt-in actuator projection; a no-op unless bounds were passed. See `_project`.
        action_candidates = self._project(action_candidates)

        flat_latent = latent.reshape(B * S, V, D)
        flat_actions = action_candidates.reshape(
            B * S, horizon, action_candidates.shape[-1]
        )
        futures = self.dynamics(flat_latent, flat_actions, horizon=horizon)

        pooled = torch.stack(
            [_pool(f).reshape(B, S, D) for f in futures], dim=2
        )
        info_dict["predicted_emb"] = torch.cat(
            [_pool(latent).unsqueeze(2), pooled], dim=2
        )
        return info_dict

    # -- the `Costable` surface the released solvers actually call --------------------

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """Distance from the final predicted embedding to the goal embedding.

        Mirrors ``LeWM.criterion``: last-step mean-squared error summed over the feature
        axis, returning ``(B, S)``. Summed -- not averaged -- over features, which is what
        makes this equal to kine-jepa's ``LatentPlanner._distance``.
        """
        predicted = info_dict["predicted_emb"]
        goal = info_dict["goal_emb"]
        goal = _pool(goal).unsqueeze(2).expand_as(predicted)
        return F.mse_loss(
            predicted[..., -1:, :], goal[..., -1:, :].detach(), reduction="none"
        ).sum(dim=tuple(range(2, predicted.dim())))

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Cost of ``action_candidates``; the one entry point ``stable-worldmodel`` needs."""
        if "goal_emb" not in info_dict:
            if "goal_latent" not in info_dict:
                raise KeyError("info_dict needs 'goal_latent' (or a cached 'goal_emb')")
            info_dict["goal_emb"] = _as_sample_batch(info_dict["goal_latent"])
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


class SWMPlanner:
    """Goal-conditioned planner backed by a ``stable-worldmodel`` solver.

    Drop-in for :class:`kineworld_jepa.rollout.LatentPlanner`: same constructor shape,
    same ``plan(latent0) -> (actions, distance)`` return, and the returned distance is
    computed with kine-jepa's own reduction so the two are directly comparable.

    Differences from ``LatentPlanner`` that a caller should know about, because they are
    properties of the upstream solvers and not of this wrapper:

    * Upstream samplers draw an unclamped Gaussian, **except** ``ICEMSolver``, which reads
      the ``action_space`` handed to ``configure`` and clamps its candidates. So of the six
      solvers here, only ``icem`` honours ``action_low`` / ``action_high`` natively;
      ``cem``, ``mppi`` and ``predictive_sampling`` ignore them. ``LatentPlanner`` clamps
      every candidate. Pass ``enforce_action_bounds=True`` to clip the candidates those
      solvers see, which is what makes an equal-box comparison possible -- otherwise a
      comparison against ``LatentPlanner`` varies two things at once (solver *and* search
      space), and the difference will be attributed to the wrong one.
    * ``LatentPlanner.plan`` returns the single best elite. ``CEMSolver`` and
      ``ICEMSolver`` return the *mean* of the elites, ``PredictiveSamplingSolver`` the
      single best sample. This wrapper re-evaluates whatever the solver returns, so the
      reported number is always the distance actually achieved by the returned actions.

    Args:
        rollout: the ``ActionRollout`` to plan with.
        goal_latent: target latent, ``(B, V, D)``.
        action_dim: flattened action dimension.
        horizon: planning horizon in steps.
        solver: one of ``cem``, ``icem``, ``mppi``, ``predictive_sampling``, ``gd``,
            ``pgd``. All are upstream classes, constructed with the keyword arguments in
            ``solver_kwargs``.
        seed: passed to the solver's own generator.
        device: torch device string.
        action_low / action_high: scalar or per-axis action box. Recorded in the
            space handed to ``configure`` (honoured natively by ``icem``), and
            enforced on every solver's candidates when ``enforce_action_bounds`` is set.
        enforce_action_bounds: clip candidates into ``[action_low, action_high]`` before
            the dynamics see them. Default ``False`` reproduces upstream behaviour exactly.
        **solver_kwargs: forwarded to the solver constructor, e.g. ``num_samples=64``,
            ``n_steps=8``, ``topk=6``, ``var_scale=0.5``.
    """

    SOLVERS = ("cem", "icem", "mppi", "predictive_sampling", "gd", "pgd")

    def __init__(
        self,
        rollout: torch.nn.Module,
        goal_latent: torch.Tensor,
        action_dim: int,
        horizon: int = 8,
        solver: str = "cem",
        seed: int = 0,
        device: str = "cpu",
        action_low=-1.0,
        action_high=1.0,
        enforce_action_bounds: bool = False,
        **solver_kwargs: Any,
    ):
        if not swm_available():
            raise RuntimeError(
                "stable-worldmodel is not importable. Install it with "
                "`pip install -r requirements-swm.txt` (see that file for the import-time "
                "extras it needs), or keep using the in-tree LatentPlanner."
            )
        if solver not in self.SOLVERS:
            raise ValueError(f"unknown solver {solver!r}; expected one of {self.SOLVERS}")

        self.dynamics = rollout
        self.goal = goal_latent.detach()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.device = device
        self.action_low, self.action_high = _action_bounds(
            action_low, action_high, self.action_dim
        )
        self.enforce_action_bounds = bool(enforce_action_bounds)
        self.solver_name = solver
        self.cost_model = SWMCostModel(
            rollout,
            action_dim=self.action_dim,
            action_low=action_low if self.enforce_action_bounds else None,
            action_high=action_high if self.enforce_action_bounds else None,
        )

        from stable_worldmodel import solver as swm_solver

        cls = getattr(swm_solver, _SOLVER_CLASSES[solver])
        self.solver = cls(
            self.cost_model, device=device, seed=seed, **solver_kwargs
        )
        self._configured = False

    def configure(self, n_envs: int) -> None:
        """Set the solver up for ``n_envs`` parallel environments."""
        import numpy as np
        from gymnasium.spaces import Box

        import stable_worldmodel as swm

        low = np.broadcast_to(self.action_low.numpy(), (n_envs, self.action_dim)).copy()
        high = np.broadcast_to(self.action_high.numpy(), (n_envs, self.action_dim)).copy()
        flat = Box(low=low, high=high, dtype=np.float32)
        config = swm.PlanConfig(
            horizon=self.horizon,
            receding_horizon=1,
            history_len=1,
            action_block=1,
        )
        self.solver.configure(action_space=flat, n_envs=n_envs, config=config)
        self._configured = True

    @torch.no_grad()
    def _achieved_distance(self, latent0: torch.Tensor, actions: torch.Tensor) -> float:
        """Distance kine-jepa would report for ``actions``, using its own reduction."""
        futures = self.dynamics(latent0, actions, horizon=actions.shape[1])
        z = futures[-1].mean(dim=1)
        g = self.goal.mean(dim=1)
        return (z - g).pow(2).sum(dim=-1).mean().item()

    def plan(self, latent0: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Return ``(actions, achieved_distance, wall_seconds)``.

        ``actions`` is ``(B, horizon, action_dim)`` exactly as the solver produced it, and
        ``achieved_distance`` is measured on those actions rather than read out of the
        solver's own bookkeeping -- so a solver that reports its elite mean is scored on
        the elite mean it actually returns.
        """
        if not self._configured:
            self.configure(latent0.shape[0])
        info = {"latent": latent0, "goal_latent": self.goal}
        started = time.perf_counter()
        outputs = self.solver.solve(info)
        elapsed = time.perf_counter() - started
        actions = outputs["actions"].to(latent0.device)
        # With bounds enforced the candidates are already clipped, so this is a no-op in
        # practice; it is kept explicit so the returned plan is in-box by construction and
        # not by an argument about convexity. Without it, `icem` is in-box (it clamps
        # natively) while `cem` is not -- a silent asymmetry between arms.
        if self.enforce_action_bounds:
            low = self.action_low.to(device=actions.device, dtype=actions.dtype)
            high = self.action_high.to(device=actions.device, dtype=actions.dtype)
            actions = torch.maximum(torch.minimum(actions, high), low)
        return actions, self._achieved_distance(latent0, actions), elapsed


_SOLVER_CLASSES = {
    "cem": "CEMSolver",
    "icem": "ICEMSolver",
    "mppi": "MPPISolver",
    "predictive_sampling": "PredictiveSamplingSolver",
    "gd": "GradientSolver",
    "pgd": "PGDSolver",
}


__all__ = ["SWMCostModel", "SWMPlanner", "swm_available"]
