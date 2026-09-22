"""Optional shims that let a kine-jepa model run inside a third-party toolchain.

Nothing in this package is imported by the rest of ``kineworld_jepa``. A shim is only
loaded when someone explicitly asks for it, so kine-jepa keeps its own dependency set and
the upstream library keeps its own release cycle.

Currently one shim: :mod:`kineworld_jepa.interop.swm`, which exposes an ``ActionRollout``
through the ``Costable`` surface of ``stable-worldmodel`` so that its planning solvers can
be used in place of the hand-rolled CEM in :class:`kineworld_jepa.rollout.LatentPlanner`.
"""

from .swm import SWMCostModel, SWMPlanner, swm_available

__all__ = ["SWMCostModel", "SWMPlanner", "swm_available"]
