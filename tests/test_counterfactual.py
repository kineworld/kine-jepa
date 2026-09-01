import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kineworld_jepa.counterfactual import CounterfactualRollout

torch.manual_seed(0)


def make():
    m = CounterfactualRollout(dim=32, action_dim=4, depth=2, heads=4, latent_clip=5.0)
    m.eval()  # disable dropout -> deterministic, so same-id case is exactly 0
    return m


class TestCounterfactual(unittest.TestCase):
    def test_shape(self):
        m = make()
        z0 = torch.randn(2, 8, 32)
        arm = torch.randn(2, 5, 4)
        do = torch.randint(0, 4, (2, 5))
        out = m(z0, arm, do)
        self.assertEqual(len(out), 5)
        for o in out:
            self.assertEqual(tuple(o.shape), (2, 8, 32))

    def test_counterfactual_diverges(self):
        # same scene + same arm commands, only the do(x) id differs -> futures
        # must diverge. This is the what-if signal.
        m = make()
        z0 = torch.randn(2, 8, 32)
        arm = torch.randn(2, 6, 4)
        _, _, div = m.counterfactual(z0, arm, base_id=0, alt_id=1)
        self.assertTrue(div > 1e-4, f"counterfactual divergence too small: {div}")
        print(f"PASS test_counterfactual_diverges  div(do0 vs do1)={div:.4f}")

    def test_same_id_noop(self):
        # identical intervention ids -> identical rollouts -> zero divergence.
        m = make()
        z0 = torch.randn(2, 8, 32)
        arm = torch.randn(2, 6, 4)
        _, _, div = m.counterfactual(z0, arm, base_id=0, alt_id=0)
        self.assertAlmostEqual(div, 0.0, places=6)
        print(f"PASS test_same_id_noop  div={div:.6f}")

    def test_long_horizon_stable(self):
        # latent_clip must keep a 40-step recursive rollout bounded.
        m = make()
        z0 = torch.randn(1, 8, 32)
        arm = torch.randn(1, 40, 4)
        do = torch.randint(0, 4, (1, 40))
        out = m(z0, arm, do)
        max_norm = max(o.norm(dim=-1).max().item() for o in out)
        self.assertLessEqual(max_norm, 5.0 + 1e-3)
        print(f"PASS test_long_horizon_stable  max_token_norm={max_norm:.3f} (clip=5.0)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
