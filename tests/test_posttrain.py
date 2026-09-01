import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kineworld_jepa.counterfactual import CounterfactualRollout
from posttrain import SyntheticWorld, train, rollout_mse

torch.manual_seed(0)


class TestPostTrain(unittest.TestCase):
    def test_trained_beats_untrained(self):
        # Small, fast setting: proves the moat recipe (teacher-forcing on a
        # known action+intervention dynamics) actually lowers rollout error.
        world = SyntheticWorld()
        untrained = CounterfactualRollout(dim=32, action_dim=4, depth=2, heads=4, latent_clip=None)
        untrained.eval()
        trained = CounterfactualRollout(dim=32, action_dim=4, depth=2, heads=4, latent_clip=None)
        train(trained, world, epochs=160, batch=64, horizon=10, lr=3e-3)
        trained.eval()
        um = rollout_mse(untrained, world)
        tm = rollout_mse(trained, world)
        self.assertLess(tm, um * 0.5,
                        f"trained rollout MSE {tm:.4f} not < half of untrained {um:.4f}")
        print(f"PASS test_trained_beats_untrained  untrained={um:.4f} -> trained={tm:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
