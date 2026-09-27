import unittest
import torch
from kineworld_jepa.rollout import ActionRollout


class RolloutTrainingTest(unittest.TestCase):
    def test_training_loss_updates_predictor_without_training_targets(self):
        torch.manual_seed(17)
        model = ActionRollout(dim=16, depth=1, heads=4, action_dim=2).eval()
        latent = torch.randn(2, 4, 16)
        actions = torch.randn(2, 2, 2)
        targets = [torch.randn(2, 4, 16, requires_grad=True) for _ in range(2)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        loss = model.training_loss(latent, actions, targets)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(t.grad is None for t in targets))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()))
        self.assertLess(model.training_loss(latent, actions, targets).item(), loss.item())


if __name__ == '__main__':
    unittest.main()
