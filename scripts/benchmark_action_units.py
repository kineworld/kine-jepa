"""CPU correctness ablation for affine action-unit changes; no learned-model claim.

The raw-unit CEM baseline below reproduces LatentPlanner.plan at kine-jepa
commit 1b1b7e3d8e30f5b256696898e303c6ab6245463e (KineWorld, MIT).
Every representation describes the same analytic dynamics and goal, with the
same candidate budget and paired seeds. No checkpoint or external data is used.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kineworld_jepa.rollout import LatentPlanner


class UnitRollout:
    def __init__(self, low, high):
        self.low, self.high = torch.tensor(low), torch.tensor(high)
        self.evaluations = 0

    def __call__(self, latent, actions, horizon):
        self.evaluations += actions.shape[0]
        futures = []
        for step in range(horizon):
            unit_action = 2 * (actions[:, step] - self.low) / (self.high - self.low) - 1
            latent = latent + unit_action[:, None]
            futures.append(latent)
        return futures


@torch.no_grad()
def legacy_plan(rollout, latent, goal, low, high, seed):
    generator = torch.Generator().manual_seed(seed)
    mean = torch.zeros(64, 4, 2)
    std = torch.ones_like(mean) * 0.5
    best, best_loss = None, None
    for _ in range(8):
        actions = torch.maximum(torch.minimum(
            mean + std * torch.randn(64, 4, 2, generator=generator), high), low)
        final = rollout(latent.repeat(64, 1, 1), actions, 4)[-1]
        loss = (final.mean(1) - goal.mean(1)).square().sum(-1)
        indices = loss.argsort()[:6]
        elites = actions[indices]
        mean = elites.mean(0)
        std = (std * 0.4 + elites.std(0) * 0.6).clamp(0.01, 1.0)
        current = loss[indices[0]].item()
        if best_loss is None or current < best_loss:
            best, best_loss = elites[0], current
    return best, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [('canonical', [-1., -1.], [1., 1.]),
             ('shifted', [100., 100.], [200., 200.]),
             ('mixed_units', [-1000., 0.5], [-500., 0.75])]
    records = []
    start, goal = torch.zeros(1, 1, 2), torch.full((1, 1, 2), 2.)
    for label, low, high in cases:
        for seed in range(10):
            previous = UnitRollout(low, high)
            old_action, old_loss = legacy_plan(previous, start, goal, previous.low, previous.high, seed)
            current = UnitRollout(low, high)
            planner = LatentPlanner(current, goal, 2, horizon=4, action_low=low, action_high=high)
            new_action, new_loss = planner.plan(start, iters=8, candidates=64, seed=seed)
            assert previous.evaluations == current.evaluations == 512
            assert (new_action >= current.low).all() and (new_action <= current.high).all()
            if label == 'canonical':
                assert torch.equal(old_action, new_action) and old_loss == new_loss
            records.append(dict(representation=label, seed=seed, old_loss=old_loss,
                                new_loss=new_loss, candidates_evaluated=512))
    report = {'baseline_commit': '1b1b7e3d8e30f5b256696898e303c6ab6245463e',
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
              'metric': 'terminal squared error in unchanged canonical state coordinates',
              'horizon': 4, 'iterations': 8, 'candidates': 64, 'seeds': list(range(10)),
              'canonical_actions_and_losses': 'bitwise identical for all 10 paired seeds',
              'scope': 'analytic dynamics/planner correctness only; no learned-world-model performance claim',
              'summary': {name: {key: sum(row[key] for row in records if row['representation'] == name)/10
                                 for key in ('old_loss', 'new_loss')} for name, _, _ in cases},
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
