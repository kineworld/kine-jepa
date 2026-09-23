# KineWorld Latent Dynamics Research

Open research code for compact, action-conditioned latent world models that can be studied on consumer hardware.

> **Status:** research-stage. This repository contains architecture prototypes, interfaces and controlled experiments. It does not contain a validated general world model, a production-ready causal reasoner or evidence of model leadership.

## What is here

- a clean-room spatiotemporal encoder implementation;
- action-conditioned latent rollout and CEM planning interfaces;
- counterfactual intervention interfaces for controlled experiments;
- synthetic and small-sample post-training probes;
- CPU regression tests and consumer-GPU profiling scripts.

The historical names `KineOne-WM`, `KINE-JEPA` and `KINE-EXP-*` remain in code and artifacts for reproducibility. They should not be interpreted as separate deployed products.

## Evidence boundary

Results in this repository establish that the software paths execute and that selected proof-of-concept objectives can be optimized under their stated settings. They do **not** establish real-world planning utility, causal identification, cross-task generality or superiority over another model.

KineWorld's current evidence ledger and latest risk-analysis artifacts are published through [KINE-Bench](https://github.com/kineworld/kine-bench) and [kineworld.com](https://kineworld.com).

## Quick start

```bash
git clone https://github.com/kineworld/kine-jepa.git
cd kine-jepa
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest tests/
```

Some experiments require separately obtained upstream checkpoints. Third-party model code, weights and datasets retain their own licenses; this repository's MIT license does not override them.

## Research direction

KineWorld is exploring a non-LLM route to world modelling based on minimal predictive state, action-conditioned dynamics, uncertainty, active verification and online adaptation. Near-term work is evaluated by reproducible prediction and closed-loop control evidence rather than generated-video appearance.

## Planning with physical action limits

`LatentPlanner` and the optional `stable-worldmodel` solver adapter accept either scalar action bounds or one lower and upper limit per action dimension. For example, a two-axis action with different units can use `action_low=[-0.2, 0.4]` and `action_high=[0.1, 0.6]`. Invalid, nonfinite, or reversed limits fail before search. The native planner also keeps its sampling distribution finite when an iteration retains only one elite candidate. Pass `enforce_action_bounds=True` to `SWMPlanner` to make the optional upstream solver evaluate actions inside the same limits; upstream CEM does not clamp candidates by default. These are planning correctness controls, not evidence of learned-model quality.

## License

KineWorld-authored code is MIT licensed. See individual experiment files for upstream dependencies and evidence limitations.
