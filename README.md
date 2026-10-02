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

The CPU checks need no GPU, downloaded checkpoint or private training data.
Create a virtual environment and use its interpreter explicitly; activation and
pytest are not required. CPU dependency installation and tests were checked on
Linux with Python 3.12.

```bash
git clone https://github.com/kineworld/kine-jepa.git
cd kine-jepa
```

Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python tests/check_collection.py
```

Windows PowerShell (equivalent commands; not run in the Linux verification):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
.\.venv\Scripts\python.exe tests/check_collection.py
```

On macOS, use the Linux interpreter paths but install torch with
`.venv/bin/python -m pip install torch` instead of the CPU-wheel index.
For CUDA work, choose a compatible PyTorch build from the
[official installer](https://pytorch.org/get-started/locally/) rather than
installing the CPU wheel above.

The `-t .` option enables this repository's test loader, which includes
module-level test functions as well as `unittest.TestCase` methods.
`check_collection.py` verifies that every test defined in source is collected.
The optional upstream-solver tests explicitly skip when `stable-worldmodel`
is absent; see [requirements-swm.txt](requirements-swm.txt) for that separate
integration environment. A CPU suite pass verifies the software checks, not
learned-model performance. Planner reproducibility across PyTorch builds is
tracked in [issue #13](https://github.com/kineworld/kine-jepa/issues/13).

Some experiments require separately obtained upstream checkpoints. Third-party model code, weights and datasets retain their own licenses; this repository's MIT license does not override them.

## Research direction

KineWorld is exploring a non-LLM route to world modelling based on minimal predictive state, action-conditioned dynamics, uncertainty, active verification and online adaptation. Near-term work is evaluated by reproducible prediction and closed-loop control evidence rather than generated-video appearance.

## Planning with physical action limits

`LatentPlanner` and the optional `stable-worldmodel` solver adapter accept either scalar action bounds or one lower and upper limit per action dimension. For example, a two-axis action with different units can use `action_low=[-0.2, 0.4]` and `action_high=[0.1, 0.6]`. Invalid, nonfinite, or reversed limits fail before search. The native planner also keeps its sampling distribution finite when an iteration retains only one elite candidate. Pass `enforce_action_bounds=True` to `SWMPlanner` to make the optional upstream solver evaluate actions inside the same limits; upstream CEM does not clamp candidates by default. These are planning correctness controls, not evidence of learned-model quality.

The native CEM planner searches in normalized per-axis coordinates and maps
each candidate back into the supplied physical action limits before scoring.
Its sampling scale therefore follows each actuator's range rather than an
assumed action unit. Returned actions remain in physical units. The default
float32 `[-1, 1]` search is unchanged. Initial and goal latents must be
floating-point tensors on the same device, and `device` must identify it.
Candidates retain the initial latent's dtype; a different floating-point goal
dtype keeps PyTorch's usual distance promotion.

Reproduce the equal-budget action-unit correctness ablation without weights or
data using `python scripts/benchmark_action_units.py --output runs/action-units.json`.
It compares the previous raw-unit search and the normalized search on one
analytic system expressed in three equivalent action conventions. It does not
measure trained-model quality or real-robot performance.
The recorded CPU result is in [ACTION-UNITS-v1.json](EXPERIMENTS/ACTION-UNITS-v1.json).

## License

KineWorld-authored code is MIT licensed. See individual experiment files for upstream dependencies and evidence limitations.
