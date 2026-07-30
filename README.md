# Where Actor–Critic Converges Under Partial Observability

Code, data, and proofs accompanying the paper *"Where Actor–Critic Converges
Under Partial Observability: An Exactly Solvable Case."*

## Overview

Actor–critic methods are used routinely in partially observable environments,
yet *where* their learning dynamics converge has not been characterized. This
repository studies a family of continuous-control POMDPs with linear–Gaussian
hidden dynamics in which the coupled critic–actor learning dynamics can be
solved **exactly** rather than merely bounded.

The central object is the *learning equilibrium* — the rest point of the
expected update. With a memoryless critic it detaches from the optimum of the
actor's own policy class: on the default instance the best memoryless policy
costs J = 0.2196, while the learning equilibrium plateaus near 0.30. We obtain
its location and cost in closed form, show that the bias is governed by the
bootstrap parameter λ (and vanishes once the return horizon covers the hidden
timescale), and confirm the pre-registered predictions with a standard PPO
implementation.

## Repository layout

**Environment and exact theory**
- `gleam_bench.py` — the exactly-solvable POMDP family, shipping exact reference
  values per instance (belief-optimal cost, best memoryless gain, and the
  predicted learning-equilibrium plateau).
- `gleam_gym.py` — Gymnasium-compatible environment wrapper.
- `lstd_lambda.py` — closed-form λ-dependent equilibria.
- `n2_test.py`, `restoration.py`, `theorem2_learning.py`, `bath_universality.py`,
  `drift_multiseed.py` — the numerical derivations and checks behind the
  theorems and figures.

**Learning**
- `ppo_torch.py` — single-file PPO, a faithful mirror of CleanRL's
  `ppo_continuous_action`, with the experiment options.
- `ppo_np.py` — an independent, compact NumPy PPO used as a cross-check.
- `analyze_runs.py` — scores recorded runs against the registered predictions.
- `configs_*.txt` — experiment definitions.

**Records**
- `preregistration-external.md`, `external-results.md` — the predictions and
  kill criteria registered before the experiments, and the scored outcomes.
- `results/` — recorded run outputs and analysis.
- `report/` — the paper (LaTeX source and PDF); `report_figs/` — figures.
- `theorem/` — machine-checked proof notebooks.
- `tests/` — unit tests.

## Requirements

Python 3.11 with CPU PyTorch (the networks are small MLPs; no GPU needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium[mujoco]" numpy scipy
```

## Reproducing the results

Print the exact reference values for the default instance:

```bash
python gleam_bench.py
```

Train on the exactly-solvable environment and score the run:

```bash
python ppo_torch.py --env-id GLEAMBench-v0 --total-timesteps 200000 --out run.jsonl
python analyze_runs.py run.jsonl
```

The full experiments (five seeds per configuration, 10M steps each) are defined
in `configs_*.txt`. They were run as batch job arrays on an SGE cluster; the
included batch scripts can be adapted to any scheduler. Unit tests (CPU, no
MuJoCo required):

```bash
python tests/test_wrappers.py
```

## Reference values (default instance)

Belief-optimal cost J\* = 0.1989; best memoryless cost J = 0.2196 at gain
k\* = 1.65; learning-equilibrium cost ceiling rΘK(0) = 0.300.

## Citation

```bibtex
@article{gozel2026actorcritic,
  author  = {G\"ozel, Idil},
  title   = {Where Actor--Critic Converges Under Partial Observability:
             An Exactly Solvable Case},
  year    = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
