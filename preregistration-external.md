# Pre-registered external experiments (require torch/MuJoCo; not runnable in the analysis sandbox)

**Date registered:** 2026-07-02, before any external run. Interpretation rules fixed in advance, following the falsifiable-gate discipline of the original GLEAN draft §10.

## E1 — Verbatim CleanRL PPO on GLEAM-bench

**Setup:** `ppo_torch.py` — a pinned single-file mirror of CleanRL `ppo_continuous_action.py` (identical architecture, hyperparameters, GAE, clipped surrogate, value clipping, lr anneal), plus one verbatim-CleanRL cross-check run (one config, one seed; two-line env-registration patch, see myriad/README.md) whose agreement with the mirror within seed noise validates the implementation. Env = `GLEAMBench-v0` (wrapper `gleam_gym.py`, default instance c=0.3, γ=1, h=0.05), total_timesteps = 10M, 5 seeds per config. Execution: UCL Myriad, SGE array jobs (`myriad/`). Configs: (a) gae_lambda=0.9, obs=v; (b) gae_lambda=1.0, obs=v; (c) gae_lambda=0.9, obs=FrameStack(32).

**Predictions (from exact theory; numbers are the λ-exact values of `lstd_lambda.py`):**
1. Config (a) deployed cost rate → 0.28–0.30 (exact λ=0.9 equilibrium k_eq = 20.6, J = 0.293), i.e. plateaus *above* its policy-class optimum 0.2196; median |action|/|v| on large-|v| states exceeds 5 by 5M steps.
2. Config (b) shows no gain inflation (implied gain within [0.5, 3]) and deployed cost ≤ 0.24.
3. Config (c) deployed cost ≤ 0.23.
4. λ-sweep on config (a) (λ ∈ {0.9, 0.95, 0.99, 0.999}): deployed cost decreases monotonically, with the large drop between 0.95 and 0.999 (λ_crit law: horizon (1−βλ)⁻¹ vs 20-step bath).

**Kill criteria:** if (a) matches (b) within seed noise, the homemade-PPO results do not transfer to standard implementations and the deep-RL section must be retracted to the linear-AC claims. If (a) plateaus but at cost < 0.25, the quantitative transfer fails (mechanism may survive; magnitudes don't).

## E2 — Partially observed MuJoCo (transfer beyond the family)

**Setup:** HalfCheetah-v4 and Walker2d-v4 with observation masked to velocity-only components (the position-integrating hidden state plays the bath's role); CleanRL PPO, 5 seeds; λ ∈ {0.9, 0.99}; matched full-observation controls.

**Predictions:**
1. Δ(return | λ: 0.9→0.99) is positive and at least 3x larger on masked variants than on full-obs controls (paired across seeds).
2. Masked-variant agents at λ=0.9 show growing action-magnitude norms over training relative to λ=0.99 (the inflation signature), absent in full-obs controls.
3. Frame-skip 4 vs 1 on masked variants: skip-4 *reduces* the λ-sensitivity (larger effective h → lower λ_crit), opposite to or absent in full-obs controls.

**Kill criteria:** prediction 1 failing on both tasks falsifies the transfer claim (the in-family results stand; the paper's scope narrows to "solvable family + mechanism," and the MuJoCo section reports the negative). Predictions 2–3 are secondary (mechanism signatures); either failing weakens but does not kill.

## E3 — Neural belief-critic cliff (λ-gated restoration, standard stack)

**Setup:** GLEAM-bench, asymmetric PPO (actor sees v; critic sees v + z̃), z̃ = z + noise, white vs AR(1) τ=1 at matched variance Rb/Σzz ∈ {0.03, 0.3}; λ ∈ {0, 0.9}; 5 seeds.

**Predictions (already confirmed with the numpy implementation; E3 is the standard-stack replication):** at λ=0, white 0.03 ruins (gain > 6) while AR 0.03 restores (gain < 3); at λ=0.9 both restore at 0.03 and both mildly degrade at 0.3.

## Analysis code

Deployed-cost and implied-gain evaluators are in `ppo_np.py::evaluate` (framework-independent: run the trained policy deterministically, average h-normalized cost after burn-in, median −a/v on |v| > 0.02 states). Exact reference values per instance: `gleam_bench.GLEBench.baselines()`; λ-exact equilibria: `lstd_lambda.py::k_eq_lambda`.
