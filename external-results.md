# External (Myriad) results E1–E3 — scored against the pre-registration of 2026-07-02

**Date analyzed:** 2026-07-03. Files: results_gleam_e{1,2,3}.jsonl (5 seeds/config, 10M steps),
analysis_report.txt (analyze_runs.py output). All numbers below recomputed from the raw jsonl.

## Headline

**The central replication PASSED.** Verbatim-CleanRL-mirror PPO, memoryless λ=0.9:
J_deploy = 0.3105 ± 0.0112, implied gain 8.5 (median 7.1) — runaway inflation, plateau at/above
the predicted broken-learner level. The λ=1.0 twin: J = 0.2260 ± 0.0032, gain 1.8 — no inflation.
**Neither E1 kill criterion triggered** (configs differ decisively; plateau ≥ 0.25).

## Gate-by-gate (as registered — no reinterpretation)

| gate | verdict | numbers |
|---|---|---|
| E1.1 ml λ=0.9 in [0.26,0.31], gain>5 | **FAIL (marginal, high side)** | median 0.3175, mean 0.3105±0.0112; gain ✓ |
| E1.2 λ=1.0 no inflation | **PASS** | gain 1.8, J 0.226 |
| E1.3 stack32 ≤ 0.23 | **FAIL** | 0.3211±0.0131, gain 16.4 — identical to memoryless |
| E1.4 λ-sweep monotone, big drop 0.95→0.999 | **FAIL (location)** | monotone ✓ (0.311→0.261→0.237→0.227) but largest drop is 0.9→0.95 |
| E2.1 HalfCheetah λ-benefit PO ≥ 3× full | **FAIL** | λ:0.9→0.99 *hurts* both (means 3034→1244 full; 3124→1478 PO); seed sd ±1900 |
| E2.1 Walker2d | **PASS (qualified)** | PO improves +371 (594→965); full-obs *collapses* −1525 (2673→1148), making the ratio test trivial |
| E3 λ=0.9 restoration (all noise types) | **PASS** | gains 1.3–3.3 across clean/white/AR at 0.03 and 0.3 |
| E3 λ=0 white ruins / AR restores | **FAIL (half)** | AR 0.03 restores ✓ (gain 1.18); white 0.03 does NOT ruin (gain 2.47±0.39 vs predicted >6) |

E2 kill criterion (prediction 1 failing on BOTH tasks) **not triggered** — Walker passed.

## Two harness bugs found (affect three gates)

1. **frame-skip was never implemented.** `ppo_torch.py` has no `--frame-skip` argument and
   `configs_e2.txt` skip4 lines carry no skip flag; the skip4 rows are byte-identical duplicates
   of the non-skip runs. E2 prediction 3 is **untested**, not failed.
2. **stack32 behaves exactly like memoryless** (J 0.3211 vs 0.3105; inflated gain), contradicting
   the in-family numpy result (stack32 restored, J = 0.2229) and common frame-stack practice.
   Config plumbing looks correct in code (FrameStack + Flatten on both train and eval env; eval
   reads newest frame). Cannot be diagnosed from results.jsonl alone — needs training curves.
   Treat as suspected harness/training pathology pending rerun, not as evidence either way.

## Post-hoc analyses (flagged as post-hoc; the registered gates above stand)

- **E1.1 is consistent with the σ-conditional theory.** Learned log σ collapsed to −4.5
  (σ² ≈ 1.2e−4 vs the 0.09 the registered window assumed). At collapsed exploration the χ law
  sends k_eq → 2/h and J(k_eq) → rΘK(0) = 0.300 exactly; observed 0.3105 ± 0.0112 brackets it.
  The registered window was computed at the wrong σ; the theory evaluated at the observed σ is
  confirmed. (The entropy-collapse feedback loop is itself a prediction of the χ law.)
- **E1.4 transition location:** seeds at λ=0.95 are bimodal (gain 3.3 ± 3.0 — some broken, some
  restored): 0.95 sits inside the transition, consistent with a collapse point shifted by the
  σ-collapse (smaller ν moves λ_crit); note λ=0.99/0.999 runs did NOT collapse σ (log σ ≈ −2.2),
  so the sweep confounds λ with exploration. Post-hoc, directionally consistent, needs a
  fixed-σ sweep to be clean.
- **E3 λ=0 white:** direction correct (white 2.47 vs AR 1.18 vs clean 0.86) but magnitude far
  below the numpy run (9.74). Differences to bisect: 10M vs 2.6M steps (ruin-then-recover?),
  num_envs/batch structure, Adam vs numpy optimizer, orthogonal init. The neural cliff row of
  the paper must be downgraded from "confirmed" to "implementation-dependent, under diagnosis."
- **E2 reality check:** velocity-only HalfCheetah ≈ full-obs HalfCheetah at λ=0.9 (3124 vs 3034)
  — HC is known to be insensitive to velocity-only masking (it cannot fall); it was a weak testbed
  choice. Walker (which can fall; position matters) shows the predicted direction. High λ hurting
  full-obs Walker (−1525) at fixed 10M budget is the usual variance cost of λ→1 and does not
  contradict the theory, but it makes the registered ratio gate too easy — a better gate would
  have compared PO improvement against |full-obs change|.

## Required actions before any submission

1. Implement `--frame-skip` (action-repeat wrapper) in ppo_torch.py + fixed configs; rerun the
   four skip rows. (Small: ~20 lines + 20 array tasks.)
2. Rerun stack32 with training-curve logging and periodic deployed evals; diagnose train-vs-eval.
   If it restores with curves visible, the earlier result was a run pathology; if it robustly
   fails, that's a genuine and interesting deviation from the numpy agent to understand.
3. E3 λ=0 white: rerun with periodic deployed evals (time-course) and one numpy-matched config
   (same batch structure) to bisect the implementation dependence.
4. Fixed-σ λ-sweep (σ²=0.09 pinned) to decouple λ from entropy collapse in E1.4.
5. Paper: fill the E-tables with these results verbatim, PASS/FAIL as registered; add the
   σ-conditional analysis as a clearly-marked post-hoc subsection; downgrade the neural-cliff
   claim; report E2 as mixed (Walker direction ✓, HC null, skip untested).

## Round 2 (reruns with fixed harness, analyzed 2026-07-03)

**Determinism check passed implicitly:** `r_asym_*_lam0.0` and `r_stack32` reproduce their
round-1 numbers byte-for-byte (same configs, same seeds, CPU-deterministic) — the round-1
results were real, not flakes.

**1. Fixed-σ λ-sweep — the registered transition location is vindicated.** With σ² pinned at
0.09: J = 1.699 (λ=0.9, destabilized — see below), 0.3007 (λ=0.95, parked *exactly* at the
rΘK(0)=0.300 ceiling), 0.2225 (λ=0.99, restored), 0.2210 (λ=0.999). The collapse sits between
0.95 and 0.99 — precisely where the λ-exact theory (k_eq: 17.7 → 2.2) and the original
registered gate put it. Round 1's E1.4 failure was the entropy-collapse confound, now
demonstrated by controlled experiment. Note λ=0.9 fixed-σ *diverged* (J=1.70, negative implied
gain): with the entropy-collapse brake removed, the phantom crawl pushes into the stability
boundary — theory-consistent (equilibrium near kh=χ, stochastic overshoot at the edge), worth
the curves file before quoting.

**2. stack32 is real, reproducible, and changes the story (for the better).** Wrapper
verified, result byte-identical on rerun: the CleanRL-mirror MLP with a 32-frame stack at
λ=0.9 sits at the memoryless plateau (J=0.3165, gain 13.9), while the numpy MLP with the same
window restored (0.2229). Same information, same λ, different optimizer/init — different
basin. Reading: **input memory is availability, not use.** With the bootstrap horizon shorter
than the environment memory, the phantom equilibrium remains an attractor even when the input
contains everything needed, and whether the learner escapes it is implementation-dependent.
Only extending the bootstrap horizon restores reliably — which sharpens, not weakens, the
λ-is-the-master-knob thesis. (stack8 diagnostic: median broken as predicted, mean 0.91 ± 1.20
— some seeds catastrophically unstable.) Paper edit required: the frame-stack row becomes
"restoration via input history is fragile/implementation-dependent below λ_crit."

**3. E3 cliff: batch structure ruled out.** The numpy-matched torch variant (64×256) gives
gain 2.91 ≈ the CleanRL-shaped 2.49 — still no ruin. Ordering at λ=0 holds as predicted
(white 2.5–2.9 > AR 1.3 > clean 0.8) but the *magnitude* of white-noise ruin is
implementation-dependent (numpy 9.7 vs torch ~2.7). Remaining suspects: optimizer (Adam vs
numpy's), init, value-loss clipping, 10M-vs-2.6M time-course. Paper keeps the exact TD(0)
cliff as theory + reports neural magnitude as implementation-dependent, ordering confirmed.

**4. Frame-skip now real and directionally right.** Skip4 rows differ from non-skip (bug
fixed); on velocity-only HalfCheetah, skip4 halves the λ-sensitivity (Δ: −1080 → −580),
consistent with prediction 3's direction on an admittedly weak testbed.

**Registered gates are unchanged by round 2** (reruns are diagnostics, not substitutions):
E1.2, E2.1-Walker, E3-λ0.9 PASS; E1.1 (marginal high, σ-conditional theory bracketed),
E1.3, E1.4 (location — now explained), E2.1-HC, E3-λ0 (magnitude) FAIL as registered.

### Curves analysis (results/curves/, figure results/curves_analysis.png)

**stack32 — transient restoration, then capture (the headline).** All five seeds *start
restored* (J = 0.22–0.26, gains 3–5.6 at 0.33M — near the numpy final state) and are then
captured by the phantom basin: J climbs monotonically to 0.30–0.34, gains inflate to 7–33
(seed 4 fails differently: gain → −0.4, degenerate). Capture co-times with exploration
collapse (log σ → −4.3, σ² ≈ 2×10⁻⁴), exactly the χ-law feedback: shrinking ν strengthens the
phantom attractor. The numpy agent that stayed restored kept 50× more exploration
(log σ ≈ −2.2). So the stack32 story is not "frame stacks don't work" but sharper: **the
belief solution is only metastable below λ_crit — entropy collapse hands memory-equipped
agents back to the phantom basin.** This is a new, mechanism-predicted phenomenon (capture
dynamics), not an anomaly.

**stack8** — bimodal as suspected: seed 1 destabilizes outright (J = 3.18, negative gain);
seeds 2/4 phantom-captured (gains 9–32); 3/5 intermediate.

**fixed-σ λ=0.9 — overshoot, not parking.** Every seed inflates early (gains 7–19 by 1–2.6M),
then passes through the stability edge and breaks (gains flip negative, J → 0.6–2.1). With the
entropy brake removed, PPO does not park at the theoretical equilibrium k_eq=20.6 (kh≈1.0) but
overshoots through it. The λ=0.95 twin parks at 0.3007 = the ceiling. The monotone λ trend and
the 0.95→0.99 transition stand; quote the λ=0.9 fixed-σ number only as "destabilizes."

**E3 λ=0 white noise — no ruin at any time.** Gains hover at 1.4–3.5 for all 10M steps in both
batch shapes; the numpy-vs-torch gap is NOT a snapshot artifact (numpy was ruined by 2.6M; the
torch stack never is). Ordering white > AR > clean holds throughout. The neural cliff
magnitude is genuinely implementation-dependent — remaining suspects: Adam, orthogonal init,
value-loss clipping.

## Venue implication

Core in-family replication with a standard implementation: strong (E1 core + E3 λ=0.9).
Transfer evidence (E2): currently mixed and underpowered — not yet NMI-grade. The general-diffusion
Lemma A (theory-side generality) and a rerun E2 with better-chosen PO variants (Walker2d-style,
position-masked; POPGym) are the two paths that could change that.
