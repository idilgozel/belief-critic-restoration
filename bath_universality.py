"""Lemma A falsification test: bath-distribution universality of the equilibrium.

Claim under test (Lemma A, general-diffusion): the learning equilibrium k_eq and
its deployed cost depend on the hidden drive only through its zero-lag variance
C(0) (and weakly through timescale via O(h) corrections) — NOT its distribution.

Three stationary drives, all matched to C(0) = 0.3 and autocovariance 0.3*exp(-t):
  ou        : Gaussian OU (the solved family; pipeline validation vs exact 24.79)
  telegraph : dichotomous +-sqrt(0.3), flip rate 0.5  (bounded, non-Gaussian)
  ousq      : sqrt(0.15)*(w^2-1), w unit OU at gamma=0.5 (skewed, excess kurtosis ~12)

Empirical k_eq: for each k, simulate the stationary ZOH closed loop
(u = -k v + eta, eta ~ N(0, s_e), s_e = 0.09, h = 0.05, q = r = 1, rho = 0.05),
fit the LSTD(0) critic over (1, v, v^2) from the trajectory, average the
score-function actor update  E[delta * (-eta v / s_e)]  — its zero crossing in k
is the equilibrium. Deployed cost at k_eq: eta = 0 rollout, J = mean cost rate.

Everything Euler-integrated with nsub substeps (common integrator across baths so
integrator bias is common-mode).
"""
import sys
import numpy as np
from numba import njit

H, SE, Q, R, RHO = 0.05, 0.09, 1.0, 1.0, 0.05
BETA = float(np.exp(-RHO * H))
C0 = 0.3
NSUB = 8
BURN, T = 100_000, 1_000_000

@njit(cache=True)
def simulate2(bath, k, se, steps, seed):
    """Correct bookkeeping: records v_n (pre-step), v_{n+1}, per-step cost, eta_n."""
    np.random.seed(seed)
    dt = H / NSUB
    sq_se = np.sqrt(se)
    z = 0.0; w = 0.0
    if bath == 0:   z = np.sqrt(C0) * np.random.standard_normal()
    elif bath == 1: z = np.sqrt(C0) * (1.0 if np.random.random() < 0.5 else -1.0)
    else:           w = np.random.standard_normal()
    sig_ou = np.sqrt(2.0 * C0)
    gw = 0.5
    s2 = np.sqrt(C0 / 2.0)
    pflip = 0.5 * (1.0 - np.exp(-2.0 * 0.5 * dt))
    v = 0.0
    V = np.empty(steps); VN = np.empty(steps)
    CO = np.empty(steps); ET = np.empty(steps)
    for t in range(steps):
        v0 = v
        eta = sq_se * np.random.standard_normal() if se > 0 else 0.0
        u = -k * v0 + eta                       # ZOH: frozen over the interval
        c = 0.0
        for _ in range(NSUB):
            if bath == 0:
                zeta = z
                z += -z * dt + sig_ou * np.sqrt(dt) * np.random.standard_normal()
            elif bath == 1:
                zeta = z
                if np.random.random() < pflip: z = -z
            else:
                zeta = s2 * (w * w - 1.0)
                w += -gw * w * dt + np.sqrt(2.0 * gw * dt) * np.random.standard_normal()
            c += (Q * v * v + R * u * u) * dt
            v += (zeta + u) * dt
        V[t] = v0; VN[t] = v; CO[t] = c; ET[t] = eta
    return V, VN, CO, ET

def expected_update(bath, k, seed):
    v, vn, c, eta = simulate2(bath, k, SE, BURN + T, seed)
    v, vn, c, eta = v[BURN:], vn[BURN:], c[BURN:], eta[BURN:]
    phi  = np.stack([np.ones_like(v), v, v * v], 1)
    phin = np.stack([np.ones_like(vn), vn, vn * vn], 1)
    A = phi.T @ (phi - BETA * phin) / len(v)
    b = phi.T @ c / len(v)
    th = np.linalg.solve(A, b)
    delta = c + BETA * (phin @ th) - (phi @ th)
    g = delta * (-eta * v / SE)
    return g.mean(), g.std() / np.sqrt(len(g)), th[2]

def find_keq(bath, seed0=1000):
    ks = np.array([2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 28.0, 34.0])
    gs = []
    for i, k in enumerate(ks):
        g, se_, th2 = expected_update(bath, k, seed0 + i)
        gs.append(g)
        print(f"  k={k:5.1f}  ghat_emp={g:+.3e} (+-{se_:.1e})  th2={th2:7.2f}")
    gs = np.array(gs)
    # bracket the sign change, then 3 bisection rounds with fresh seeds
    idx = np.where(np.sign(gs[:-1]) != np.sign(gs[1:]))[0]
    if len(idx) == 0:
        print("  NO SIGN CHANGE on grid"); return np.nan
    lo, hi = ks[idx[0]], ks[idx[0] + 1]
    glo = gs[idx[0]]
    for j in range(3):
        mid = 0.5 * (lo + hi)
        g, se_, _ = expected_update(bath, mid, seed0 + 100 + j)
        print(f"  bisect k={mid:5.2f}  ghat_emp={g:+.3e} (+-{se_:.1e})")
        if np.sign(g) == np.sign(glo): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def deployed_J(bath, k, seed=9999):
    v, vn, c, eta = simulate2(bath, k, 0.0, BURN + T, seed)
    return c[BURN:].mean() / H

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = {"ou": 0, "telegraph": 1, "ousq": 2}
    todo = names if which == "all" else {which: names[which]}
    for name, code in todo.items():
        print(f"=== bath: {name} (C0 = {C0}) ===")
        keq = find_keq(code)
        J = deployed_J(code, keq) if np.isfinite(keq) else np.nan
        print(f"  -> k_eq = {keq:.2f}   k_eq*h = {keq * H:.3f}   "
              f"J(k_eq) = {J:.4f}   (r*C0 = {R * C0:.3f})\n")
