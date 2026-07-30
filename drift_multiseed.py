"""Multi-seed drift-from-optimum rerun (Phase 0, item 13).

Replaces the two-seed result behind Fig 1(b): online TD(0) actor-critic on the
default N=1 instance, initialized at k* = 1.649, s_e = 0.09, critic warm-started
at its exact Prop-1 fixed point. Six independent chains + the mean-field
trajectory from integrating the exact expected-update field ghat(k) (Prop 2).

Calibration check (frozen k, critic pinned at fixed point) verifies that the
score-function estimator's expectation equals ghat: measured ratios
1.000 / 1.019 / 0.988 at k = 1.65 / 3 / 6 -> CAL = 1.0 used below.

Outputs: drift_multiseed.csv (chain trajectories + mean-field), summary stdout.
Engine is numba-jitted so the full 1.2M-step run takes seconds.
"""
import numpy as np
from numba import njit
from scipy.linalg import solve_discrete_lyapunov
from gleam_bench import GLEBench

SE, KSTAR = 0.09, 1.649
NSEEDS, STEPS, SNAP = 6, 1_200_000, 1_000
ALPHA_C, ALPHA_A = 5e-3, 1e-4
CAL = 1.0

env = GLEBench(n_envs=NSEEDS, seed=7)
beta = float(np.exp(-env.rho * env.h))
F, Gc, Lw, L0 = env.F, env.G[:, 0].copy(), env.Lw, env.L0
H, Q, R = env.h, env.q, env.r

def ghat(k):
    Krow = np.zeros((1, env.nx)); Krow[0, 0] = k
    Fk = env.F - env.G @ Krow
    if np.max(np.abs(np.linalg.eigvals(Fk))) >= 1: return np.nan
    Sig = solve_discrete_lyapunov(Fk, SE * env.G @ env.G.T + env.Sd)
    m2 = Sig[0, 0]; x = (Fk @ Sig)[0, 0]
    qt = env.q + env.r * k * k
    th2 = env.h * qt * m2 * m2 / (m2 * m2 - beta * x * x)
    return 2 * env.h * env.r * k * m2 - 2 * beta * th2 * x * Gc[0]

def exact_theta(k):
    Krow = np.zeros((1, env.nx)); Krow[0, 0] = k
    Fk = env.F - env.G @ Krow
    Sig = solve_discrete_lyapunov(Fk, SE * env.G @ env.G.T + env.Sd)
    m2, x = Sig[0, 0], (Fk @ Sig)[0, 0]
    qt = env.q + env.r * k * k
    th2 = env.h * qt * m2 * m2 / (m2 * m2 - beta * x * x)
    Ecost = env.h * (env.q * m2 + env.r * (k * k * m2 + SE))
    th0 = (Ecost + (beta - 1.0) * th2 * m2) / (1.0 - beta)
    return np.array([th0, 0.0, th2])

@njit(cache=True)
def chains(seed, steps, snap, F, Gc, Lw, L0, th_init, k_init,
           nseeds, se, h, q, r, beta, a_c, a_a):
    np.random.seed(seed)
    nx = F.shape[0]
    s = np.empty((nseeds, nx))
    for i in range(nseeds):                       # stationary init
        z = np.random.standard_normal(nx)
        for a in range(nx):
            acc = 0.0
            for b in range(nx): acc += L0[a, b] * z[b]
            s[i, a] = acc
    k = np.full(nseeds, k_init)
    th = np.empty((nseeds, 3))
    for i in range(nseeds):
        for j in range(3): th[i, j] = th_init[j]
    nsnap = steps // snap
    out = np.empty((nsnap, nseeds))
    sq = np.sqrt(se)
    for t in range(steps):
        for i in range(nseeds):
            v = s[i, 0]
            eta = sq * np.random.standard_normal()
            u = -k[i] * v + eta
            cost = h * (q * v * v + r * u * u)
            # dynamics: s' = F s + Gc*u + Lw z
            z0 = np.random.standard_normal()
            z1 = np.random.standard_normal()
            n0 = Lw[0, 0] * z0 + Lw[0, 1] * z1
            n1 = Lw[1, 0] * z0 + Lw[1, 1] * z1
            s0 = F[0, 0] * s[i, 0] + F[0, 1] * s[i, 1] + Gc[0] * u + n0
            s1 = F[1, 0] * s[i, 0] + F[1, 1] * s[i, 1] + Gc[1] * u + n1
            vn = s0
            V  = th[i, 0] + th[i, 1] * v + th[i, 2] * v * v
            Vn = th[i, 0] + th[i, 1] * vn + th[i, 2] * vn * vn
            delta = cost + beta * Vn - V
            th[i, 0] += a_c * delta
            th[i, 1] += a_c * delta * v
            th[i, 2] += a_c * delta * v * v
            g = delta * (-eta * v / se)
            kk = k[i] - a_a * g
            if kk < 0.02: kk = 0.02
            if kk > 35.0: kk = 35.0
            k[i] = kk
            s[i, 0] = s0; s[i, 1] = s1
        if (t + 1) % snap == 0:
            for i in range(nseeds): out[(t + 1) // snap - 1, i] = k[i]
    return out

snaps = chains(20260723, STEPS, SNAP, F, Gc, Lw, L0,
               exact_theta(KSTAR), KSTAR, NSEEDS, SE, H, Q, R,
               beta, ALPHA_C, ALPHA_A)

# mean-field: dk/dn = -ALPHA_A * CAL * ghat(k), SNAP-coarse RK2
kmf, mf = KSTAR, []
for n in range(STEPS // SNAP):
    half = kmf - 0.5 * SNAP * ALPHA_A * CAL * ghat(kmf)
    kmf = float(np.clip(kmf - SNAP * ALPHA_A * CAL * ghat(half), 0.02, 35.0))
    mf.append(kmf)
mf = np.array(mf)

steps_ax = SNAP * np.arange(1, len(mf) + 1)
out = np.column_stack([steps_ax, snaps, mf])
hdr = "step," + ",".join(f"seed{i}" for i in range(NSEEDS)) + ",meanfield"
np.savetxt("drift_multiseed.csv", out, delimiter=",", header=hdr, comments="")

fin = snaps[-1]
print("final k after %.1fM steps: " % (STEPS / 1e6)
      + ", ".join("%.2f" % x for x in fin))
print("mean %.2f +- %.2f ; mean-field %.2f" % (fin.mean(), fin.std(), mf[-1]))
for frac, lab in ((0.25, "0.3M"), (0.5, "0.6M"), (0.75, "0.9M"), (1.0, "1.2M")):
    row = snaps[int(frac * len(snaps)) - 1]
    mfv = mf[int(frac * len(mf)) - 1]
    print("k @ %s: chains %.2f +- %.2f | mean-field %.2f"
          % (lab, row.mean(), row.std(), mfv))
