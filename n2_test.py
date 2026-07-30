"""
General-N test of the sampling-scale equilibrium law.

Conjecture: for a Prony kernel K(tau) = sum_i c_i exp(-gamma_i tau),
the aliased-AC equilibrium obeys
    k_eq * h = chi( s_e / (Theta * K(0)) ),     K(0) = sum_i c_i,
and J(k_eq) -> r * Theta * K(0)   (deployment, h->0),
i.e. the kernel enters ONLY through its zero moment K(0).

Uses the EXACT Propositions 1-2 (dimension-agnostic): only m2 = E[v^2],
x = E[v v'], g = G_v are needed, from an (N+1)-dim discrete Lyapunov solve.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov
from scipy.optimize import brentq

TH, Q, R, RHO = 1.0, 1.0, 1.0, 0.05

def model_N(cs, gs):
    N = len(cs); n = N + 1
    A = np.zeros((n, n)); A[0, 1:] = 1.0
    B = np.zeros((n, 1)); B[0, 0] = 1.0
    W = np.zeros((n, n))
    for i, (c, gm) in enumerate(zip(cs, gs)):
        A[1 + i, 0] = -c; A[1 + i, 1 + i] = -gm
        W[1 + i, 1 + i] = 2 * TH * gm * c
    return A, B, W

def disc(A, B, W, h, nsub=400):
    F = expm(A * h)
    G = np.zeros_like(B); Sd = np.zeros_like(W)
    dt = h / nsub
    for i in range(nsub):
        E = expm(A * ((i + 0.5) * dt))
        G += E @ B * dt; Sd += E @ W @ E.T * dt
    return F, G, Sd

def keq_exact(cs, gs, h, se):
    A, B, W = model_N(cs, gs)
    F, G, Sd = disc(A, B, W, h)
    beta = np.exp(-RHO * h); g0 = G[0, 0]; n = A.shape[0]
    def ghat(k):
        Krow = np.zeros((1, n)); Krow[0, 0] = k
        Fk = F - G @ Krow
        if np.max(np.abs(np.linalg.eigvals(Fk))) >= 1.0: return np.nan
        Sig = solve_discrete_lyapunov(Fk, se * G @ G.T + Sd)
        m2 = Sig[0, 0]; x = (Fk @ Sig)[0, 0]
        qt = Q + R * k * k
        th2 = h * qt * m2 * m2 / (m2 * m2 - beta * x * x)
        return 2 * h * R * k * m2 - 2 * beta * th2 * x * g0
    ks = np.geomspace(0.5, 1.95 / h, 50)
    vs = [ghat(k) for k in ks]
    for i in range(len(ks) - 1):
        if (np.isfinite(vs[i]) and np.isfinite(vs[i + 1])
                and np.sign(vs[i]) != np.sign(vs[i + 1])):
            return brentq(ghat, ks[i], ks[i + 1], xtol=1e-6)
    return np.nan

def chi(ratio):
    """root of M^2 - X^2 = kap X M with Theta*K0 normalized to 1."""
    def eq(kap):
        M = 1 / kap ** 2 + ratio / (kap * (2 - kap))
        X = (1 - kap) * M + 1 / kap
        return M * M - X * X - kap * X * M
    return brentq(eq, 0.02, 1.98)

def J_deploy(cs, gs, h, k):
    A, B, W = model_N(cs, gs)
    F, G, Sd = disc(A, B, W, h)
    n = A.shape[0]
    Krow = np.zeros((1, n)); Krow[0, 0] = k
    Fk = F - G @ Krow
    Sig = solve_discrete_lyapunov(Fk, Sd)
    return (Q + R * k * k) * Sig[0, 0]

if __name__ == "__main__":
    cases = [
        ("N=1 baseline     ", [0.3], [1.0]),
        ("N=2 split K0=0.3 ", [0.15, 0.15], [0.5, 2.0]),
        ("N=2 asym  K0=0.6 ", [0.2, 0.4], [0.5, 3.0]),
        ("N=3       K0=0.6 ", [0.1, 0.2, 0.3], [0.3, 1.0, 4.0]),
        ("N=2 wide  K0=0.6 ", [0.3, 0.3], [0.1, 10.0]),
    ]
    h = 0.01
    for se in [0.09, 0.3]:
        print(f"--- s_e={se}, h={h} ---")
        print(f"{'case':>18} {'K0':>5} {'k_eq':>9} {'chi/h pred':>10} "
              f"{'ratio':>6} {'J(k_eq)':>8} {'rThK0':>6}")
        for name, cs, gs in cases:
            K0 = sum(cs)
            ke = keq_exact(cs, gs, h, se)
            pred = chi(se / (TH * K0)) / h
            J = J_deploy(cs, gs, h, ke)
            print(f"{name:>18} {K0:5.2f} {ke:9.2f} {pred:10.2f} "
                  f"{ke/pred:6.3f} {J:8.4f} {R*TH*K0:6.3f}", flush=True)
