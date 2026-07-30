"""
Exact LSTD(lambda) / GAE(lambda) extension of the aliased-AC theory.

Critic: LSTD(lambda) fixed point over features phi = (1, v, v^2),
    A theta = b,  A = sum_j (bl)^j E[phi_{t-j} (phi_t - beta phi_{t+1})'],
                  b = sum_j (bl)^j E[phi_{t-j} c_t],      bl := beta*lambda.
Actor: expected GAE gradient
    ghat = E[score_t * sum_{j>=0} (bl)^j delta_{t+j}],  score_t = -v_t eta_t / s_e.

Lag moments are Gaussian and sum to RESOLVENTS (no truncation):
    x_j  = E[v_{t} v_{t+j}]      = e0' F^j  Sig e0
    y_j  = E[eta_t v_{t+j}]      = e0' F^{j-1} G s_e   (j>=1),  y_0 = 0
    sum (bl)^j x_j   = e0' (I - bl F)^{-1} Sig e0
    sum (bl)^j x_j^2 = (e0 ox e0)' (I - bl F ox F)^{-1} (Sig e0 ox Sig e0)
Verified vs truncated sums.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq
import theorem2_learning as T

E0 = None  # set per model size (N=1 here: dim 2)

def lag_objects(gamma, k, s_e):
    F, G, Sd, Fk, Sig = T.closed_loop(gamma, k, s_e)
    e0 = np.array([1.0, 0.0])
    g0 = G[:, 0]
    return Fk, Sig, e0, g0

def resolvent_sums(Fk, Sig, e0, g0, bl, s_e):
    """returns Sx = sum_{j>=1} bl^j x_j, Sxx = sum_{j>=1} bl^j x_j^2,
    (x_0 = m2 handled separately), and helpers."""
    n = len(e0)
    R1 = np.linalg.solve(np.eye(n) - bl * Fk.T, e0)        # (I-bl F')^-1 e0
    Sx = bl * (Fk @ Sig @ R1)[0] if False else None
    # cleaner: sum_{j>=1} bl^j F^j = bl F (I - bl F)^{-1}
    M1 = bl * np.linalg.solve(np.eye(n) - bl * Fk, Fk)
    Sx = e0 @ M1 @ Sig @ e0
    K2 = np.kron(Fk, Fk)
    M2 = bl * np.linalg.solve(np.eye(n * n) - bl * K2, K2)
    v2 = np.kron(Sig @ e0, Sig @ e0)
    Sxx = np.kron(e0, e0) @ M2 @ v2
    return Sx, Sxx

def lstd_lambda_theta2(gamma, k, s_e, lam, beta=None):
    """exact LSTD(lambda) quadratic-feature coefficient theta2."""
    if beta is None: beta = T.BETA
    h, q, r = T.H, T.Q, T.R
    bl = beta * lam
    Fk, Sig, e0, g0 = lag_objects(gamma, k, s_e)
    m2 = Sig[0, 0]
    qt = q + r * k * k
    n = len(e0)
    # x_j for j>=0 ; need sums with x_j (into c and phi_t) and x_{j+1} (phi_{t+1})
    # centered v^2-row of A theta = b  (theta1=0 by parity, verified):
    #   sum_j bl^j [ 2 qt h x_j^2  + theta2 ( 2 beta x_{j+1}^2 - 2 x_j^2 ) ] ... sign:
    #   E[(v_{t-j}^2 - m2)(c_t)] = 2 h qt x_j^2
    #   E[(v_{t-j}^2 - m2)(v_t^2 - beta v_{t+1}^2)] = 2 x_j^2 - 2 beta x_{j+1}^2
    # theta2 = [sum_j bl^j h qt x_j^2] / [sum_j bl^j (x_j^2 - beta x_{j+1}^2)]
    x0 = m2
    Sx, Sxx = resolvent_sums(Fk, Sig, e0, g0, bl, s_e)      # j>=1 sums
    Sxx0 = x0 * x0 + Sxx                                     # sum_{j>=0} bl^j x_j^2
    # sum_{j>=0} bl^j x_{j+1}^2 = (1/bl)(Sxx)  [since j>=1 terms of x_j^2 with shift]
    if bl > 0:
        Sxx1 = Sxx / bl
    else:
        x1 = (Fk @ Sig)[0, 0]
        Sxx1 = x1 * x1
    num = T.H * qt * Sxx0
    den = Sxx0 - beta * Sxx1
    return num / den

def gae_gradient(gamma, k, s_e, lam, beta=None):
    """exact expected GAE policy-gradient estimate for u = -k v + eta."""
    if beta is None: beta = T.BETA
    h, q, r = T.H, T.Q, T.R
    bl = beta * lam
    Fk, Sig, e0, g0 = lag_objects(gamma, k, s_e)
    m2 = Sig[0, 0]
    th2 = lstd_lambda_theta2(gamma, k, s_e, lam, beta)
    n = len(e0)
    # E[score_t delta_{t+j}] terms (score = -v_t eta_t / s_e):
    # j = 0: 2 r k h m2 - 2 beta th2 x1 g0v      (Prop 2)
    # j >= 1: delta_{t+j} pieces correlated with (v_t, eta_t):
    #   E[v_t eta_t v_{t+j}^2]       = 2 x_j y_j
    #   E[v_t eta_t v_{t+j+1}^2]     = 2 x_{j+1} y_{j+1}
    #   E[v_t eta_t c_{t+j}]         = h qt' 2 x_j y_j with qt'=q + r k^2 (the -2rk v eta
    #                                  term needs eta_{t+j}: independent -> 0)
    #   => E[score delta_{t+j}] = -(1/s_e)[ 2 h (q + r k^2) x_j y_j
    #                             + th2( 2 beta x_{j+1} y_{j+1} - 2 x_j y_j ) ]
    # with y_j = (F^{j-1} G)_v s_e  for j>=1.
    g0v = g0[0]
    x1 = (Fk @ Sig)[0, 0]
    qt = q + r * k * k
    gj0 = 2 * r * k * h * m2 - 2 * beta * th2 * x1 * g0v
    # resolvent for sum_{j>=1} bl^j x_j y_j:
    #   x_j = e0' F^j Sig e0,  y_j = s_e e0' F^{j-1} G
    #   x_j y_j = s_e (e0 ox e0)' (F ox F)^{j-1} (F Sig e0 ox G)
    K2 = np.kron(Fk, Fk)
    ee = np.kron(e0, e0)
    w = np.kron(Fk @ Sig @ e0, g0)
    R = np.linalg.solve(np.eye(n * n) - bl * K2, np.eye(n * n))
    Sxy = bl * s_e * ee @ R @ w                      # sum_{j>=1} bl^j x_j y_j
    # sum_{j>=1} bl^j x_{j+1} y_{j+1} = (1/bl) sum_{j>=2} ... = (1/bl)(Sxy_all - bl*x1y1)
    x1y1 = s_e * (ee @ w)
    if bl > 0:
        Sxy1 = (Sxy - bl * x1y1) / bl + (ee @ (K2 @ R) @ w) * 0  # simplify below
        Sxy1 = (Sxy / bl) - x1y1 + bl * s_e * ee @ R @ (K2 @ w) * 0
        # direct: sum_{j>=1} bl^j x_{j+1} y_{j+1} ; x_{j+1}y_{j+1} = s_e (ee)'(FoxF)^j (F Sig e0 ox G)
        Sxy1 = s_e * ee @ (bl * K2 @ np.linalg.solve(np.eye(n*n) - bl*K2, w))
    else:
        Sxy1 = 0.0
    tail = -(1.0 / s_e) * (2 * h * qt * Sxy
                           + th2 * (2 * beta * Sxy1 - 2 * Sxy))
    return gj0 + tail

def k_eq_lambda(gamma, s_e, lam, kmax=None):
    kmax = kmax or 1.9 / T.H
    ks = np.geomspace(0.3, kmax, 40)
    vs = [gae_gradient(gamma, k, s_e, lam) for k in ks]
    roots = []
    for i in range(len(ks) - 1):
        if np.sign(vs[i]) != np.sign(vs[i + 1]):
            roots.append(brentq(lambda k: gae_gradient(gamma, k, s_e, lam),
                                ks[i], ks[i + 1], xtol=1e-5))
    return roots

if __name__ == "__main__":
    g = 1.0
    # --- validation: lambda = 0 must reproduce Prop 1-3 exactly ---
    st = T.one_step_setup(g, 1.0)
    th_ref, _, _ = T.td0_critic(st, full=False)
    th0 = lstd_lambda_theta2(g, 1.0, T.S_E, 0.0)
    g_ref = T.ac_gradient(g, 1.0, full=False)
    g0 = gae_gradient(g, 1.0, T.S_E, 0.0)
    print(f"validation lam=0: theta2 {th0:.5f} vs {th_ref[2]:.5f} | "
          f"ghat {g0:.6f} vs {g_ref:.6f}")
    # --- brute-force check of resolvent sums at lam=0.9, k=5 ---
    lam = 0.9; k = 5.0; bl = T.BETA * lam
    Fk, Sig, e0, g0v = lag_objects(g, k, T.S_E)
    xs = [ (np.linalg.matrix_power(Fk, j) @ Sig)[0,0] for j in range(400) ]
    brute = sum(bl**j * xs[j]**2 for j in range(1, 400))
    _, Sxx = resolvent_sums(Fk, Sig, e0, g0v, bl, T.S_E)
    print(f"validation resolvent: Sxx {Sxx:.6e} vs brute {brute:.6e}")
    # --- the master-knob curve ---
    print(f"\nk_eq(lambda), gamma=1, h={T.H}, s_e={T.S_E} "
          f"(k*_ml=1.649, TD0 k_eq=24.79):")
    for lam in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]:
        roots = k_eq_lambda(g, T.S_E, lam)
        rs = ", ".join(f"{r:.2f}" for r in roots) if roots else "none in range"
        print(f"  lam={lam:5.3f}: equilibria at k = {rs}", flush=True)
