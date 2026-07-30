"""
Restoration theorem: the AC equilibrium as a function of belief quality.

Critic receives a NOISY belief feature zh = z + eps, eps ~ N(0, Rb) i.i.d.
(Rb = belief error variance; Rb=0 is the asymmetric/full-state critic,
Rb -> inf is the memoryless critic). The ACTOR stays memoryless
(u = -k v + eta) throughout — restoration happens purely through the critic.

Critic features: (1, v, zh, v^2, v*zh, zh^2). Exact Wick-moment LSTD fixed
point + exact expected AC gradient, as in theorem2_learning.py.
Output: k_eq(Rb) and deployed cost J(k_eq(Rb)).
"""
import numpy as np
from scipy.optimize import brentq
import theorem2_learning as T

def setup7(gamma, k, s_e, Rb):
    """X = (v, z, eta, w_v, w_z, eps, eps'); returns forms dict + cov."""
    F, G, Sd, Fk, Sig = T.closed_loop(gamma, k, s_e)
    d = 7
    cov = np.zeros((d, d))
    cov[:2, :2] = Sig
    cov[2, 2] = s_e
    cov[3:5, 3:5] = Sd
    cov[5, 5] = Rb
    cov[6, 6] = Rb
    def e(i):
        a = np.zeros(d); a[i] = 1.0; return a
    L = T.Poly.lin
    v, z, eta = L(e(0)), L(e(1)), L(e(2))
    vp = L(np.array([Fk[0, 0], Fk[0, 1], G[0, 0], 1, 0, 0, 0], float))
    zp = L(np.array([Fk[1, 0], Fk[1, 1], G[1, 0], 0, 1, 0, 0], float))
    zh = z + L(e(5))
    zhp = zp + L(e(6))
    u = eta - k * v
    c = (T.Q * (v * v) + T.R * (u * u)) * T.H
    return dict(cov=cov, v=v, z=z, eta=eta, vp=vp, zp=zp, zh=zh, zhp=zhp, c=c)

def critic_noisy(st):
    one = T.Poly.const(1.0)
    v, zh, vp, zhp, c, cov = (st[x] for x in ["v", "zh", "vp", "zhp", "c", "cov"])
    phi  = [one, v, zh, v * v, v * zh, zh * zh]
    phip = [one, vp, zhp, vp * vp, vp * zhp, zhp * zhp]
    m = len(phi)
    A = np.array([[(phi[i] * (phi[j] - T.BETA * phip[j])).E(cov)
                   for j in range(m)] for i in range(m)])
    b = np.array([(phi[i] * c).E(cov) for i in range(m)])
    th = np.linalg.solve(A, b)
    return th, phi, phip

def ac_grad_noisy(gamma, k, s_e, Rb):
    st = setup7(gamma, k, s_e, Rb)
    th, phi, phip = critic_noisy(st)
    Vh  = sum((th[i] * phi[i]  for i in range(6)), T.Poly.const(0.0))
    Vhp = sum((th[i] * phip[i] for i in range(6)), T.Poly.const(0.0))
    delta = st["c"] + T.BETA * Vhp - Vh
    score = (st["v"] * st["eta"]) * (-1.0 / s_e)
    return (score * delta).E(st["cov"])

def k_eq_noisy(gamma, s_e, Rb, kmax=None):
    kmax = kmax or 1.9 / T.H
    ks = np.geomspace(0.3, kmax, 26)
    vs = [ac_grad_noisy(gamma, k, s_e, Rb) for k in ks]
    for i in range(len(ks) - 1):
        if np.sign(vs[i]) != np.sign(vs[i + 1]):
            return brentq(lambda k: ac_grad_noisy(gamma, k, s_e, Rb),
                          ks[i], ks[i + 1], xtol=1e-5)
    return np.nan

if __name__ == "__main__":
    g = 1.0
    km, Jm = T.best_ml(g)
    Js = T.J_opt(g)
    Szz = T.TH * T.CC     # stationary z variance = belief-error scale
    print(f"γ=1, h={T.H}, s_e={T.S_E}, ρ=0.05. k*_ml={km:.3f}, "
          f"J(k*)={Jm:.4f}, J*={Js:.4f}, rΘc={T.R*T.TH*T.CC:.3f}, Σzz={Szz:.2f}")
    print(f"{'Rb/Σzz':>8} {'k_eq':>8} {'J(k_eq)':>8} {'excess/J*%':>10}")
    for ratio in [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1e4]:
        Rb = ratio * Szz
        ke = k_eq_noisy(g, T.S_E, Rb)
        J = T.J_deploy(g, ke) if np.isfinite(ke) else np.nan
        print(f"{ratio:8.2f} {ke:8.3f} {J:8.4f} {100*(J-Js)/Js:10.2f}", flush=True)
