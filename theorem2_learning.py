"""
Learning-channel damage from partial observability, N=1 GLE model.
Everything exact: linear-Gaussian closed loop + Wick (Isserlis) moments.

Channels measured, per gamma:
  (1) Actor-critic equilibrium with memoryless TD(0) critic vs best
      memoryless policy: amplification ratio
          (J(k_AC^ml) - J(k*_ml)) / (J(k*_ml) - J*)
      = learning damage in units of the representational gap.
      Control: same actor-critic with a FULL-STATE quadratic critic
      (isolates aliasing from discounting-mismatch bias).
  (2) TD-error autocorrelation (integrated autocorrelation time):
      memoryless critic (colored) vs full critic (martingale -> white).

Setup: exact ZOH discretization (step h), policy u = -k v + eta,
eta ~ N(0, s_e). Cost rate q v^2 + r u^2. Discount beta = exp(-rho h).
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov, solve_discrete_are
from scipy.optimize import minimize_scalar, brentq

H = 0.05
Q, R, TH, CC = 1.0, 1.0, 1.0, 0.3
RHO = 0.05
BETA = np.exp(-RHO * H)
S_E = 0.3 ** 2      # exploration noise variance

# ---------------- discrete model (exact ZOH) ----------------
_DM_CACHE = {}
def disc_model(g):
    if g in _DM_CACHE: return _DM_CACHE[g]
    A = np.array([[0.0, 1.0], [-CC, -g]])
    B = np.array([[1.0], [0.0]])
    W = np.diag([0.0, 2 * TH * g * CC])
    F = expm(A * H)
    # G = int_0^h e^{A tau} B dtau ; Sd = int_0^h e^{A tau} W e^{A' tau} dtau
    nsub = 200; dt = H / nsub
    G = np.zeros((2, 1)); Sd = np.zeros((2, 2))
    for i in range(nsub):
        t = (i + 0.5) * dt
        E = expm(A * t)
        G += E @ B * dt
        Sd += E @ W @ E.T * dt
    _DM_CACHE[g] = (F, G, Sd)
    return F, G, Sd

# ---------------- Wick machinery ----------------
def wick(forms, cov):
    n = len(forms)
    if n == 0: return 1.0
    if n % 2 == 1: return 0.0
    if n == 2: return float(forms[0] @ cov @ forms[1])
    tot = 0.0
    f0 = forms[0]
    for j in range(1, n):
        c = float(f0 @ cov @ forms[j])
        if c != 0.0:
            rest = forms[1:j] + forms[j + 1:]
            tot += c * wick(rest, cov)
    return tot

class Poly:
    """polynomial in mean-zero jointly Gaussian linear forms"""
    def __init__(self, terms=None):
        self.t = terms or []          # list of (coef, tuple_of_form_arrays)
    @staticmethod
    def const(c): return Poly([(c, ())])
    @staticmethod
    def lin(a): return Poly([(1.0, (np.asarray(a, float),))])
    def __add__(s, o): return Poly(s.t + o.t)
    def __sub__(s, o): return s + o * (-1.0)
    def __mul__(s, o):
        if isinstance(o, (int, float)):
            return Poly([(c * o, f) for c, f in s.t])
        return Poly([(c1 * c2, f1 + f2) for c1, f1 in s.t for c2, f2 in o.t])
    __rmul__ = __mul__
    def E(self, cov):
        return sum(c * wick(list(f), cov) for c, f in self.t)

# ---------------- per-k objects ----------------
def closed_loop(g, k, s_e=S_E):
    F, G, Sd = disc_model(g)
    Fk = F - G @ np.array([[k, 0.0]])
    Sig = solve_discrete_lyapunov(Fk, G @ G.T * s_e + Sd)
    return F, G, Sd, Fk, Sig

def one_step_setup(g, k, s_e=S_E):
    """X = (s(2), eta, w(2)); returns linear forms and cov."""
    F, G, Sd, Fk, Sig = closed_loop(g, k, s_e)
    d = 5
    cov = np.zeros((d, d))
    cov[:2, :2] = Sig; cov[2, 2] = s_e; cov[3:, 3:] = Sd
    def e(i):
        a = np.zeros(d); a[i] = 1.0; return a
    v = Poly.lin(e(0)); z = Poly.lin(e(1)); eta = Poly.lin(e(2))
    # s' = Fk s + G eta + w
    row = lambda i: Poly.lin(np.concatenate([Fk[i, :], [G[i, 0]], np.eye(2)[i]]))
    vp, zp = row(0), row(1)
    # u = -k v + eta ; cost per step = H*(q v^2 + r u^2)
    u = eta - k * v
    c = (Q * (v * v) + R * (u * u)) * H
    return dict(cov=cov, v=v, z=z, eta=eta, vp=vp, zp=zp, c=c,
                Fk=Fk, Sig=Sig, G=G, Sd=Sd)

def td0_critic(setup, full):
    """LSTD/TD(0) fixed point. features: ml (1,v,v^2); full adds z,vz,z^2."""
    v, z, vp, zp, c, cov = (setup[x] for x in ["v", "z", "vp", "zp", "c", "cov"])
    one = Poly.const(1.0)
    if full:
        phi  = [one, v, z, v * v, v * z, z * z]
        phip = [one, vp, zp, vp * vp, vp * zp, zp * zp]
    else:
        phi  = [one, v, v * v]
        phip = [one, vp, vp * vp]
    m = len(phi)
    Amat = np.array([[(phi[i] * (phi[j] - BETA * phip[j])).E(cov)
                      for j in range(m)] for i in range(m)])
    b = np.array([(phi[i] * c).E(cov) for i in range(m)])
    th = np.linalg.solve(Amat, b)
    return th, phi, phip

def ac_gradient(g, k, full, s_e=S_E):
    """E[ score * (c + beta*Vhat(o') - Vhat(o)) ], critic = TD0 fixed point."""
    st = one_step_setup(g, k, s_e)
    th, phi, phip = td0_critic(st, full)
    Vh  = sum((Poly.const(0.0),) + tuple(th[i] * phi[i]  for i in range(len(th))), Poly.const(0.0))
    Vhp = sum((Poly.const(0.0),) + tuple(th[i] * phip[i] for i in range(len(th))), Poly.const(0.0))
    delta = st["c"] + BETA * Vhp - Vh
    score = (st["v"] * st["eta"]) * (-1.0 / s_e)
    return (score * delta).E(st["cov"])

def J_deploy(g, k):
    """average cost rate, no exploration noise."""
    F, G, Sd, Fk, Sig = closed_loop(g, k, s_e=0.0)
    return (Q + R * k * k) * Sig[0, 0]

def J_opt(g):
    F, G, Sd = disc_model(g)
    P = solve_discrete_are(F, G, np.diag([Q, 0.0]) * H, np.array([[R * H]]))
    return np.trace(P @ Sd) / H

def best_ml(g):
    res = minimize_scalar(lambda k: J_deploy(g, k), bounds=(0.05, 5), method="bounded")
    return res.x, res.fun

def ac_equilibrium(g, full, klo=0.1, khi=4.0):
    grid = np.linspace(klo, khi, 24)
    vals = [ac_gradient(g, k, full) for k in grid]
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            return brentq(lambda k: ac_gradient(g, k, full), grid[i], grid[i + 1],
                          xtol=1e-6)
    return np.nan

# ---------------- TD-error autocorrelation ----------------
def td_autocorr(g, k, full, jmax=60, s_e=S_E):
    F, G, Sd, Fk, Sig = closed_loop(g, k, s_e)
    st = one_step_setup(g, k, s_e)
    th, _, _ = td0_critic(st, full)
    def Vh_of(vP, zP):
        one = Poly.const(1.0)
        if full:
            ph = [one, vP, zP, vP * vP, vP * zP, zP * zP]
        else:
            ph = [one, vP, vP * vP]
        return sum((th[i] * ph[i] for i in range(len(th))), Poly.const(0.0))
    # X = (s_t, eta_t, w_t, zeta(2), eta', w')  dim 10
    d = 10
    def e(i):
        a = np.zeros(d); a[i] = 1.0; return a
    rows = []
    out = []
    for j in range(1, jmax + 1):
        cov = np.zeros((d, d))
        cov[:2, :2] = Sig; cov[2, 2] = s_e; cov[3:5, 3:5] = Sd
        Fj1 = np.linalg.matrix_power(Fk, j - 1)
        Sz = Sig - Fj1 @ Sig @ Fj1.T
        cov[5:7, 5:7] = Sz + 1e-14 * np.eye(2)
        cov[7, 7] = s_e; cov[8:, 8:] = Sd
        L = lambda a: Poly.lin(np.asarray(a, float))
        s_t = [e(0), e(1)]
        v, z, eta = L(e(0)), L(e(1)), L(e(2))
        # s_{t+1} = Fk s + G eta + w
        s1 = [np.concatenate([Fk[i, :], [G[i, 0]], np.eye(2)[i], np.zeros(5)])
              for i in range(2)]
        # s_{t+j} = Fj1 s_{t+1} + zeta
        sj = [Fj1[i, 0] * s1[0] + Fj1[i, 1] * s1[1] + e(5 + i) for i in range(2)]
        # s_{t+j+1} = Fk s_{t+j} + G eta' + w'
        sj1 = [Fk[i, 0] * sj[0] + Fk[i, 1] * sj[1] + G[i, 0] * e(7) + e(8 + i)
               for i in range(2)]
        def delta_of(vP, zP, etaP, vpP, zpP):
            u = etaP - k * vP
            c = (Q * (vP * vP) + R * (u * u)) * H
            return c + BETA * Vh_of(vpP, zpP) - Vh_of(vP, zP)
        d0 = delta_of(v, z, eta, L(s1[0]), L(s1[1]))
        dj = delta_of(L(sj[0]), L(sj[1]), L(e(7)), L(sj1[0]), L(sj1[1]))
        m0 = d0.E(cov); mj = dj.E(cov)
        var0 = (d0 * d0).E(cov) - m0 ** 2
        cvj = (d0 * dj).E(cov) - m0 * mj
        out.append(cvj / var0)
    return np.array(out)

if __name__ == "__main__":
    print(f"h={H} beta={BETA:.5f} rho={RHO} s_e={S_E} c={CC}")
    hdr = (f"{'γ':>4} {'J*':>8} {'k*_ml':>7} {'J(k*)':>8} {'repgap%':>8} "
           f"{'k_AC^full':>9} {'k_AC^ml':>8} {'J(k_AC)':>8} {'learngap%':>9} "
           f"{'AMP':>6} {'IACT_ml':>8} {'IACT_fu':>8}")
    print(hdr)
    for g in [0.5, 1.0, 2.0]:
        Js = J_opt(g)
        km, Jm = best_ml(g)
        kf = ac_equilibrium(g, full=True)
        kl = ac_equilibrium(g, full=False)
        Jl = J_deploy(g, kl) if np.isfinite(kl) else np.nan
        rep = (Jm - Js) / Js
        lrn = (Jl - Jm) / Js
        amp = (Jl - Jm) / (Jm - Js)
        rho_ml = td_autocorr(g, kl if np.isfinite(kl) else km, full=False)
        rho_fu = td_autocorr(g, kl if np.isfinite(kl) else km, full=True)
        iact_ml = 1 + 2 * rho_ml.sum()
        iact_fu = 1 + 2 * rho_fu.sum()
        print(f"{g:4.1f} {Js:8.4f} {km:7.3f} {Jm:8.4f} {100*rep:8.2f} "
              f"{kf:9.3f} {kl:8.3f} {Jl:8.4f} {100*lrn:9.2f} "
              f"{amp:6.2f} {iact_ml:8.2f} {iact_fu:8.2f}", flush=True)
