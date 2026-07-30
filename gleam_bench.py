"""
GLEAM-bench: exactly solvable POMDP family for studying learning under
partial observability.

Environment: v is the observed coordinate of a generalized Langevin system
with Prony memory kernel K(tau) = sum_i c_i exp(-gamma_i tau):
    dv = (sum_i z_i + u) dt
    dz_i = (-gamma_i z_i - c_i v) dt + sigma_i dW_i,  sigma_i^2 = 2 Theta gamma_i c_i
Observation: v only. Cost rate q v^2 + r u^2. Exact ZOH discretization at
step h. Every instance ships with baselines:
    J*        : optimal (belief-state) cost rate
    k*_ml     : best memoryless linear gain, J(k*_ml)
    k_AC, J_AC: predicted memoryless actor-critic equilibrium (exact Prop 1-3
                formulas) -- where a TD-critic memoryless learner plateaus
    rThK0     : r*Theta*K(0), the h->0 plateau law
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov, solve_discrete_are
from scipy.optimize import minimize_scalar, brentq


class GLEBench:
    def __init__(self, cs=(0.3,), gammas=(1.0,), theta=1.0, h=0.05,
                 q=1.0, r=1.0, rho=0.05, n_envs=64, seed=0, nl=0.0):
        self.nl = nl                       # cubic damping on v (nonlinear variant)
        self.cs, self.gammas = np.array(cs, float), np.array(gammas, float)
        self.theta, self.h, self.q, self.r, self.rho = theta, h, q, r, rho
        N = len(cs); self.N = N; self.nx = N + 1
        A = np.zeros((self.nx, self.nx)); A[0, 1:] = 1.0
        W = np.zeros((self.nx, self.nx))
        for i in range(N):
            A[1 + i, 0] = -self.cs[i]; A[1 + i, 1 + i] = -self.gammas[i]
            W[1 + i, 1 + i] = 2 * theta * self.gammas[i] * self.cs[i]
        B = np.zeros((self.nx, 1)); B[0, 0] = 1.0
        self.A, self.B, self.W = A, B, W
        self.F = expm(A * h)
        nsub = 400; dt = h / nsub
        G = np.zeros((self.nx, 1)); Sd = np.zeros((self.nx, self.nx))
        for i in range(nsub):
            E = expm(A * ((i + 0.5) * dt))
            G += E @ B * dt; Sd += E @ W @ E.T * dt
        self.G, self.Sd = G, Sd
        self.Lw = np.linalg.cholesky(Sd + 1e-14 * np.eye(self.nx))
        # stationary (uncontrolled) covariance for resets
        self.Sig0 = solve_discrete_lyapunov(self.F, Sd)
        self.L0 = np.linalg.cholesky(self.Sig0 + 1e-14 * np.eye(self.nx))
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)
        self.s = None

    # ---- vectorized gym-like API (continuing task) ----
    def reset(self):
        self.s = (self.L0 @ self.rng.standard_normal((self.nx, self.n_envs))).T
        return self.s[:, 0:1].copy()          # obs = v

    def step(self, u):
        """u: (n_envs,) or (n_envs,1). Returns obs, cost (per step)."""
        u = np.asarray(u, float).reshape(-1)
        cost = self.h * (self.q * self.s[:, 0] ** 2 + self.r * u ** 2)
        noise = (self.Lw @ self.rng.standard_normal((self.nx, self.n_envs))).T
        self.s = self.s @ self.F.T + np.outer(u, self.G[:, 0]) + noise
        if self.nl:                        # symmetric split for cubic drag
            self.s[:, 0] -= self.h * self.nl * self.s[:, 0] ** 3
        return self.s[:, 0:1].copy(), cost

    def privileged(self):
        """hidden bath total (for asymmetric-critic experiments)."""
        return self.s[:, 1:].sum(axis=1, keepdims=True).copy()

    # ---- exact baselines ----
    def K0(self): return float(self.cs.sum())

    def J_opt(self):
        P = solve_discrete_are(self.F, self.G,
                               np.diag([self.q] + [0.0] * self.N) * self.h,
                               np.array([[self.r * self.h]]))
        return float(np.trace(P @ self.Sd)) / self.h

    def J_gain(self, k):
        Krow = np.zeros((1, self.nx)); Krow[0, 0] = k
        Fk = self.F - self.G @ Krow
        if np.max(np.abs(np.linalg.eigvals(Fk))) >= 1: return np.inf
        Sig = solve_discrete_lyapunov(Fk, self.Sd)
        return float((self.q + self.r * k * k) * Sig[0, 0])

    def best_memoryless(self):
        res = minimize_scalar(self.J_gain, bounds=(0.02, 1.9 / self.h),
                              method="bounded")
        return float(res.x), float(res.fun)

    def ac_equilibrium(self, s_e=0.09):
        """Exact Prop 1-3: predicted plateau of a memoryless TD-critic AC."""
        beta = np.exp(-self.rho * self.h); g0 = self.G[0, 0]
        def ghat(k):
            Krow = np.zeros((1, self.nx)); Krow[0, 0] = k
            Fk = self.F - self.G @ Krow
            if np.max(np.abs(np.linalg.eigvals(Fk))) >= 1: return np.nan
            Sig = solve_discrete_lyapunov(Fk, s_e * self.G @ self.G.T + self.Sd)
            m2 = Sig[0, 0]; x = (Fk @ Sig)[0, 0]
            qt = self.q + self.r * k * k
            th2 = self.h * qt * m2 * m2 / (m2 * m2 - beta * x * x)
            return 2 * self.h * self.r * k * m2 - 2 * beta * th2 * x * g0
        ks = np.geomspace(0.3, 1.9 / self.h, 40)
        vs = [ghat(k) for k in ks]
        for i in range(len(ks) - 1):
            if (np.isfinite(vs[i]) and np.isfinite(vs[i + 1])
                    and np.sign(vs[i]) != np.sign(vs[i + 1])):
                keq = brentq(ghat, ks[i], ks[i + 1], xtol=1e-5)
                return keq, self.J_gain(keq)
        return np.nan, np.nan

    def baselines(self, s_e=0.09):
        km, Jm = self.best_memoryless()
        ka, Ja = self.ac_equilibrium(s_e)
        return dict(J_opt=self.J_opt(), k_ml=km, J_ml=Jm,
                    k_AC=ka, J_AC=Ja, rThK0=self.r * self.theta * self.K0())


if __name__ == "__main__":
    env = GLEBench()
    b = env.baselines()
    print("default instance (c=0.3, gamma=1, h=0.05):")
    for k, v in b.items(): print(f"  {k:8s} = {v:.4f}")
