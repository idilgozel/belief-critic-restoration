"""
MLP actor + MLP critic, GAE(lambda),
clipped surrogate, Adam. Chunked training with npz checkpoints.

Configs (obs -> agent input):
  ml    : current v only (memoryless)
  stack : last STACK observations (memory via frame stack)
GAE lambda is a config knob: lam<1 => bootstrapped critic (theory: phantom
gradient, plateau at J_AC ~ rThK0); lam=1 => Monte-Carlo advantages (theory:
no bootstrap aliasing bias => near best-memoryless).
"""
import numpy as np, os, sys, json
from gleam_bench import GLEBench

import os as _os
STACK = int(_os.environ.get("GLEAM_STACK", "8"))
CFG = json.loads(_os.environ.get("GLEAM_CFG", "{}"))     # GLEBench kwargs
ENT = float(_os.environ.get("GLEAM_ENT", "0"))           # entropy coef
LOGSTD0 = float(_os.environ.get("GLEAM_LOGSTD0", "-1.2"))
FIXSTD = bool(int(_os.environ.get("GLEAM_FIXSTD", "0")))
HID = 32

# ---------------- tiny MLP with manual backprop ----------------
def init_mlp(din, dout, rng, scale_out=0.01):
    p = {}
    p["W1"] = rng.standard_normal((din, HID)) / np.sqrt(din)
    p["b1"] = np.zeros(HID)
    p["W2"] = rng.standard_normal((HID, HID)) / np.sqrt(HID)
    p["b2"] = np.zeros(HID)
    p["W3"] = rng.standard_normal((HID, dout)) * scale_out
    p["b3"] = np.zeros(dout)
    return p

def mlp_fwd(p, x):
    h1 = np.tanh(x @ p["W1"] + p["b1"])
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    return h2 @ p["W3"] + p["b3"], (x, h1, h2)

def mlp_bwd(p, cache, dout):
    x, h1, h2 = cache
    g = {}
    g["W3"] = h2.T @ dout; g["b3"] = dout.sum(0)
    dh2 = (dout @ p["W3"].T) * (1 - h2 ** 2)
    g["W2"] = h1.T @ dh2; g["b2"] = dh2.sum(0)
    dh1 = (dh2 @ p["W2"].T) * (1 - h1 ** 2)
    g["W1"] = x.T @ dh1; g["b1"] = dh1.sum(0)
    return g

class Adam:
    def __init__(self, params, lr):
        self.lr = lr; self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

# ---------------- PPO ----------------
class PPO:
    def __init__(self, obs_dim, seed, lr=3e-4, ent_coef=0.0, log_std0=-1.2,
                 vf_dim=None):
        rng = np.random.default_rng(seed)
        self.pi = init_mlp(obs_dim, 1, rng)
        self.pi["log_std"] = np.array([log_std0])
        self.vf = init_mlp(vf_dim or obs_dim, 1, rng, scale_out=0.1)
        self.opt_pi = Adam(self.pi, lr)
        self.opt_vf = Adam(self.vf, lr)
        self.ent = ent_coef
        self.rng = rng

    def act(self, obs):
        mu, _ = mlp_fwd(self.pi, obs)
        std = np.exp(self.pi["log_std"])
        a = mu[:, 0] + std * self.rng.standard_normal(mu.shape[0])
        logp = -0.5 * ((a - mu[:, 0]) / std) ** 2 - np.log(std) - 0.9189385
        return a, logp

    def value(self, obs):
        v, _ = mlp_fwd(self.vf, obs)
        return v[:, 0]

    def update(self, obs, act, logp_old, adv, ret, epochs=6, nmb=4, clip=0.2,
               obs_vf=None):
        if obs_vf is None: obs_vf = obs
        n = len(act); idx = np.arange(n)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            self.rng.shuffle(idx)
            for mb in np.array_split(idx, nmb):
                o, a, lo, ad, rt = obs[mb], act[mb], logp_old[mb], adv[mb], ret[mb]
                ov = obs_vf[mb]
                # policy
                mu, cache = mlp_fwd(self.pi, o)
                std = np.exp(self.pi["log_std"][0])
                zed = (a - mu[:, 0]) / std
                logp = -0.5 * zed ** 2 - np.log(std) - 0.9189385
                ratio = np.exp(logp - lo)
                un = ratio * ad
                cl = np.clip(ratio, 1 - clip, 1 + clip) * ad
                use = (un <= cl) | ((ratio > 1 - clip) & (ratio < 1 + clip))
                # d(-surr)/dlogp = -ratio*ad where unclipped active
                dlogp = np.where(use, -ratio * ad, 0.0) / len(mb)
                dmu = dlogp * (zed / std)
                dlogstd = np.sum(dlogp * (zed ** 2 - 1)) - self.ent * len(mb) / n
                g = mlp_bwd(self.pi, cache, dmu[:, None])
                g["log_std"] = np.array([0.0 if FIXSTD else dlogstd])
                self.opt_pi.step(self.pi, g)
                self.pi["log_std"][:] = np.clip(self.pi["log_std"], -3.5, 0.5)
                # value
                v, cachev = mlp_fwd(self.vf, ov)
                dv = 2 * (v[:, 0] - rt)[:, None] / len(mb)
                gv = mlp_bwd(self.vf, cachev, dv)
                self.opt_vf.step(self.vf, gv)

def make_obs(hist):
    return np.concatenate(hist, axis=1)

def train_chunk(tag, obs_mode, lam, iters, steps=256, seed=1):
    ck = f"ck_{tag}.npz"; log = f"log_{tag}.csv"
    env = GLEBench(n_envs=64, seed=seed + 100, **CFG)
    beta = np.exp(-env.rho * env.h)
    obs_dim = STACK if obs_mode == "stack" else 1
    # asymmetric-critic mode: actor sees v; critic sees (v, zh) with
    # zh = privileged z + noise (GLEAM_BNOISE = white|ar, GLEAM_RB = variance)
    asym = obs_mode == "asym"
    if asym: obs_dim = 1
    RB = float(_os.environ.get("GLEAM_RB", "0"))
    BN = _os.environ.get("GLEAM_BNOISE", "white")
    TAU = 1.0
    a_ar = np.exp(-env.h / TAU)
    agent = PPO(obs_dim, seed, ent_coef=ENT, log_std0=LOGSTD0,
                vf_dim=(2 if asym else None))
    it0 = 0
    if os.path.exists(ck):
        d = np.load(ck, allow_pickle=True)
        for k in agent.pi: agent.pi[k] = d["pi_" + k]
        for k in agent.vf: agent.vf[k] = d["vf_" + k]
        agent.opt_pi = Adam(agent.pi, 3e-4); agent.opt_vf = Adam(agent.vf, 3e-4)
        agent.opt_pi.t = int(d["t_pi"]); agent.opt_vf.t = int(d["t_vf"])
        it0 = int(d["it"])
    v0 = env.reset()
    hist = [v0.copy() for _ in range(STACK)]
    eps = np.sqrt(RB) * np.random.default_rng(seed + 7).standard_normal((64, 1))
    rng_n = np.random.default_rng(seed + 9)
    def critic_ob(vob):
        nonlocal eps
        if BN == "ar":
            eps = a_ar * eps + np.sqrt((1 - a_ar ** 2) * RB) * \
                  rng_n.standard_normal((64, 1))
        else:
            eps = np.sqrt(RB) * rng_n.standard_normal((64, 1))
        return np.concatenate([vob, env.privileged() + eps], axis=1)
    for it in range(it0, it0 + iters):
        vdim = 2 if asym else obs_dim
        OBS = np.empty((steps, 64, obs_dim)); ACT = np.empty((steps, 64))
        LOGP = np.empty((steps, 64)); CST = np.empty((steps, 64))
        VAL = np.empty((steps + 1, 64)); OBV = np.empty((steps, 64, vdim))
        for t in range(steps):
            ob = make_obs(hist) if obs_mode == "stack" else hist[-1]
            obv = critic_ob(hist[-1]) if asym else ob
            OBS[t] = ob; OBV[t] = obv
            a, lp = agent.act(ob)
            ACT[t], LOGP[t] = a, lp
            VAL[t] = agent.value(obv)
            vnew, cost = env.step(a)
            CST[t] = cost
            hist.pop(0); hist.append(vnew)
        ob = make_obs(hist) if obs_mode == "stack" else hist[-1]
        obv = critic_ob(hist[-1]) if asym else ob
        VAL[steps] = agent.value(obv)
        RW = -CST
        adv = np.zeros((steps, 64)); last = 0.0
        for t in range(steps - 1, -1, -1):
            delta = RW[t] + beta * VAL[t + 1] - VAL[t]
            last = delta + beta * lam * last
            adv[t] = last
        ret = adv + VAL[:steps]
        agent.update(OBS.reshape(-1, obs_dim), ACT.ravel(), LOGP.ravel(),
                     adv.ravel(), ret.ravel(),
                     obs_vf=OBV.reshape(-1, vdim) if asym else None)
        Jhat = CST.mean() / env.h
        with open(log, "a") as f:
            f.write(f"{it},{Jhat:.5f},{float(agent.pi['log_std'][0]):.3f}\n")
    np.savez(ck, it=it0 + iters, t_pi=agent.opt_pi.t, t_vf=agent.opt_vf.t,
             **{"pi_" + k: v for k, v in agent.pi.items()},
             **{"vf_" + k: v for k, v in agent.vf.items()})
    return it0 + iters

def evaluate(tag, obs_mode, seed=7, T=4000):
    ck = np.load(f"ck_{tag}.npz", allow_pickle=True)
    obs_dim = STACK if obs_mode == "stack" else 1
    agent = PPO(obs_dim, seed)
    for k in agent.pi: agent.pi[k] = ck["pi_" + k]
    env = GLEBench(n_envs=64, seed=seed, **CFG)
    v0 = env.reset(); hist = [v0.copy() for _ in range(STACK)]
    tot = 0.0
    gains = []
    for t in range(T):
        ob = make_obs(hist) if obs_mode == "stack" else hist[-1]
        mu, _ = mlp_fwd(agent.pi, ob)
        vnew, cost = env.step(mu[:, 0])
        if t > 200: tot += cost.mean()
        vcur = hist[-1][:, 0]
        m = np.abs(vcur) > 0.02
        if m.sum() > 3 and t > 200: gains.append(np.median(-mu[m, 0] / vcur[m]))
        hist.pop(0); hist.append(vnew)
    return tot / ((T - 200) * env.h), float(np.median(gains))

if __name__ == "__main__":
    mode = sys.argv[1]           # train / eval
    tag = sys.argv[2]            # e.g. ml_boot
    obs_mode = sys.argv[3]       # ml / stack
    if mode == "train":
        lam = float(sys.argv[4]); iters = int(sys.argv[5])
        it = train_chunk(tag, obs_mode, lam, iters)
        print(f"{tag}: now at iter {it}", flush=True)
    else:
        J, kimp = evaluate(tag, obs_mode)
        print(f"{tag}: J_deploy={J:.4f}  implied gain={kimp:.2f}", flush=True)
