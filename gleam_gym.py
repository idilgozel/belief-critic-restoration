"""
Gymnasium wrapper for GLEAM-bench (single-env; use gym.vector for batches).

    import gymnasium as gym, gleam_gym
    env = gym.make("GLEAMBench-v0", cs=[0.3], gammas=[1.0], h=0.05)

Continuing task exposed as episodic (episode_len steps, stationary resets).
Reward = -h*(q v^2 + r u^2). 
baselines: env.unwrapped.baselines().
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gleam_bench import GLEBench


class GLEAMGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cs=(0.3,), gammas=(1.0,), theta=1.0, h=0.05,
                 q=1.0, r=1.0, rho=0.05, nl=0.0, episode_len=1024,
                 act_limit=40.0, seed=None):
        self.core = GLEBench(cs=cs, gammas=gammas, theta=theta, h=h, q=q,
                             r=r, rho=rho, nl=nl, n_envs=1,
                             seed=seed if seed is not None else 0)
        self.episode_len = episode_len
        self.t = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float64)
        self.action_space = spaces.Box(-act_limit, act_limit, (1,), np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        obs = self.core.reset()
        self.t = 0
        return obs[0], {}

    def step(self, action):
        obs, cost = self.core.step(np.asarray(action).reshape(1))
        self.t += 1
        trunc = self.t >= self.episode_len
        info = {"privileged_z": float(self.core.privileged()[0, 0]),
                "cost": float(cost[0])}
        return obs[0], float(-cost[0]), False, trunc, info

    def baselines(self, s_e=0.09):
        return self.core.baselines(s_e=s_e)


try:
    gym.register(id="GLEAMBench-v0", entry_point="gleam_gym:GLEAMGym")
except Exception:
    pass
