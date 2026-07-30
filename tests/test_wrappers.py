"""
Wrapper correctness tests — no MuJoCo required (uses a scripted counter env and
the GLEAM env). Run: `python tests/test_wrappers.py` or `pytest tests/`.

Covers the stack32 diagnosis from external-results.md:
  (a) FrameStack + Flatten puts the NEWEST frame at index [-1] (the eval code
      reads `v = obs.reshape(-1)[-1]`).
  (b) the stacked observation actually VARIES across its window (not a frozen
      frame re-stacked).
  (c) SyncVectorEnv-style thunks receive DISTINCT seeds (distinct streams).
plus the new ActionRepeat (frame-skip) wrapper:
  (d) repeats the action k times, sums reward and info['cost'], returns the LAST
      observation, and stops early on termination.
"""
import os, sys, types
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo_torch import ActionRepeat, make_env  # noqa: E402


class CounterEnv(gym.Env):
    """obs = step counter; reward = 1/step; info['cost'] = 0.5/step.
    Optionally terminates at `term_at` to test early-stop."""
    metadata = {"render_modes": []}

    def __init__(self, term_at=None):
        self.observation_space = spaces.Box(-1e9, 1e9, (1,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.t = 0
        self.term_at = term_at

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0]), {}

    def step(self, action):
        self.t += 1
        term = self.term_at is not None and self.t >= self.term_at
        return np.array([float(self.t)]), 1.0, term, False, {"cost": 0.5}


def _framestack(env, n):
    try:
        env = gym.wrappers.FrameStackObservation(env, n)
    except AttributeError:
        env = gym.wrappers.FrameStack(env, n)
    return gym.wrappers.FlattenObservation(env)


def test_framestack_newest_last_and_varies():
    """(a) newest frame at [-1] and monotone; (b) window actually varies."""
    env = _framestack(CounterEnv(), 4)
    env.reset(seed=0)
    obs = None
    for _ in range(6):                      # step past the initial zero-fill
        obs, *_ = env.step(env.action_space.sample())
    flat = np.asarray(obs).reshape(-1)
    assert flat[-1] == flat.max(), f"newest frame not at [-1]: {flat}"
    assert np.all(np.diff(flat) > 0), f"frames not oldest->newest ordered: {flat}"
    assert flat.std() > 0, f"stacked window is frozen: {flat}"
    print(f"[a,b] frame order OK, newest={flat[-1]} window={flat.tolist()}")


def test_sync_thunks_distinct_seeds():
    """(c) distinct seeds -> distinct streams; same seed -> identical."""
    args = types.SimpleNamespace(env_id="GLEAMBench-v0", frame_skip=0,
                                 velocity_only=0, frame_stack=0)
    o_a = make_env(args, 1)().reset(seed=1)[0]
    o_b = make_env(args, 2)().reset(seed=2)[0]
    o_a2 = make_env(args, 1)().reset(seed=1)[0]
    assert not np.allclose(o_a, o_b), "distinct seeds gave identical first obs"
    assert np.allclose(o_a, o_a2), "same seed not reproducible"
    print(f"[c] distinct-seed streams OK: seed1={float(o_a[0]):.4f} seed2={float(o_b[0]):.4f}")


def test_action_repeat_sums_and_returns_last():
    """(d) k repeats: last obs, summed reward + cost, fewer inner env-steps."""
    env = ActionRepeat(CounterEnv(), 4)
    env.reset()
    obs, r, term, trunc, info = env.step(np.zeros(1))
    assert float(obs[0]) == 4.0, f"expected last obs 4.0, got {obs}"
    assert r == 4.0, f"reward not summed over 4 steps: {r}"
    assert abs(info["cost"] - 2.0) < 1e-9, f"cost not summed: {info}"
    assert not term and not trunc
    print(f"[d] action-repeat sums OK: obs={float(obs[0])} r={r} cost={info['cost']}")

    # early stop: base env terminates at t=2, so a k=4 repeat stops after 2
    env2 = ActionRepeat(CounterEnv(term_at=2), 4)
    env2.reset()
    obs2, r2, term2, _, info2 = env2.step(np.zeros(1))
    assert term2 and float(obs2[0]) == 2.0 and r2 == 2.0, (obs2, r2, term2)
    print(f"[d] action-repeat early-stop OK: obs={float(obs2[0])} r={r2} term={term2}")


if __name__ == "__main__":
    test_framestack_newest_last_and_varies()
    test_sync_thunks_distinct_seeds()
    test_action_repeat_sums_and_returns_last()
    print("\nALL WRAPPER TESTS PASSED")
