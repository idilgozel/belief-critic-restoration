"""
Torch PPO, faithful to CleanRL ppo_continuous_action.py
(architecture, hyperparameters, update rule identical; additions are marked):
  + --frame-stack N          (memory agent)
  + --velocity-only          (masks MuJoCo positions; PO variant)
  + --asym-critic MODE       (critic sees obs + privileged z with
                              white|ar noise of variance --rb; GLEAM only)
  + built-in deployed evaluation (cost rate + implied gain for GLEAM;
                                  return + action norm for MuJoCo)
Results appended as one JSON line per run to results.jsonl.
"""
import argparse, json, os, random, time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="GLEAMBench-v0")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--gamma", type=float, default=-1)   # -1 => auto
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--frame-stack", type=int, default=0)
    p.add_argument("--velocity-only", type=int, default=0)
    p.add_argument("--asym-critic", type=str, default="",
                   choices=["", "clean", "white", "ar"])
    p.add_argument("--rb", type=float, default=0.0)
    p.add_argument("--ar-tau", type=float, default=1.0)
    # --- round-2 additions (all default off; round-1 configs run verbatim) ---
    p.add_argument("--frame-skip", type=int, default=0)     # action-repeat k (0/1 = off)
    p.add_argument("--fixed-sigma", "--fix-sigma", dest="fixed_sigma",
                   type=float, default=0.0)  # pin policy variance SIGMA2 (0 = off)
    # curves cadence, in updates: 0 = auto (~every <=100k steps), -1 = off, N>0 = every N
    p.add_argument("--eval-every", type=int, default=0)
    # --- round-3 additions (default mlp = round-2 behavior; lstm mirrors CleanRL
    #     ppo_atari_lstm recurrence mechanics, see RecurrentAgent/main_lstm) ---
    p.add_argument("--critic-arch", type=str, default="mlp", choices=["mlp", "lstm"])
    p.add_argument("--actor-arch", type=str, default="mlp", choices=["mlp", "lstm"])
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--out", type=str, default="results.jsonl")
    return p.parse_args()


class VelocityOnly(gym.ObservationWrapper):
    """Keeps velocity block of MuJoCo obs. Indices verified for
    HalfCheetah-v4 / Walker2d-v4 (obs = 8 pos + 9 vel)."""
    def __init__(self, env, vel_start=8):
        super().__init__(env)
        self.vs = vel_start
        d = env.observation_space.shape[0] - vel_start
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d,),
                                                np.float64)
    def observation(self, obs):
        return obs[self.vs:]


class ActionRepeat(gym.Wrapper):
    """Action-repeat (frame-skip): apply the chosen action k times, summing
    reward (and info['cost'] for GLEAM), returning the LAST observation and the
    last step's info; stop early if any inner step terminates/truncates.
    Placed innermost so downstream obs wrappers see the decimated stream."""
    def __init__(self, env, k):
        super().__init__(env)
        self.k = int(k)

    def step(self, action):
        total_r, total_cost, has_cost = 0.0, 0.0, False
        obs, term, trunc, info = None, False, False, {}
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += r
            if "cost" in info:
                total_cost += info["cost"]; has_cost = True
            if term or trunc:
                break
        if has_cost:
            info = dict(info); info["cost"] = total_cost
        return obs, total_r, term, trunc, info


def make_env(args, seed):
    def thunk():
        if args.env_id.startswith("GLEAM"):
            import gleam_gym  # noqa: F401  registers GLEAMBench-v0
        env = gym.make(args.env_id)
        if args.frame_skip and args.frame_skip > 1:
            env = ActionRepeat(env, args.frame_skip)
        if args.velocity_only:
            env = VelocityOnly(env)
        if args.frame_stack > 1:
            try:
                env = gym.wrappers.FrameStackObservation(env, args.frame_stack)
            except AttributeError:
                env = gym.wrappers.FrameStack(env, args.frame_stack)
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_dim=None):
        super().__init__()
        cdim = critic_dim or obs_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(cdim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, xc=None, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return (action, probs.log_prob(action).sum(1),
                probs.entropy().sum(1),
                self.critic(xc if xc is not None else x))


def _lstm_init(lstm):
    # CleanRL ppo_atari_lstm init: orthogonal weights (std 1.0), zero biases
    for name, param in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.orthogonal_(param, 1.0)
    return lstm


class RecurrentAgent(nn.Module):
    """E5 architectures (selected only by --critic-arch/--actor-arch lstm).
    Recurrence mechanics mirror CleanRL ppo_atari_lstm.py: one-layer
    LSTM(obs->64), orthogonal init, hidden carried across steps within a
    rollout, reset where episodes reset (done masks state BEFORE the step's
    input), detached at rollout boundaries (truncated BPTT), initial states
    saved and replayed for minibatch updates. mlp towers and all heads are
    identical to Agent. LSTM towers consume the raw observation stream."""
    HID = 64

    def __init__(self, obs_dim, act_dim, actor_lstm=False, critic_lstm=True):
        super().__init__()
        self.actor_lstm, self.critic_lstm = actor_lstm, critic_lstm
        if critic_lstm:
            self.critic_rnn = _lstm_init(nn.LSTM(obs_dim, self.HID))
            self.critic_head = layer_init(nn.Linear(self.HID, 1), std=1.0)
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0))
        if actor_lstm:
            self.actor_rnn = _lstm_init(nn.LSTM(obs_dim, self.HID))
            self.actor_head = layer_init(nn.Linear(self.HID, act_dim), std=0.01)
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def zero_state(self, n):
        return (torch.zeros(1, n, self.HID), torch.zeros(1, n, self.HID))

    @staticmethod
    def _seq(rnn, x, done, state):
        """CleanRL get_states: x (T*N, obs) time-major over the state's batch
        N; done (T*N,); resets state where done=1 before consuming that step."""
        n = state[0].shape[1]
        xs = x.reshape(-1, n, x.shape[-1])
        ds = done.reshape(-1, n)
        outs = []
        for xt, dt in zip(xs, ds):
            state = (state[0] * (1.0 - dt).view(1, -1, 1),
                     state[1] * (1.0 - dt).view(1, -1, 1))
            out, state = rnn(xt.unsqueeze(0), state)
            outs.append(out)
        return torch.cat(outs).flatten(0, 1), state

    def critic_value(self, x, done, cstate):
        if self.critic_lstm:
            h, cstate = self._seq(self.critic_rnn, x, done, cstate)
            return self.critic_head(h), cstate
        return self.critic(x), cstate

    def actor_mean_step(self, x, done, astate):
        if self.actor_lstm:
            h, astate = self._seq(self.actor_rnn, x, done, astate)
            return self.actor_head(h), astate
        return self.actor_mean(x), astate

    def get_action_and_value(self, x, done, astate, cstate, action=None):
        mean, astate = self.actor_mean_step(x, done, astate)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        value, cstate = self.critic_value(x, done, cstate)
        return (action, probs.log_prob(action).sum(1), probs.entropy().sum(1),
                value, astate, cstate)


def deployed_eval(agent, args, seed_offset, n_steps, burnin, max_episodes=None):
    """Run the deterministic policy (actor mean) on a fresh eval env and return
    the deployed metrics. Uses only the env's own RNG and a no-grad forward, so
    it never advances the training torch/numpy RNG streams. GLEAM: one long
    episode, h-normalized cost rate + median implied gain after burn-in. MuJoCo:
    mean return + mean action norm. With the final-eval params this reproduces
    the round-1 inline evaluation exactly."""
    is_gleam = args.env_id.startswith("GLEAM")
    if is_gleam:
        # single long episode: episodic resets draw from the UNCONTROLLED
        # stationary law and would bias the cost-rate estimate upward
        import gleam_gym  # noqa: F401
        eval_env = gym.make(args.env_id, episode_len=10 ** 9)
        if args.frame_skip and args.frame_skip > 1:
            eval_env = ActionRepeat(eval_env, args.frame_skip)
        if args.frame_stack > 1:
            try:
                eval_env = gym.wrappers.FrameStackObservation(
                    eval_env, args.frame_stack)
            except AttributeError:
                eval_env = gym.wrappers.FrameStack(eval_env, args.frame_stack)
            eval_env = gym.wrappers.FlattenObservation(eval_env)
    else:
        eval_env = make_env(args, seed_offset)()
    tot_cost, gains, rets, act_norms, T = 0.0, [], [], [], 0
    uv_v, uv_u = [], []          # (v, u) pairs for the regressed implied gain
    recurrent_actor = bool(getattr(agent, "actor_lstm", False))
    astate = agent.zero_state(1) if recurrent_actor else None
    adone = torch.zeros(1)
    obs, _ = eval_env.reset(seed=seed_offset)
    ep_ret = 0.0
    for t in range(n_steps):
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs), dtype=torch.float32).reshape(1, -1)
            if recurrent_actor:
                mean, astate = agent.actor_mean_step(x, adone, astate)
                a = mean.numpy()[0]
            else:
                a = agent.actor_mean(x).numpy()[0]
        obs2, r, term, trunc, info = eval_env.step(a)
        ep_ret += r
        act_norms.append(float(np.linalg.norm(a)))
        if is_gleam and t > burnin:
            tot_cost += info["cost"]; T += 1
            v = float(np.asarray(obs).reshape(-1)[-1])
            uv_v.append(v); uv_u.append(float(a[0]))
            if abs(v) > 0.02:
                gains.append(-float(a[0]) / v)
        if term or trunc:
            rets.append(ep_ret); ep_ret = 0.0
            if max_episodes and len(rets) >= max_episodes:
                break
            obs, _ = eval_env.reset()
            adone = torch.ones(1)
        else:
            obs = obs2
            adone = torch.zeros(1)
    out = {}
    if is_gleam:
        h = eval_env.unwrapped.core.h
        out["J_deploy"] = tot_cost / (T * h) if T else None
        out["implied_gain"] = float(np.median(gains)) if gains else None
        den = float(np.dot(uv_v, uv_v)) if uv_v else 0.0
        out["implied_gain_regressed"] = \
            (-float(np.dot(uv_u, uv_v)) / den) if den > 0 else None
        out["baselines"] = eval_env.unwrapped.baselines()
    else:
        out["mean_return"] = float(np.mean(rets)) if rets else None
        out["mean_act_norm"] = float(np.mean(act_norms))
    eval_env.close()
    return out


CURVES_HEADER = ("global_step,deployed_J,implied_gain,implied_gain_regressed,"
                 "sigma2,value_loss,policy_loss,entropy,theta2_hat\n")


def theta2_probe(agent, args):
    """Critic curvature on a fixed v-grid (GLEAM only; diagnostics, never a
    gate): fit V(v) ~ c2 v^2 + c1 v + c0, return c2. LSTM critic: hidden state
    frozen after a 200-step burn-in trajectory under the current actor mean."""
    if not args.env_id.startswith("GLEAM"):
        return None
    grid = np.linspace(-2.0, 2.0, 41).astype(np.float32)
    with torch.no_grad():
        if getattr(agent, "critic_lstm", False):
            import gleam_gym  # noqa: F401
            env = gym.make(args.env_id, episode_len=10 ** 9)
            obs, _ = env.reset(seed=args.seed + 88_000)
            cstate = agent.zero_state(1)
            astate = agent.zero_state(1) if agent.actor_lstm else None
            done = torch.zeros(1)
            for _ in range(200):
                x = torch.tensor(np.asarray(obs),
                                 dtype=torch.float32).reshape(1, -1)
                _, cstate = agent.critic_value(x, done, cstate)
                mean, astate = agent.actor_mean_step(x, done, astate)
                obs, _, _, _, _ = env.step(mean.numpy()[0])
            env.close()
            vals = [float(agent.critic_value(
                        torch.tensor([[v]], dtype=torch.float32), done,
                        (cstate[0].clone(), cstate[1].clone()))[0])
                    for v in grid]
        else:
            reps = args.frame_stack if args.frame_stack > 1 else 1
            X = np.repeat(grid[:, None], reps, axis=1)
            if args.asym_critic:
                X = np.concatenate(
                    [X, np.zeros((len(grid), 1), np.float32)], axis=1)
            critic = agent.critic if hasattr(agent, "critic") else None
            vals = critic(torch.tensor(X, dtype=torch.float32)
                          ).flatten().numpy() if critic is not None else None
            if vals is None:
                return None
    return float(np.polyfit(grid.astype(float), np.asarray(vals, float), 2)[0])


def _periodic_eval(agent, args, update, loss_stats=None, probe_state=None):
    """Deployed eval + training stats appended to curves_<exp>_s<seed>.csv.
    Logging only: uses env RNG + no-grad forwards, never the training streams.
    MuJoCo rows reuse the columns as deployed_J:=mean_return,
    implied_gain:=mean_act_norm, implied_gain_regressed empty."""
    is_gleam = args.env_id.startswith("GLEAM")
    m = deployed_eval(agent, args, seed_offset=args.seed + 77_000,
                      n_steps=(20_000 if is_gleam else 10_000),
                      burnin=(2_000 if is_gleam else 0),
                      max_episodes=(None if is_gleam else 5))
    gstep = update * args.num_envs * args.num_steps
    primary = m.get("J_deploy", m.get("mean_return"))
    secondary = m.get("implied_gain", m.get("mean_act_norm"))
    reg = m.get("implied_gain_regressed")
    reg = "" if reg is None else reg
    sig2 = float(np.exp(2.0 * float(agent.actor_logstd.detach().mean())))
    vl = pl = en = ""
    if loss_stats and loss_stats["n"]:
        vl = loss_stats["v"] / loss_stats["n"]
        pl = loss_stats["p"] / loss_stats["n"]
        en = loss_stats["e"] / loss_stats["n"]
    th2 = ""
    if probe_state is not None and \
            gstep - probe_state.get("last", -10 ** 18) >= 500_000:
        t2 = theta2_probe(agent, args)
        if t2 is not None:
            th2 = t2
            probe_state["last"] = gstep
    cpath = f"curves_{args.exp_name}_s{args.seed}.csv"
    new = not os.path.exists(cpath)
    with open(cpath, "a") as cf:
        if new:
            cf.write(CURVES_HEADER)
        cf.write(f"{gstep},{primary},{secondary},{reg},{sig2},"
                 f"{vl},{pl},{en},{th2}\n")


def main():
    args = parse_args()
    if args.fixed_sigma > 0 and args.ent_coef != 0:
        print(f"WARNING: --fixed-sigma pins the policy std, so "
              f"--ent-coef {args.ent_coef} is inert; forcing it to 0")
        args.ent_coef = 0.0
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.critic_arch != "mlp" or args.actor_arch != "mlp":
        return main_lstm(args, device)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.seed + i) for i in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    asym = bool(args.asym_critic)
    critic_dim = obs_dim + 1 if asym else None
    agent = Agent(obs_dim, act_dim, critic_dim).to(device)
    if args.fixed_sigma > 0:
        # pin exploration: log-std = log(sqrt(SIGMA2)), frozen out of the optimizer
        with torch.no_grad():
            agent.actor_logstd.fill_(0.5 * float(np.log(args.fixed_sigma)))
        agent.actor_logstd.requires_grad_(False)
    optimizer = optim.Adam([p for p in agent.parameters() if p.requires_grad],
                           lr=args.lr, eps=1e-5)

    if args.gamma < 0:
        gamma = 0.9975 if args.env_id.startswith("GLEAM") else 0.99
    else:
        gamma = args.gamma

    # AR(1) belief-noise state (asym mode)
    a_ar = np.exp(-0.05 / args.ar_tau)
    eps = np.sqrt(args.rb) * np.random.randn(args.num_envs)

    def critic_obs(next_obs_np, infos):
        nonlocal eps
        z = np.array([infos.get("privileged_z", np.zeros(args.num_envs))]
                     ).reshape(args.num_envs) \
            if isinstance(infos, dict) else np.zeros(args.num_envs)
        if args.asym_critic == "white":
            eps = np.sqrt(args.rb) * np.random.randn(args.num_envs)
        elif args.asym_critic == "ar":
            eps = a_ar * eps + np.sqrt((1 - a_ar ** 2) * args.rb) * \
                  np.random.randn(args.num_envs)
        else:
            eps = np.zeros(args.num_envs)
        zt = torch.tensor(z + eps, dtype=torch.float32).unsqueeze(1)
        return torch.cat([torch.as_tensor(next_obs_np, dtype=torch.float32),
                          zt], dim=1).to(device)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    obsc_buf = torch.zeros((args.num_steps, args.num_envs,
                            critic_dim or obs_dim)).to(device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    done_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    val_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs_np, infos = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    eval_every = args.eval_every if args.eval_every != 0 else max(
        1, 100_000 // (args.num_envs * args.num_steps))
    probe_state = {"last": -10 ** 18}
    t0 = time.time()

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = \
                args.lr * (1.0 - (update - 1.0) / num_updates)
        _ls = dict(v=0.0, p=0.0, e=0.0, n=0)
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            oc = critic_obs(next_obs_np, infos) if asym else next_obs
            obsc_buf[step] = oc
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(
                    next_obs, oc if asym else None)
                val_buf[step] = value.flatten()
            act_buf[step] = action; logp_buf[step] = logp
            next_obs_np, reward, term, trunc, infos = envs.step(
                action.cpu().numpy())
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32
                                         ).to(device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32
                                    ).to(device)
            next_done = torch.tensor(np.logical_or(term, trunc),
                                     dtype=torch.float32).to(device)
        with torch.no_grad():
            oc = critic_obs(next_obs_np, infos) if asym else next_obs
            next_value = agent.get_value(oc).reshape(1, -1)
            adv = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nextnonterm = (1.0 - next_done if t == args.num_steps - 1
                               else 1.0 - done_buf[t + 1])
                nextval = (next_value if t == args.num_steps - 1
                           else val_buf[t + 1])
                delta = rew_buf[t] + gamma * nextval * nextnonterm - val_buf[t]
                lastgaelam = delta + gamma * args.gae_lambda * \
                    nextnonterm * lastgaelam
                adv[t] = lastgaelam
            ret = adv + val_buf
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_obsc = obsc_buf.reshape(-1, critic_dim or obs_dim)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1, act_dim)
        b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)
        bsz = args.num_envs * args.num_steps
        inds = np.arange(bsz)
        mbsz = bsz // args.num_minibatches
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, bsz, mbsz):
                mb = inds[s:s + mbsz]
                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_obsc[mb] if asym else None, b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                            1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                newval = newval.view(-1)
                v_unc = (newval - b_ret[mb]) ** 2
                v_clip = b_val[mb] + torch.clamp(newval - b_val[mb],
                                                 -args.clip_coef,
                                                 args.clip_coef)
                v_loss = 0.5 * torch.max(v_unc,
                                         (v_clip - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + \
                    args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                if args.fixed_sigma > 0 and update == 1:
                    assert agent.actor_logstd.grad is None or \
                        not agent.actor_logstd.grad.abs().any(), \
                        "fixed-sigma: log-std received gradient"
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         args.max_grad_norm)
                optimizer.step()
                _ls["v"] += v_loss.item(); _ls["p"] += pg_loss.item()
                _ls["e"] += entropy.mean().item(); _ls["n"] += 1

        if eval_every > 0 and update % eval_every == 0:
            _periodic_eval(agent, args, update, _ls, probe_state)

    # ---------------- deployed evaluation ----------------
    is_gleam = args.env_id.startswith("GLEAM")
    metrics = deployed_eval(agent, args, seed_offset=args.seed + 10_000,
                            n_steps=(200_000 if is_gleam else 20_000),
                            burnin=2000)
    result = dict(exp=args.exp_name, env=args.env_id, lam=args.gae_lambda,
                  seed=args.seed, frame_stack=args.frame_stack,
                  frame_skip=args.frame_skip, fixed_sigma=args.fixed_sigma,
                  critic_arch=args.critic_arch, actor_arch=args.actor_arch,
                  velocity_only=args.velocity_only,
                  asym=args.asym_critic, rb=args.rb,
                  steps=args.total_timesteps,
                  wall_s=round(time.time() - t0, 1),
                  logstd=float(agent.actor_logstd.detach().mean()))
    result.update(metrics)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def main_lstm(args, device):
    """E5 recurrent path (reached only via --critic-arch/--actor-arch lstm;
    flags-off runs never enter here). Rollout, GAE, and all loss formulas are
    identical to main(); the differences are the CleanRL ppo_atari_lstm
    recurrence mechanics and its whole-env sequence minibatching (initial LSTM
    states saved per rollout and replayed; num_minibatches clamped to a divisor
    of num_envs, recorded in the results dict)."""
    if args.frame_stack > 1:
        raise SystemExit("lstm arch replaces frame-stack (recurrence supplies "
                         "memory): drop --frame-stack")
    if args.asym_critic:
        raise SystemExit("asym-critic + lstm is out of round-3 scope")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args, args.seed + i) for i in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = RecurrentAgent(obs_dim, act_dim,
                           actor_lstm=(args.actor_arch == "lstm"),
                           critic_lstm=(args.critic_arch == "lstm")).to(device)
    if args.fixed_sigma > 0:
        with torch.no_grad():
            agent.actor_logstd.fill_(0.5 * float(np.log(args.fixed_sigma)))
        agent.actor_logstd.requires_grad_(False)
    optimizer = optim.Adam([p for p in agent.parameters() if p.requires_grad],
                           lr=args.lr, eps=1e-5)
    if args.gamma < 0:
        gamma = 0.9975 if args.env_id.startswith("GLEAM") else 0.99
    else:
        gamma = args.gamma

    nmb = min(args.num_minibatches, args.num_envs)
    while args.num_envs % nmb:
        nmb -= 1
    if nmb != args.num_minibatches:
        print(f"lstm: num_minibatches {args.num_minibatches} -> {nmb} "
              f"(whole-env sequence minibatching)")
    envsperbatch = args.num_envs // nmb

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    T, N = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, N, obs_dim)).to(device)
    act_buf = torch.zeros((T, N, act_dim)).to(device)
    logp_buf = torch.zeros((T, N)).to(device)
    rew_buf = torch.zeros((T, N)).to(device)
    done_buf = torch.zeros((T, N)).to(device)
    val_buf = torch.zeros((T, N)).to(device)

    next_obs_np, infos = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.zeros(N).to(device)
    astate = agent.zero_state(N)
    cstate = agent.zero_state(N)
    eval_every = args.eval_every if args.eval_every != 0 else max(
        1, 100_000 // (N * T))
    probe_state = {"last": -10 ** 18}
    t0 = time.time()

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = \
                args.lr * (1.0 - (update - 1.0) / num_updates)
        init_astate = (astate[0].clone(), astate[1].clone())
        init_cstate = (cstate[0].clone(), cstate[1].clone())
        _ls = dict(v=0.0, p=0.0, e=0.0, n=0)
        for step in range(T):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value, astate, cstate = \
                    agent.get_action_and_value(next_obs, next_done,
                                               astate, cstate)
                val_buf[step] = value.flatten()
            act_buf[step] = action; logp_buf[step] = logp
            next_obs_np, reward, term, trunc, infos = envs.step(
                action.cpu().numpy())
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32
                                         ).to(device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32
                                    ).to(device)
            next_done = torch.tensor(np.logical_or(term, trunc),
                                     dtype=torch.float32).to(device)
        with torch.no_grad():
            next_value, _ = agent.critic_value(next_obs, next_done, cstate)
            next_value = next_value.reshape(1, -1)
            adv = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(T)):
                nextnonterm = (1.0 - next_done if t == T - 1
                               else 1.0 - done_buf[t + 1])
                nextval = (next_value if t == T - 1 else val_buf[t + 1])
                delta = rew_buf[t] + gamma * nextval * nextnonterm - val_buf[t]
                lastgaelam = delta + gamma * args.gae_lambda * \
                    nextnonterm * lastgaelam
                adv[t] = lastgaelam
            ret = adv + val_buf
        # truncated BPTT: the next rollout starts from detached states
        astate = (astate[0].detach(), astate[1].detach())
        cstate = (cstate[0].detach(), cstate[1].detach())

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1, act_dim)
        b_done = done_buf.reshape(-1)
        b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)
        envinds = np.arange(N)
        flatinds = np.arange(T * N).reshape(T, N)
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, N, envsperbatch):
                mbenv = envinds[start:start + envsperbatch]
                mb = flatinds[:, mbenv].ravel()
                ia = (init_astate[0][:, mbenv], init_astate[1][:, mbenv])
                ic = (init_cstate[0][:, mbenv], init_cstate[1][:, mbenv])
                _, newlogp, entropy, newval, _, _ = agent.get_action_and_value(
                    b_obs[mb], b_done[mb], ia, ic, b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                            1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                newval = newval.view(-1)
                v_unc = (newval - b_ret[mb]) ** 2
                v_clip = b_val[mb] + torch.clamp(newval - b_val[mb],
                                                 -args.clip_coef,
                                                 args.clip_coef)
                v_loss = 0.5 * torch.max(v_unc,
                                         (v_clip - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + \
                    args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                if args.fixed_sigma > 0 and update == 1:
                    assert agent.actor_logstd.grad is None or \
                        not agent.actor_logstd.grad.abs().any(), \
                        "fixed-sigma: log-std received gradient"
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         args.max_grad_norm)
                optimizer.step()
                _ls["v"] += v_loss.item(); _ls["p"] += pg_loss.item()
                _ls["e"] += entropy.mean().item(); _ls["n"] += 1

        if eval_every > 0 and update % eval_every == 0:
            _periodic_eval(agent, args, update, _ls, probe_state)

    is_gleam = args.env_id.startswith("GLEAM")
    metrics = deployed_eval(agent, args, seed_offset=args.seed + 10_000,
                            n_steps=(200_000 if is_gleam else 20_000),
                            burnin=2000)
    result = dict(exp=args.exp_name, env=args.env_id, lam=args.gae_lambda,
                  seed=args.seed, frame_stack=args.frame_stack,
                  frame_skip=args.frame_skip, fixed_sigma=args.fixed_sigma,
                  critic_arch=args.critic_arch, actor_arch=args.actor_arch,
                  lstm_hidden=RecurrentAgent.HID, lstm_layers=1,
                  lstm_init="orthogonal", lstm_minibatches=nmb,
                  velocity_only=args.velocity_only,
                  asym=args.asym_critic, rb=args.rb,
                  steps=args.total_timesteps,
                  wall_s=round(time.time() - t0, 1),
                  logstd=float(agent.actor_logstd.detach().mean()))
    result.update(metrics)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
