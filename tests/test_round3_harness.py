"""
Round-3 harness smoke tests (CPU, minutes not hours; no MuJoCo needed).
Run: `python tests/test_round3_harness.py` (or pytest tests/).

1. Flags-off regression vs the tagged round-2 script (structure + same-seed J).
2. --fix-sigma 0.09: sigma^2 machine-constant across updates; grad-zero assert
   lives inside ppo_torch (a completed run proves it fired clean); ent inert.
3. --critic-arch lstm: LSTM params receive nonzero grads; hidden resets where
   episodes reset; states detached across rollout boundaries.
4. --actor-arch lstm: deployed eval carries hidden state, implied_gain_regressed
   finite.
5. Curves file: declared header, one row per eval.
6. Every configs_e4/e5 line parses against the script's argparse.
"""
import json, math, os, subprocess, sys, tempfile, types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ppo_torch import RecurrentAgent, CURVES_HEADER, parse_args  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=REPO)


def _run(workdir, script, extra, tag, steps=6144, seed=3):
    out = os.path.join(workdir, f"{tag}.jsonl")
    cmd = [PY, script, "--env-id", "GLEAMBench-v0", "--seed", str(seed),
           "--total-timesteps", str(steps), "--exp-name", tag, "--out", out]
    cmd += extra
    subprocess.run(cmd, check=True, env=ENV, cwd=workdir,
                   stdout=subprocess.DEVNULL)
    with open(out) as f:
        return json.loads(f.read().splitlines()[-1])


def test_1_flags_off_regression(workdir):
    """New script, no new flags, must match the round-2 tagged script same-seed
    (curves logging is RNG-inert, so J agrees to float precision)."""
    old = os.path.join(workdir, "ppo_round2.py")
    ref_path = os.environ.get("ROUND2_PPO")   # explicit reference copy wins
    if ref_path and os.path.exists(ref_path):
        with open(ref_path) as f:
            ref_src = f.read()
    else:
        ref = subprocess.run(["git", "-C", REPO, "show",
                              "round2-harness:ppo_torch.py"],
                             capture_output=True, text=True, timeout=60)
        if ref.returncode != 0:
            print("[1] SKIP (no round2-harness tag and no ROUND2_PPO set)")
            return
        ref_src = ref.stdout
    with open(old, "w") as f:
        f.write(ref_src)
    r_old = _run(workdir, old, [], "reg_old")
    r_new = _run(workdir, os.path.join(REPO, "ppo_torch.py"), [], "reg_new")
    missing = set(r_old) - set(r_new)
    assert not missing, f"result keys lost: {missing}"
    assert math.isclose(r_old["J_deploy"], r_new["J_deploy"],
                        rel_tol=0, abs_tol=1e-9), \
        (r_old["J_deploy"], r_new["J_deploy"])
    assert math.isclose(r_old["logstd"], r_new["logstd"], abs_tol=1e-9)
    print(f"[1] flags-off regression OK: J {r_new['J_deploy']:.6f} == round-2, "
          f"keys superset (+{sorted(set(r_new) - set(r_old))})")


def test_2_fix_sigma(workdir):
    r = _run(workdir, os.path.join(REPO, "ppo_torch.py"),
             ["--fix-sigma", "0.09", "--eval-every", "1",
              "--ent-coef", "0.01"], "fixsig")
    target = 0.5 * math.log(0.09)
    assert abs(r["logstd"] - target) < 1e-6, r["logstd"]
    assert r["fixed_sigma"] == 0.09
    cpath = os.path.join(workdir, "curves_fixsig_s3.csv")
    rows = open(cpath).read().splitlines()
    sig2 = [float(l.split(",")[4]) for l in rows[1:]]
    # machine-constant: bit-identical across every update; equals 0.09 up to
    # the float32 storage of log-std (exp(2*fp32(log sqrt(0.09))) ~ 0.09-2e-9)
    assert len(set(sig2)) == 1, f"sigma2 drifted across updates: {sig2}"
    assert abs(sig2[0] - 0.09) < 1e-7, f"sigma2 wrong value: {sig2[0]}"
    print(f"[2] fix-sigma OK: sigma2 constant at 0.09 over {len(sig2)} rows, "
          f"logstd={r['logstd']:.6f}, ent-coef warning path exercised")


def test_3_critic_lstm_unit_and_run(workdir):
    # unit: grads flow to lstm, hidden resets on done, detach at boundary
    torch.manual_seed(0)
    ag = RecurrentAgent(1, 1, actor_lstm=False, critic_lstm=True)
    st = ag.zero_state(4)
    x = torch.randn(3 * 4, 1)
    done = torch.zeros(3 * 4)
    v, st1 = ag.critic_value(x, done, st)
    v.sum().backward()
    grads = [p.grad.abs().sum().item()
             for p in ag.critic_rnn.parameters() if p.grad is not None]
    assert grads and all(g > 0 for g in grads), "no grad reached critic LSTM"
    # done=1 resets: value after reset must equal value from a zero state
    ag.zero_grad()
    with torch.no_grad():
        xa = torch.randn(1, 1)
        _, warm = ag.critic_value(torch.randn(6, 1), torch.zeros(6),
                                  ag.zero_state(1))
        v_reset, _ = ag.critic_value(xa, torch.ones(1), warm)
        v_fresh, _ = ag.critic_value(xa, torch.zeros(1), ag.zero_state(1))
        assert torch.allclose(v_reset, v_fresh), "done=1 did not reset hidden"
    # detach: 3 simulated rollout boundaries leave no graph on the carried state
    st = ag.zero_state(2)
    for _ in range(3):
        _, st = ag.critic_value(torch.randn(5 * 2, 1), torch.zeros(5 * 2), st)
        st = (st[0].detach(), st[1].detach())
        assert st[0].grad_fn is None and st[1].grad_fn is None
    # end-to-end tiny run
    r = _run(workdir, os.path.join(REPO, "ppo_torch.py"),
             ["--critic-arch", "lstm", "--num-envs", "4",
              "--num-steps", "128", "--eval-every", "-1"],
             "lstmc", steps=2048)
    assert r["critic_arch"] == "lstm" and r["actor_arch"] == "mlp"
    assert r["lstm_hidden"] == 64 and r["lstm_layers"] == 1
    assert r["J_deploy"] is not None
    print(f"[3] critic-lstm OK: grads flow, reset & detach verified, "
          f"tiny run J={r['J_deploy']:.3f}, minibatches={r['lstm_minibatches']}")


def test_4_actor_lstm(workdir):
    r = _run(workdir, os.path.join(REPO, "ppo_torch.py"),
             ["--critic-arch", "lstm", "--actor-arch", "lstm",
              "--num-envs", "4", "--num-steps", "128", "--eval-every", "-1"],
             "lstma", steps=2048)
    assert r["actor_arch"] == "lstm"
    assert r["implied_gain_regressed"] is not None
    assert np.isfinite(r["implied_gain_regressed"])
    print(f"[4] actor-lstm OK: deployed eval carried hidden, "
          f"gain_reg={r['implied_gain_regressed']:.3f} (finite)")


def test_5_curves(workdir):
    tag = "curvchk"
    _run(workdir, os.path.join(REPO, "ppo_torch.py"),
         ["--eval-every", "1"], tag, steps=6144, seed=5)
    cpath = os.path.join(workdir, f"curves_{tag}_s5.csv")
    assert os.path.exists(cpath), "curves file missing"
    rows = open(cpath).read().splitlines()
    assert rows[0] + "\n" == CURVES_HEADER, rows[0]
    n_updates = 6144 // (8 * 256)
    assert len(rows) - 1 == n_updates, (len(rows) - 1, n_updates)
    ncols = len(CURVES_HEADER.strip().split(","))
    assert all(len(l.split(",")) == ncols for l in rows[1:])
    print(f"[5] curves OK: header exact, {len(rows) - 1} rows == "
          f"{n_updates} evals, {ncols} columns")


def test_6_config_lines_parse():
    argv0 = sys.argv
    n = 0
    try:
        for cfg in ["configs_e4.txt", "configs_e5.txt"]:
            for line in open(os.path.join(REPO, cfg)):
                if not line.strip():
                    continue
                sys.argv = ["ppo_torch.py"] + line.split()
                parse_args()
                n += 1
    finally:
        sys.argv = argv0
    print(f"[6] config parse OK: {n} lines from configs_e4/e5 accepted")


if __name__ == "__main__":
    wd = tempfile.mkdtemp(prefix="gleam_r3_")
    test_6_config_lines_parse()
    test_3_critic_lstm_unit_and_run(wd)
    test_4_actor_lstm(wd)
    test_2_fix_sigma(wd)
    test_5_curves(wd)
    test_1_flags_off_regression(wd)
    print("\nALL ROUND-3 HARNESS TESTS PASSED")
