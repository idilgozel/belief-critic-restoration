"""
Scores results_*.jsonl against the predictions from theory
. Run: python analyze_runs.py results_gleam_e1.jsonl ...
"""
import json, sys, collections
import numpy as np

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        rows += [json.loads(l) for l in f if l.strip()]

by = collections.defaultdict(list)
for r in rows:
    by[r["exp"]].append(r)

def med(exp, key):
    vals = [r[key] for r in by.get(exp, []) if r.get(key) is not None]
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)

print(f"{'experiment':<28} {'n':>2} {'J/ret (med)':>12} {'gain/|a| (med)':>14}")
for exp in sorted(by):
    n = len(by[exp])
    if exp.startswith(("e1", "e3")) or "GLEAM" in by[exp][0]["env"]:
        J, _ = med(exp, "J_deploy"); g, _ = med(exp, "implied_gain")
        print(f"{exp:<28} {n:>2} {J!s:>12} {g!s:>14}")
    else:
        R, _ = med(exp, "mean_return"); a, _ = med(exp, "mean_act_norm")
        print(f"{exp:<28} {n:>2} {R!s:>12} {a!s:>14}")

print("\n--- pre-registered gates ---")
def gate(name, cond):
    print(f"[{'PASS' if cond else 'FAIL' if cond is False else '????'}] {name}")

J_a, _ = med("e1_ml_lam0.9", "J_deploy")
J_b, _ = med("e1_ml_lam1.0", "J_deploy")
J_c, _ = med("e1_stack32_lam0.9", "J_deploy")
g_a, _ = med("e1_ml_lam0.9", "implied_gain")
g_b, _ = med("e1_ml_lam1.0", "implied_gain")
if J_a is not None:
    gate("E1.1 memoryless lam=.9 plateaus in [0.26,0.31], gain>5",
         0.26 <= J_a <= 0.31 and (g_a or 0) > 5)
    gate("E1.2 lam=1.0 no inflation (gain in [0.5,3], J<=0.24)",
         J_b is not None and J_b <= 0.24 and 0.5 <= (g_b or 0) <= 3)
    gate("E1.3 stack32 J<=0.23", J_c is not None and J_c <= 0.23)
    sweep = [med(f"e1_lamsweep_{l}", "J_deploy")[0]
             for l in ["0.95", "0.99", "0.999"]]
    if all(v is not None for v in sweep) and J_a is not None:
        seq = [J_a] + sweep
        gate("E1.4 J decreases along lam sweep, big drop after 0.95",
             all(seq[i] >= seq[i+1] - 0.005 for i in range(len(seq)-1))
             and (seq[1] - seq[3]) > (seq[0] - seq[1]))
for env, tag in [("HalfCheetah", "hc"), ("Walker2d", "w2")]:
    dv = [med(f"e2_{tag}_velonly_lam{l}", "mean_return")[0] for l in ["0.9", "0.99"]]
    df = [med(f"e2_{tag}_full_lam{l}", "mean_return")[0] for l in ["0.9", "0.99"]]
    if all(v is not None for v in dv + df):
        gain_po = dv[1] - dv[0]; gain_full = df[1] - df[0]
        gate(f"E2.1 {env}: lam-benefit on PO >= 3x full-obs",
             gain_po >= 3 * max(gain_full, 1e-9) if gain_po > 0 else False)
for lam in ["0.9", "0.0"]:
    gw, _ = med(f"e3_asym_white003_lam{lam}", "implied_gain")
    ga, _ = med(f"e3_asym_ar003_lam{lam}", "implied_gain")
    if gw is not None and ga is not None:
        if lam == "0.0":
            gate("E3 lam=0: white ruins (gain>6), AR restores (gain<3)",
                 gw > 6 and ga < 3)
        else:
            gate("E3 lam=.9: both restore at Rb=0.03*Szz (gains<3.5)",
                 gw < 3.5 and ga < 3.5)

# ============================================================================
# Diagnostics (post-hoc, NOT registered gates). Round-2 only; prints what it
# finds. Nothing here feeds the gates above.
# ============================================================================
print("\n--- diagnostics (post-hoc, not registered gates) ---")

def spread(exp, key):
    vals = [r[key] for r in by.get(exp, []) if r.get(key) is not None]
    if not vals:
        return None
    return float(np.median(vals)), float(np.mean(vals)), float(np.std(vals)), len(vals)

# frame-skip sanity: the fixed skip4 rows must now DIFFER from their non-skip base
for sk, base in [("r_hc_velonly_lam0.9_skip4", "e2_hc_velonly_lam0.9"),
                 ("r_hc_velonly_lam0.99_skip4", "e2_hc_velonly_lam0.99")]:
    s = med(sk, "mean_return")[0]; b = med(base, "mean_return")[0]
    if s is not None and b is not None:
        tag = "OK differs" if abs(s - b) > 1e-6 else "STILL IDENTICAL (skip is a no-op!)"
        print(f"[skip] {sk} ret={s:.1f} vs {base} ret={b:.1f} -> {tag}")

fs = {exp: sorted({r.get("frame_skip", 0) for r in rs})
      for exp, rs in by.items() if any((r.get("frame_skip") or 0) > 1 for r in rs)}
if fs:
    print("[skip] frame_skip recorded per exp:", fs)

# h_eff baselines for any GLEAM row that actually used frame-skip
gleam_skip = sorted({(r["exp"], r.get("frame_skip", 0)) for r in rows
                     if "GLEAM" in r.get("env", "") and (r.get("frame_skip") or 0) > 1})
if gleam_skip:
    try:
        from gleam_bench import GLEBench
        for exp, k in gleam_skip:
            b = GLEBench(h=0.05 * k).baselines()
            print(f"[h_eff] {exp} skip={k}: baselines at h_eff={0.05 * k:.3f} -> "
                  f"J_opt={b['J_opt']:.4f} J_ml={b['J_ml']:.4f} rThK0={b['rThK0']:.4f}")
    except Exception as e:
        print("[h_eff] baseline recompute skipped:", e)

# stack diagnostic: stack32 (should restore) vs stack8 (negative control)
for exp in ["r_stack32_lam0.9", "r_stack8_lam0.9"]:
    sp = spread(exp, "J_deploy")
    if sp:
        print(f"[stack] {exp}: J med={sp[0]:.4f} mean={sp[1]:.4f} sd={sp[2]:.4f} n={sp[3]}")

# fixed-sigma lambda-sweep: clean re-test of E1.4 without entropy collapse
fx = [(l, med(f"r_fixsig_lam{l}", "J_deploy")[0]) for l in ["0.9", "0.95", "0.99", "0.999"]]
fx = [(l, v) for l, v in fx if v is not None]
if len(fx) >= 2:
    print("[fixed-sigma sweep] " + ", ".join(f"lam{l}={v:.4f}" for l, v in fx))

# E3 lambda=0 time-course endpoints: torch vs numpy-matched batch structure
for exp in ["r_asym_white003_lam0.0", "r_asym_white003_lam0.0_np", "r_asym_ar003_lam0.0"]:
    g = med(exp, "implied_gain")[0]
    if g is not None:
        print(f"[E3] {exp}: implied_gain median={g:.3f}")

# ============================================================================
# E4 registered gate — predictions and windows pre-registered before launch
# (see preregistration-external.md). Three-way per-seed decision with a >=3/5
# majority: restored -> supported, capture -> killed, divergence disambiguated
# by the lr/4 control. E1-E3 registered gates are untouched.
# ============================================================================
import glob as _glob, csv as _csv4
E4_FIXSIG_J_WINDOW      = (0.20, 0.27)  # restored band for deployed J
E4_FIXSIG_GAIN_MAX      = 5             # restored gain ceiling; capture = gain > this, sustained
E4_DIVERGE_J            = 0.45          # divergence J threshold
E4_SEED_MAJORITY        = 3             # of 5; a mixed row is reported as mixed, no verdict
E4_LEARNED_MUST_CAPTURE = (0.29, 7)     # E4.2 batch-validity: J >= 0.29 AND gain >= 7
E4_CAPTURE_SUSTAIN      = 1_000_000     # capture must persist >= this many steps
E5_WINDOWS = None                       # deliberately unfrozen; E5 frozen only after E4 lands

def _seed_captured(exp, seed, final_gain):
    """Capture := implied gain > E4_FIXSIG_GAIN_MAX sustained >= E4_CAPTURE_SUSTAIN
    steps, read from the seed's curve (col 2 = implied gain). If no curve is
    present, fall back to the final-row gain as a proxy."""
    paths = sorted(_glob.glob(f"curves_{exp}_s{seed}.csv")
                   + _glob.glob(f"results/curves/curves_{exp}_s{seed}.csv"))
    for path in paths:
        pts = []
        for row in list(_csv4.reader(open(path)))[1:]:
            try:
                pts.append((int(row[0]), float(row[2])))
            except (ValueError, IndexError):
                pass
        for i, (s0, g0) in enumerate(pts):
            if g0 > E4_FIXSIG_GAIN_MAX and all(
                    g > E4_FIXSIG_GAIN_MAX for s, g in pts[i:]
                    if s <= s0 + E4_CAPTURE_SUSTAIN):
                return True
        return False  # curve present, no sustained crossing
    return (final_gain or 0) > E4_FIXSIG_GAIN_MAX  # no curve: final-gain proxy

def _classify_seed(exp, r):
    """diverged (J>0.45) > captured (sustained gain>5) > restored (J in window & gain<=5)."""
    J = r.get("J_deploy"); g = r.get("implied_gain") or 0.0
    if J is None:
        return "nodata"
    if J > E4_DIVERGE_J:
        return "diverged"
    if _seed_captured(exp, r.get("seed"), g):
        return "captured"
    lo, hi = E4_FIXSIG_J_WINDOW
    if lo <= J <= hi and g <= E4_FIXSIG_GAIN_MAX:
        return "restored"
    return "other"

def _row_verdict(exp):
    rows = by.get(exp, [])
    if not rows:
        return None, {}
    cls = collections.Counter(_classify_seed(exp, r) for r in rows)
    for label in ("restored", "captured", "diverged"):
        if cls[label] >= E4_SEED_MAJORITY:
            return label, dict(cls)
    return "mixed", dict(cls)

def _learned_reproduces_capture(exp):
    """E4.2 batch-validity: >= majority seeds with J >= 0.29 AND gain >= 7."""
    rows = by.get(exp, [])
    jmin, gmin = E4_LEARNED_MUST_CAPTURE
    hit = sum(1 for r in rows if (r.get("J_deploy") or 0) >= jmin
              and (r.get("implied_gain") or 0) >= gmin)
    return hit, len(rows)

print("\n--- E4 registered gate (FROZEN; per-seed, >=3/5 majority) ---")
v_f,  c_f  = _row_verdict("e4_stack32_fixsig_lam0.9")       # E4.1 the kill test
v_hi, c_hi = _row_verdict("e4_stack32_fixsig_lam0.99")      # E4.3 above-threshold control
v_lr, c_lr = _row_verdict("e4_stack32_fixsig_lam0.9_lr4")   # divergence disambiguator
cap_ml, n_ml = _learned_reproduces_capture("e4_stack32_learned_lam0.9")  # E4.2

batch_ok = True
if n_ml and cap_ml < E4_SEED_MAJORITY:
    print(f"[BATCH INVALID] E4.2 learned-sigma did not reproduce capture: "
          f"{cap_ml}/{n_ml} seeds reached J>=0.29 & gain>=7 (need >={E4_SEED_MAJORITY}); "
          f"diagnose harness/env drift, not the conjecture")
    batch_ok = False
if v_hi is not None and v_hi != "restored":
    print(f"[BATCH INVALID] E4.3 lambda=0.99 did not hold restoration "
          f"({c_hi} -> {v_hi}); diagnose, not the conjecture")
    batch_ok = False

if v_f is None:
    print("[E4.1] no results yet")
elif not batch_ok:
    print(f"E4.1 stack32+fixsig lam0.9: seed-classes {c_f} -> {v_f} "
          f"[VERDICT SUPPRESSED — batch invalid]")
else:
    print(f"E4.1 stack32+fixsig lam0.9: seed-classes {c_f} -> {v_f}")
    if v_lr is not None:
        print(f"      lr/4 disambiguator:      {c_lr} -> {v_lr}")
    if v_f == "restored":
        verdict = "PASS: conjecture SUPPORTED (pinned entropy protected restoration)"
    elif v_f == "captured":
        verdict = "KILL: capture WITHOUT entropy collapse -> the gate is not entropy"
    elif v_f == "diverged":
        if v_lr == "restored":
            verdict = ("SUPPORTED with step-size caveat: basin exists (lr/4 restored); "
                       "PPO's default step overshoots it")
        elif v_lr is None:
            verdict = "diverged; lr/4 row missing -> inconclusive (pull the lr/4 row)"
        else:
            verdict = f"KILL: restoration failed at the lr/4 control too (lr/4 -> {v_lr})"
    else:
        verdict = "MIXED -> no verdict (report the seed split as-is)"
    print(f"[E4 VERDICT] {verdict}")

print("\n--- E5 gates (LSTM critic) — freeze only after E4 lands ---")
for exp in ["e5_lstmcritic_lam0.9", "e5_lstmcritic_fixsig_lam0.9",
            "e5_recurrent_lam0.9", "e5_lstmcritic_lam0.99",
            "e5_recurrent_lam0.99"]:
    J5, n5 = med(exp, "J_deploy")
    g5, _ = med(exp, "implied_gain")
    gr, _ = med(exp, "implied_gain_regressed")
    if J5 is not None:
        gate(f"E5 {exp} [IDIL: freeze before launch] "
             f"(J={J5:.4f}, gain={g5}, gain_reg={gr}, n={n5})",
             None if E5_WINDOWS is None else None)

# ---- E4b sigma^2-pin sweep (DIAGNOSTICS, never a gate) ----
def _capture_fraction(exp):
    rows = by.get(exp, [])
    if not rows:
        return None, 0
    caps = sum(1 for r in rows if _classify_seed(exp, r) == "captured")
    return caps / len(rows), len(rows)

_sweep = []
for _pin, _exp in [(0.001, "e4b_stack32_pin0.001_lam0.9"),
                   (0.003, "e4b_stack32_pin0.003_lam0.9"),
                   (0.01,  "e4b_stack32_pin0.01_lam0.9"),
                   (0.03,  "e4b_stack32_pin0.03_lam0.9"),
                   (0.09,  "e4_stack32_fixsig_lam0.9")]:      # E4.1 extends the top
    _fr, _n = _capture_fraction(_exp)
    if _fr is not None:
        _sweep.append((_pin, _fr, _n))
_mlrows = by.get("e4_stack32_learned_lam0.9", [])            # E4.2 at its observed sigma^2
if _mlrows:
    _s2 = [float(np.exp(2 * r["logstd"])) for r in _mlrows if r.get("logstd") is not None]
    _fr, _n = _capture_fraction("e4_stack32_learned_lam0.9")
    if _s2 and _fr is not None:
        _sweep.append((float(np.median(_s2)), _fr, _n))

if _sweep:
    _sweep.sort()
    print("\n--- E4b sigma^2-pin sweep (diagnostics, NOT gates) ---")
    for _pin, _fr, _n in _sweep:
        print(f"  pin sigma2={_pin:.4g}: capture fraction {_fr:.2f} "
              f"({int(round(_fr * _n))}/{_n})")
    for (p0, f0, _n0), (p1, f1, n1) in zip(_sweep, _sweep[1:]):
        if f1 > f0 + 1.0 / max(n1, 1):     # more captures at a HIGHER pin, > 1-seed slack
            print(f"  [ANOMALY] non-monotone: pin {p1:.4g} captures more than {p0:.4g} "
                  f"({f1:.2f} > {f0:.2f}) -> challenges the sigma2-threshold reading")
    _d = {round(p, 4): f for p, f, _ in _sweep}
    if _d.get(0.03, 0) >= 0.6 and _d.get(0.09, 0) >= 0.6:
        print("  [KILL PATTERN] capture at BOTH 0.03 and 0.09 (>=3/5) -> entropy gating killed")
    if _d.get(0.001) == 0:
        print("  [KILL PATTERN] no capture even at 0.001 -> gate is not sigma2 (try time/decay model)")
    if len(_sweep) >= 4:
        try:
            import scipy.optimize as _opt
            _x = np.log10([p for p, _, _ in _sweep])
            _y = np.array([f for _, f, _ in _sweep])
            _logit = lambda x, x0, k: 1.0 / (1.0 + np.exp(k * (x - x0)))
            _popt, _pcov = _opt.curve_fit(_logit, _x, _y,
                                          p0=[np.log10(0.009), 3.0], maxfev=10000)
            _x0, _se = _popt[0], float(np.sqrt(_pcov[0, 0]))
            print(f"  [sigma2_crit] logistic midpoint = {10 ** _x0:.4g} "
                  f"(95% CI {10 ** (_x0 - 1.96 * _se):.4g}-{10 ** (_x0 + 1.96 * _se):.4g}, "
                  f"{len(_sweep)} pts)")
        except Exception as _e:
            print(f"  [sigma2_crit] logistic fit unavailable ({_e})")
    else:
        print(f"  [sigma2_crit] fit deferred: {len(_sweep)}/6 sweep points present")

# ---- round-3 diagnostics from curves files (never gates) ----
import glob, csv as _csv

def _curve_files(prefix):
    pats = [f"curves_{prefix}*.csv", f"results/curves/curves_{prefix}*.csv"]
    return sorted(set(sum((glob.glob(p) for p in pats), [])))

def _capture_time(path, thresh=5.0):
    """First global_step where implied gain exceeds thresh (col 2 in both the
    round-2 and round-3 curve formats); None if never."""
    with open(path) as f:
        rows = list(_csv.reader(f))[1:]
    for r in rows:
        try:
            if float(r[2]) > thresh:
                return int(r[0])
        except (ValueError, IndexError):
            continue
    return None

cap_summary = {}
for prefix in ["e4_", "e5_", "r_stack32", "r_stack8"]:
    for path in _curve_files(prefix):
        exp = path.split("curves_")[-1].rsplit("_", 1)[0]
        cap_summary.setdefault(exp, []).append(_capture_time(path))
if cap_summary:
    print("\n[capture] first step with implied gain > 5, per seed (None = never):")
    for exp in sorted(cap_summary):
        print(f"  {exp:34s} {cap_summary[exp]}")

def _theta2_trajectory(path):
    with open(path) as f:
        rdr = list(_csv.reader(f))
    if not rdr or "theta2_hat" not in rdr[0]:
        return None
    idx = rdr[0].index("theta2_hat")
    vals = [(int(r[0]), float(r[idx])) for r in rdr[1:]
            if len(r) > idx and r[idx] not in ("", "None")]
    return vals or None

th_seen = False
for prefix in ["e4_", "e5_"]:
    for path in _curve_files(prefix):
        tr = _theta2_trajectory(path)
        if tr:
            if not th_seen:
                print("\n[theta2] critic-curvature trajectory (first -> last):")
                th_seen = True
            print(f"  {path}: {tr[0][1]:+.4f} @ {tr[0][0]} -> "
                  f"{tr[-1][1]:+.4f} @ {tr[-1][0]}  ({len(tr)} probes)")
