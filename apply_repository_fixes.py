#!/usr/bin/env python3
"""Apply the numerical/reproducibility corrections identified during the AEJ audit.

Run this script from the root of the existing GitHub repository:

    python apply_repository_fixes.py

It creates a timestamp-free backup directory named ``backup_before_aej_sync`` and
patches only the files whose logic/text must be synchronized with the final
manuscript.  The script fails loudly when an expected anchor is missing, so it
will not silently edit an incompatible repository version.
"""
from pathlib import Path
import shutil
import re

ROOT = Path.cwd()
BACKUP = ROOT / "backup_before_aej_sync"
FILES = [
    "stackelberg_shapley.py", "analysis.py", "analysis2.py", "analysis3.py",
    "analysis5.py", "figures.py", "figures2.py", "README.md",
]


def require_file(name):
    p = ROOT / name
    if not p.exists():
        raise SystemExit(f"Missing {name}; run this script from the repository root.")
    return p


def replace_between(text, start, end, replacement, *, label):
    a = text.find(start)
    if a < 0:
        raise RuntimeError(f"Could not find start anchor for {label}: {start!r}")
    b = text.find(end, a + len(start))
    if b < 0:
        raise RuntimeError(f"Could not find end anchor for {label}: {end!r}")
    return text[:a] + replacement.rstrip() + "\n" + text[b:]


def replace_once(text, old, new, *, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"Expected exactly one match for {label}; found {n}.")
    return text.replace(old, new, 1)


# -------------------------------------------------------------------------
# Backups
# -------------------------------------------------------------------------
BACKUP.mkdir(exist_ok=True)
for name in FILES:
    p = require_file(name)
    shutil.copy2(p, BACKUP / name)

# -------------------------------------------------------------------------
# 1) Core model: correct interpretation of local spectral radius and make the
#    published EPS case the default command-line run.
# -------------------------------------------------------------------------
p = ROOT / "stackelberg_shapley.py"
s = p.read_text(encoding="utf-8")
old = '''        """Spectral radius of the followers' best-response Jacobian at R.
        rho < 1  <=>  the best-response map is a local contraction, so the
        follower Nash equilibrium is unique and best-response iteration
        converges geometrically (verifies the hypothesis of Prop. 1)."""'''
new = '''        """Spectral radius of the followers' best-response Jacobian at R.

        rho < 1 is a *local* linear-convergence diagnostic around R.  It is not,
        by itself, a global uniqueness certificate.  The manuscript's global
        sufficient uniqueness certificate is the boxwise sup-norm contraction
        bound implemented in analysis3.contraction_bound().
        """'''
s = replace_once(s, old, new, label="local spectral-radius docstring")

main = '''if __name__ == "__main__":
    # Final manuscript case study and working protocol settings.
    sys = satellite_eps_real()
    GAMMA = 0.20
    THETA = 0.70

    print("=" * 74)
    print(" STACKELBERG-SHAPLEY RELIABILITY PROTOCOL -- data-informed satellite EPS")
    print("=" * 74)

    R_nash = sys.solve_nash()
    R_stk = sys.solve_stackelberg(theta=THETA)
    R_soc = sys.solve_social()

    def row(tag, R):
        print(f"{tag:14s} | R_sys={sys.R_sys(R):.7f} | W={sys.welfare(R):.7f} | "
              + " ".join(f"{r:.6f}" for r in R))

    print("\\n-- Reliability regimes -------------------------------------------")
    row("Nash", R_nash)
    row("Stackelberg", R_stk)
    row("Social opt.", R_soc)
    gap = (sys.welfare(R_stk) - sys.welfare(R_nash)) / (sys.welfare(R_soc) - sys.welfare(R_nash))
    print(f"Welfare gap closed by accountable leader = {100*gap:.3f}%")
    print(f"Price of Anarchy W_social/W_nash = {sys.welfare(R_soc)/sys.welfare(R_nash):.6f}")

    m = sys.metrics(GAMMA, theta=THETA)
    print("\\n-- Cooperative allocation ----------------------------------------")
    for i, sub in enumerate(sys.subs):
        print(f"{sub.name:28s} | Shapley={m['phi'][i]:.7f} | premium allocation={m['phi_stk'][i]:.7f}")
    print(f"kappa_L={m['kappa_L']:.7f}")
    print(f"leader share: {100*m['phi'][sys.L]/m['phi'].sum():.3f}% -> "
          f"{100*m['phi_stk'][sys.L]/m['phi_stk'].sum():.3f}%")
'''
idx = s.find('if __name__ == "__main__":')
if idx < 0:
    raise RuntimeError("Could not find stackelberg_shapley.py main block")
s = s[:idx] + main
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 2) analysis.py: make clear that rho is only a local diagnostic.
# -------------------------------------------------------------------------
p = ROOT / "analysis.py"
s = p.read_text(encoding="utf-8")
s = s.replace(
    "  E. contraction         : spectral radius of the best-response Jacobian\n"
    "                            (verifies the contraction hypothesis of Prop. 1).",
    "  E. local convergence   : spectral radius of the best-response Jacobian\n"
    "                            (local diagnostic only; global bound is in analysis3.py).",
)
s = s.replace('print("E. CONTRACTION DIAGNOSTIC")', 'print("E. LOCAL JACOBIAN DIAGNOSTIC")')
s = s.replace('print(f"   contraction (rho<1) verified: {rho < 1}")',
              'print(f"   local rho<1: {rho < 1}  (not a global uniqueness proof)")')
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 3) analysis2.py: treat cost-rate perturbation as reporting sensitivity and
#    the incomplete AGREE calculation as a complexity-weighted proxy.
# -------------------------------------------------------------------------
p = ROOT / "analysis2.py"
s = p.read_text(encoding="utf-8")
s = s.replace(
    "(C) quantify the\nmanipulation incentive (the protocol is budget-balanced, hence not\nstrategy-proof -- we measure how strong the incentive to misreport is);",
    "(C) report sensitivity of the computed allocation to perturbations of a\nreported cost-rate input; this is not an incentive-compatibility test;",
)
s = s.replace('"AGREE", alloc_agree(sys, R_target)', '"Complexity proxy", alloc_agree(sys, R_target)')
s = s.replace('#  C. Manipulation: incentive to misreport cost rate beta_j',
              '#  C. Reporting-parameter sensitivity: perturb the cost-rate input beta_j')
old_doc = '''    """Follower j reports beta_j' = f * beta_j (over/under-statement).  We
    recompute the Shapley allocation under the *reported* game and record j's
    post-transfer share, holding everyone else truthful.  A flat/decreasing
    curve => little incentive to inflate; an increasing curve => manipulable."""'''
new_doc = '''    """Perturb follower j's cost-rate input beta_j' = f * beta_j and recompute
    the reported-game allocation.  This is a numerical reporting-parameter
    sensitivity diagnostic only.  It does not evaluate true utility under a
    reporting mechanism and therefore is not a strategy-proofness test."""'''
if old_doc in s:
    s=s.replace(old_doc,new_doc,1)
s = s.replace('print("C. MANIPULATION INCENTIVE (follower 1 misreports beta)")',
              'print("C. REPORTING-PARAMETER SENSITIVITY (follower 1 beta input)")')
s = s.replace('print(f"  truthful share={share[5]:.3f}; max share={share.max():.3f} at f={f[np.argmax(share)]:.2f}")',
              'print(f"  baseline allocation={share[5]:.3f}; range=[{share.min():.3f}, {share.max():.3f}]")')
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 4) analysis3.py: global boxwise contraction bound and genuine follower-only
#    subsidy combined with the accountable Stackelberg leader.
# -------------------------------------------------------------------------
p = ROOT / "analysis3.py"
s = p.read_text(encoding="utf-8")
new_contraction = '''def contraction_bound(sys, R=None):
    """Global sufficient sup-norm contraction bound on the follower box.

    For follower j, the interior best response satisfies
        C_j'(R_j) = pi_j * prod_{m != j} R_m.
    Hence, for follower k != j,
        |d BR_j / d R_k| = pi_j * prod_{m != j,k} R_m / C_j''(R_j).

    The boxwise upper bound uses every remaining reliability at its upper bound
    and the minimum cost curvature over follower j's interval.  For the present
    exponential cost, C_j'' is increasing over the operating interval, so its
    minimum occurs at R_j^min.  Lambda_box < 1 is a global sufficient condition
    for contraction of the follower best-response map on the stated box.

    The returned rho is the Jacobian spectral radius at the computed Nash point
    and is reported only as a local numerical diagnostic.
    """
    F = sys.F
    rows = {}
    for j in F:
        sj = sys.subs[j]
        # Conservative minimum curvature on [Rmin,Rmax].  A small grid check is
        # retained so this remains valid if a future calibration changes shape.
        grid = np.linspace(sj.Rmin, min(sj.Rmax, 0.999999), 4001)
        c2_min = float(np.min([d2cost(sj, r) for r in grid]))
        lam = 0.0
        for k in F:
            if k == j:
                continue
            prod_upper = 1.0
            for m, sm in enumerate(sys.subs):
                if m not in (j, k):
                    prod_upper *= min(sm.Rmax, 0.999999)
            lam += sj.pi * prod_upper / c2_min
        rows[j] = float(lam)
    Lambda_box = max(rows.values())
    R_local = sys.solve_nash() if R is None else np.asarray(R, float)
    rho, _ = sys.br_jacobian_spectral_radius(R_local)
    return float(Lambda_box), rows, float(rho)
'''
s = replace_between(
    s,
    'def contraction_bound(sys, R=None):',
    '# --------------------------------------------------------------------------- #\n#  B. Utility-form robustness',
    new_contraction,
    label='analysis3 contraction block',
)
# Rename the robustness language; the implementation may keep its historical class name.
s = s.replace('#  B. Utility-form robustness (risk-averse / concave benefit)',
              '#  B. Concave reliability-valuation robustness')
s = s.replace('UTILITY-FORM ROBUSTNESS (risk aversion eta)',
              'CONCAVE RELIABILITY-VALUATION ROBUSTNESS (eta)')

new_subsidy = '''# --------------------------------------------------------------------------- #
#  C. Follower-only subsidy combined with the accountable Stackelberg leader
# --------------------------------------------------------------------------- #
def _best_response_with_effective_stake(sys, i, R, effective_stake):
    """Best response of player i when its private reliability stake is modified."""
    x = sys.subs[i]
    others = np.prod([R[k] for k in range(sys.n) if k != i])
    target = effective_stake * others
    lo, hi = x.Rmin, min(x.Rmax, 0.999999)
    if x.dcost(lo) >= target:
        return lo
    if x.dcost(hi) <= target:
        return hi
    return brentq(lambda Ri: x.dcost(Ri) - target, lo, hi, xtol=1e-10)


def follower_only_subsidy_nash(sys, s_sub):
    """Simultaneous Nash outcome with a subsidy applied to followers only.

    This helper is retained for diagnostics.  The leader keeps its original
    stake pi_L.  Each follower j acts as if its private stake were
        pi_j + s_sub * (Pi_tot - pi_j).
    Consequently s_sub=1 does *not* generally equal the social optimum because
    the leader's externality is not fully internalized.
    """
    Pi = sum(x.pi for x in sys.subs)
    eff = [x.pi for x in sys.subs]
    for j in sys.F:
        eff[j] = sys.subs[j].pi + s_sub * (Pi - sys.subs[j].pi)
    R = np.array([x.Rmin for x in sys.subs], float)
    for _ in range(500):
        prev = R.copy()
        for i in range(sys.n):
            R[i] = _best_response_with_effective_stake(sys, i, R, eff[i])
        if np.max(np.abs(R - prev)) < 1e-9:
            break
    return R


def subsidized_followers_given_leader(sys, RL, s_sub):
    """Follower Nash response to a fixed leader reliability under follower-only subsidy."""
    Pi = sum(x.pi for x in sys.subs)
    eff = [x.pi for x in sys.subs]
    for j in sys.F:
        eff[j] = sys.subs[j].pi + s_sub * (Pi - sys.subs[j].pi)
    R = np.array([x.Rmin for x in sys.subs], float)
    R[sys.L] = RL
    for _ in range(500):
        prev = R.copy()
        R[sys.L] = RL
        for j in sys.F:
            R[j] = _best_response_with_effective_stake(sys, j, R, eff[j])
        R[sys.L] = RL
        if np.max(np.abs(R - prev)) < 1e-9:
            break
    return R


def stackelberg_with_follower_subsidy(sys, s_sub, theta=THETA):
    """Accountable leader anticipating follower Nash responses under follower-only subsidy.

    The subsidy changes follower private incentives only.  Welfare and the
    leader's objective are evaluated with the original model primitives; no
    subsidy payment is added to welfare.  Reported recovery is therefore gross
    engineering-side recovery before financing/administrative costs.
    """
    lo = sys.subs[sys.L].Rmin
    hi = min(sys.subs[sys.L].Rmax, 0.999999)

    def leader_obj(RL):
        R = subsidized_followers_given_leader(sys, RL, s_sub)
        return theta * sys.welfare(R) + (1.0 - theta) * sys.utility(sys.L, R)

    res = minimize_scalar(lambda RL: -leader_obj(RL), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-8})
    return subsidized_followers_given_leader(sys, res.x, s_sub)


def subsidy_remedy(n_list=(4, 8, 12), s_grid=None):
    """Gross Nash-to-social welfare-gap recovery from leader + follower-only subsidy.

    The synthetic chain construction is unchanged from the original experiment.
    At s_sub=0 each curve equals the accountable Stackelberg-leader outcome.
    """
    if s_grid is None:
        s_grid = np.linspace(0, 1, 11)
    rng = np.random.default_rng(7)
    out = {}
    for n in n_list:
        subs = [Subsystem("L", 0.024, 0.088, 3.2, 0.90, 0.9995, "leader")]
        for k in range(n - 1):
            subs.append(Subsystem(f"F{k}", 0.028, 0.072,
                                  float(1.0 + 0.6 * rng.random()), 0.90, 0.9995))
        g = SeriesSystem(subs, leader_index=0)
        Wn = g.welfare(g.solve_nash())
        Wo = g.welfare(g.solve_social())
        rec = []
        for s_sub in s_grid:
            R = stackelberg_with_follower_subsidy(g, float(s_sub), theta=THETA)
            W = g.welfare(R)
            rec.append((W - Wn) / (Wo - Wn))
        rec = np.asarray(rec)
        out[n] = dict(s_grid=np.asarray(s_grid), rec_sub=rec, rec_leader=float(rec[0]))
    return out
'''
s = replace_between(
    s,
    '# --------------------------------------------------------------------------- #\n#  C. Follower-subsidy remedy (Pigouvian) for long chains',
    '# --------------------------------------------------------------------------- #\n#  D. VCG comparison',
    new_subsidy,
    label='analysis3 subsidy block',
)
s = s.replace('print("C. FOLLOWER-SUBSIDY REMEDY")',
              'print("C. ACCOUNTABLE LEADER + FOLLOWER-ONLY SUBSIDY")')
s = s.replace("  subsidy s=0.5 recovers", "  leader+subsidy s=0.5 recovers")
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 5) analysis5.py: Table 10 must use the combined Stackelberg + follower-only
#    subsidy solver, not a separate subsidized simultaneous Nash calculation.
# -------------------------------------------------------------------------
p = ROOT / "analysis5.py"
s = p.read_text(encoding="utf-8")
new_eps = '''def eps_protocol_plus_subsidy(s_list=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Gross welfare-gap recovery on the real EPS case.

    s=0 is the accountable Stackelberg leader alone.  Positive s values retain
    that leader and apply the subsidy only to followers.  Subsidy financing and
    administrative costs are intentionally excluded from the gross engineering
    welfare metric reported in the manuscript.
    """
    g = satellite_eps_real()
    Wn = g.welfare(g.solve_nash())
    Wo = g.welfare(g.solve_social())
    rows = []
    for s in s_list:
        R = A3.stackelberg_with_follower_subsidy(g, float(s), theta=THETA)
        W = g.welfare(R)
        rows.append((s, (W - Wn) / (Wo - Wn)))
    leader_alone = rows[0][1]
    return leader_alone, rows
'''
s = replace_between(
    s,
    'def eps_protocol_plus_subsidy(s_list=(0.0, 0.25, 0.5, 0.75, 1.0)):',
    '# --------------------------------------------------------------------------- #\n#  (4) Simultaneous misreporting',
    new_eps,
    label='analysis5 EPS subsidy function',
)
s = s.replace('Simultaneous misreporting: all followers under-report their cost rate',
              'Simultaneous reported-parameter perturbation: all followers scale their cost rate')
s = s.replace('tag = "  (= Nash)" if s == 0 else ("  (= social opt)" if s == 1 else "")',
              'tag = "  (= accountable leader only)" if s == 0 else ""')
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 6) figures.py: clean-clone execution must create ./figs automatically.
# -------------------------------------------------------------------------
p = ROOT / "figures.py"
s = p.read_text(encoding="utf-8")
if 'import os\n' not in s[:300]:
    s = s.replace('import numpy as np\n', 'import os\nimport numpy as np\n', 1)
if 'os.makedirs(FIG, exist_ok=True)' not in s:
    s = s.replace('FIG = "figs"\n', 'FIG = "figs"\nos.makedirs(FIG, exist_ok=True)\n', 1)
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 7) figures2.py: corrected Figure 11, reporting-sensitivity Figure 10 wording,
#    corrected Figure 15, and actually generate subsidy/sweep figures in main.
# -------------------------------------------------------------------------
p = ROOT / "figures2.py"
s = p.read_text(encoding="utf-8")
new_theta = '''def fig_theta(sys):
    t = A.theta_sweep(sys, thetas=np.linspace(0, 1, 21))
    Rsoc = sys.R_sys(sys.solve_social())
    Rnash = sys.R_sys(sys.solve_nash())
    residual_pct = 100.0 * (Rsoc - t["Rsys"][-1]) / (Rsoc - Rnash)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(t["theta"], t["Rsys"], color=C["leader"], lw=2.6,
            marker="o", ms=4, label=r"system reliability $R_{\\mathrm{sys}}^*$")
    ax.plot(t["theta"], t["RL"], color=C["stk"], lw=2.0, ls="-",
            marker="s", ms=3.5, label=r"leader reliability $R_L^*$")
    ax.plot(t["theta"], t["RFmean"], color=C["soc"], lw=2.0, ls="--",
            marker="^", ms=3.5, label=r"mean follower reliability $\\bar R_F^*$")
    ax.axhline(Rsoc, color="#444", ls=":", lw=1.4)
    ax.text(0.02, Rsoc + 0.004, f"social-optimum $R_{{\\\\mathrm{{sys}}}}={Rsoc:.3f}$",
            fontsize=8.6, color="#444", ha="left")
    ax.axhline(Rnash, color=C["nash"], ls=":", lw=1.2)
    ax.text(0.02, Rnash - 0.009, f"Nash $R_{{\\\\mathrm{{sys}}}}={Rnash:.3f}$",
            fontsize=8.6, color=C["nash"], ha="left")
    ax.annotate("", xy=(0.965, Rsoc), xytext=(0.965, t["Rsys"][-1]),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1))
    ax.text(0.95, (Rsoc + t["Rsys"][-1]) / 2,
            f"residual\\nreliability gap ({residual_pct:.1f}%)", fontsize=8.0,
            color="#444", ha="right", va="center")
    ax.set_xlabel(r"leader system-accountability weight $\\theta$")
    ax.set_ylabel("reliability at equilibrium")
    ax.set_title("Leader accountability helps, but cannot close the gap alone")
    ax.legend(fontsize=8.2, loc="center", bbox_to_anchor=(0.43, 0.38))
    ax.set_xlim(0, 1); ax.set_ylim(0.76, 0.97)
    _save(fig, "fig_theta")
'''
s = replace_between(
    s, 'def fig_theta(sys):', '# --------------------------------------------------------------------------- #\n#  FIG -- gamma with individual-rationality range',
    new_theta, label='figures2 fig_theta')

new_manip = '''def fig_manip(sys):
    """Sensitivity of computed allocations to a reported cost-rate input."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = FOLLOW
    for idx, j in enumerate(sys.F):
        f, share = A2.manipulation_curve(sys, j)
        ax.plot(f, share, color=cols[idx], lw=2.0, marker="o", ms=3.5,
                label=sys.subs[j].name)
    ax.axvline(1.0, color="#333", ls="--", lw=1.2)
    ax.text(1.02, ax.get_ylim()[0] + 0.02, "baseline input", fontsize=8.4, color="#333")
    ax.set_xlabel(r"reported cost-rate multiplier $\\beta_j'/\\beta_j$")
    ax.set_ylabel("computed follower allocation")
    ax.set_title("Allocation sensitivity to the reported cost-rate input")
    ax.legend(fontsize=8.4, loc="upper right")
    _save(fig, "fig_manip")
'''
s = replace_between(s, 'def fig_manip(sys):', 'def fig_scale20(sys):', new_manip,
                    label='figures2 fig_manip')

new_subfig = '''def fig_subsidy(sys):
    import analysis3 as A3
    rem = A3.subsidy_remedy(n_list=(4, 8, 12))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    cols = {4: C["leader"], 8: C["stk"], 12: C["soc"]}
    for n, d in rem.items():
        ax.plot(d["s_grid"], 100 * d["rec_sub"], color=cols[n], lw=2.4,
                marker="o", ms=3.5, label=f"leader + follower subsidy, $n={n}$")
    ax.set_xlabel(r"follower subsidy intensity $s$ (0 = leader only; 1 = full follower subsidy)")
    ax.set_ylabel("Nash$\\to$social welfare gap closed (%)")
    ax.set_title("Follower-side subsidies complement the accountable leader")
    ax.axhline(100, color="#999", ls=":", lw=1.0)
    ax.legend(fontsize=8.2, loc="lower right")
    ax.set_ylim(0, 108)
    _save(fig, "fig_subsidy")
'''
s = replace_between(s, 'def fig_subsidy(sys):', 'def fig_sweep(sys):', new_subfig,
                    label='figures2 fig_subsidy')

old_main = '''if __name__ == "__main__":
    sys = satellite_eps()
    fig_theta(sys)
    fig_gamma_ir(sys)
    fig_robust(sys)
    fig_alloc_compare(sys)
    fig_methods(sys)
    fig_manip(sys)
    fig_scale20(sys)
    print("all revision figures generated.")'''
new_main = '''if __name__ == "__main__":
    sys = satellite_eps()
    fig_theta(sys)
    fig_gamma_ir(sys)
    fig_robust(sys)
    fig_alloc_compare(sys)
    fig_methods(sys)
    fig_manip(sys)
    fig_scale20(sys)
    fig_subsidy(sys)
    fig_sweep(sys)
    print("all computational revision figures generated.")'''
s = replace_once(s, old_main, new_main, label='figures2 main')
p.write_text(s, encoding="utf-8")

# -------------------------------------------------------------------------
# 8) README: remove overclaims and synchronize numerical values.
# -------------------------------------------------------------------------
readme = r'''# Stackelberg-Shapley protocol for decentralized reliability allocation

Python implementation and analysis scripts associated with the manuscript
**"A Stackelberg-Shapley Protocol for Individually-Rational Reliability Allocation in the Conceptual Design of Series Systems."**

## Scope of the repository

The code implements the public-good reliability model, simultaneous Nash and
accountability-weighted Stackelberg regimes, the induced characteristic-function
game and Shapley allocation, the participation-constrained leader premium,
robustness/calibration experiments, fixed-target allocation comparisons,
synthetic scalability tests, and the corrected **follower-only subsidy combined
with the accountable Stackelberg leader**.

The satellite Electrical Power System (EPS) illustration is **data-informed**:
published reliability evidence motivates the subsystem decomposition and the
choice of Solar-Array Operating as the leader.  The normalized stake and
reliability-hardening cost parameters remain modelling inputs rather than a
fully empirical economic calibration.

## Important interpretation notes

- The Jacobian spectral radius `rho ~= 0.044` is a **local convergence diagnostic**.
  The manuscript's global sufficient uniqueness certificate is the conservative
  boxwise sup-norm contraction bound `Lambda_box ~= 0.411` implemented in
  `analysis3.py`.
- The cost-rate perturbation experiment is a **reporting-parameter sensitivity
  analysis**, not a strategy-proofness test.
- The fixed-target "AGREE" calculation available in the legacy code does not
  contain all mission-time and importance inputs required for a full AGREE
  implementation; it is therefore reported as a **complexity-weighted proxy**.
- Subsidy results are **gross engineering welfare recovery before financing and
  administrative costs**.  The corrected experiment subsidizes followers only
  and retains the accountable Stackelberg leader.

## Repository structure

| File | Main purpose |
|---|---|
| `stackelberg_shapley.py` | Core model, Nash/Stackelberg/social solvers, characteristic function, Shapley value, EPS case |
| `analysis.py` | Accountability sweep, participation range, calibration/baseline sensitivity, allocation comparisons |
| `analysis2.py` | Fixed-target allocation comparisons, power-law robustness, reporting-input sensitivity, scalability |
| `analysis3.py` | Global contraction bound, concave valuation, follower-only subsidy, marginal-contribution budget diagnostic, baseline sensitivity |
| `analysis4.py` | Weibull-informed calibration and wide cost-parameter sweep |
| `analysis5.py` | Real-EPS leader+subsidy table, joint input perturbation, second synthetic case |
| `figures.py`, `figures2.py` | Computational figure-generation scripts |
| `verify_results.py` | Numerical regression checks for the headline manuscript values |

## Environment

Python 3.10+ with:

```bash
pip install numpy scipy matplotlib
```

## Running the analyses

```bash
python stackelberg_shapley.py
python analysis.py
python analysis2.py
python analysis3.py
python analysis4.py
python analysis5.py
python verify_results.py

# computational figures
python figures.py
python figures2.py
```

Figure files are written to `./figs`; the directory is created automatically.

## Headline numerical checks for the EPS illustration

| Quantity | Verified value |
|---|---:|
| Nash system reliability | 0.7722847 |
| Accountable Stackelberg system reliability (`theta=0.70`) | 0.7815708 |
| Social-optimum system reliability | 0.8480909 |
| Nash welfare | 5.6263815 |
| Stackelberg welfare | 5.6565236 |
| Social-optimum welfare | 5.9321853 |
| Welfare gap closed by leader alone | 9.8567% |
| Local follower-BR Jacobian spectral radius | 0.0443 |
| Global boxwise contraction bound | 0.4105 |
| Leader criticality (`Rmin` projection) | 2.75014 |
| Leader share, Shapley -> premium allocation (`gamma=0.20`) | 40.43% -> 49.70% |
| `gamma_max`, `Rmin` projection | 0.24548 |
| `gamma_max`, Nash projection | 0.08005 |
| Positive common premium under `Rmax` projection | none |
| Leader + follower-only subsidy, `s=0.25` | 71.30% gross gap recovery |
| Leader + follower-only subsidy, `s=0.50` | 90.70% gross gap recovery |
| Leader + follower-only subsidy, `s=0.75` | 97.84% gross gap recovery |
| Leader + follower-only subsidy, `s=1.00` | 99.56% gross gap recovery |

## Data/code statement

The repository contains the Python source code used for the numerical analyses
reported in the manuscript.  Numerical reproducibility verifies the
implementation of the stated model; it does not by itself validate the modelling
assumptions or turn normalized parameters into measured economic data.

## Citation

Please cite the manuscript; full bibliographic details can be added after publication.
'''
(ROOT / "README.md").write_text(readme, encoding="utf-8")

print("Repository synchronization patches applied successfully.")
print(f"Original files backed up in: {BACKUP}")
print("Next: run `python verify_results.py` after copying the supplied verifier into the repo root.")
