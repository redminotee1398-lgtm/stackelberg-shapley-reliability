"""
Additional publication-quality figures for the revised paper.
Reuses the global style and palette defined in figures.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from figures import C, FOLLOW, _save, THETA, GAMMA
from stackelberg_shapley import satellite_eps_real as satellite_eps
import analysis as A


# --------------------------------------------------------------------------- #
#  FIG -- theta sensitivity (leader system-accountability)
# --------------------------------------------------------------------------- #
def fig_theta(sys):
    t = A.theta_sweep(sys, thetas=np.linspace(0, 1, 21))
    Rsoc = sys.R_sys(sys.solve_social())
    Rnash = sys.R_sys(sys.solve_nash())
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(t["theta"], t["Rsys"], color=C["leader"], lw=2.6,
            marker="o", ms=4, label=r"system reliability $R_{\mathrm{sys}}^*$")
    ax.plot(t["theta"], t["RL"], color=C["stk"], lw=2.0, ls="-",
            marker="s", ms=3.5, label=r"leader reliability $R_L^*$")
    ax.plot(t["theta"], t["RFmean"], color=C["soc"], lw=2.0, ls="--",
            marker="^", ms=3.5, label=r"mean follower reliability $\bar R_F^*$")
    ax.axhline(Rsoc, color="#444", ls=":", lw=1.4)
    ax.text(0.02, Rsoc + 0.004, "social-optimum $R_{\\mathrm{sys}}=0.848$",
            fontsize=8.6, color="#444", ha="left")
    ax.axhline(Rnash, color=C["nash"], ls=":", lw=1.2)
    ax.text(0.02, Rnash - 0.009, "Nash $R_{\\mathrm{sys}}=0.772$",
            fontsize=8.6, color=C["nash"], ha="left")
    # residual-gap annotation at theta=1
    ax.annotate("", xy=(0.965, Rsoc), xytext=(0.965, t["Rsys"][-1]),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1))
    ax.text(0.95, (Rsoc + t["Rsys"][-1]) / 2,
            "residual\ngap (92%)", fontsize=8.0, color="#444",
            ha="right", va="center")
    ax.set_xlabel(r"leader system-accountability weight $\theta$")
    ax.set_ylabel("reliability at equilibrium")
    ax.set_title("Leader accountability helps, but cannot close the gap alone")
    ax.legend(fontsize=8.8, loc="center", bbox_to_anchor=(0.42, 0.34))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.76, 0.97)
    _save(fig, "fig_theta")


# --------------------------------------------------------------------------- #
#  FIG -- gamma with individual-rationality range
# --------------------------------------------------------------------------- #
def fig_gamma_ir(sys):
    g = A.gamma_ir_range(sys)
    phi, kap, tot = g["phi"], g["kappa_L"], g["phi"].sum()
    gmax = g["gamma_max"]
    gammas = np.linspace(0, 1, 121)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    # leader share
    shareL = (phi[sys.L] + gammas * kap) / tot
    ax.plot(gammas, 100 * shareL, color=C["leader"], lw=2.8,
            label="Leader (Solar-Array Op.)")
    cols = FOLLOW
    for idx, j in enumerate(sys.F):
        sh = (phi[j] - gammas * kap / (sys.n - 1)) / tot
        ax.plot(gammas, 100 * sh, color=cols[idx], lw=1.8,
                label=sys.subs[j].name)
        # IR floor as % of total
        floor_pct = 100 * g["floors"][j] / tot
        ax.axhline(floor_pct, color=cols[idx], ls=":", lw=1.0, alpha=0.6)

    # IR-feasible region
    ax.axvspan(0, gmax, color="#1E8449", alpha=0.08)
    ax.axvline(gmax, color="#1E8449", lw=1.6, ls="--")
    ax.text(gmax - 0.01, 8, f"$\\gamma_{{\\max}}={gmax:.2f}$\n(follower IR binds)",
            ha="right", va="bottom", fontsize=8.6, color="#1E8449")
    ax.axvline(GAMMA, color="#333", lw=1.4, ls="-")
    ax.text(GAMMA - 0.015, 30, f"chosen $\\gamma={GAMMA}$",
            fontsize=8.6, color="#333", rotation=90, va="center")

    ax.set_xlabel(r"transfer parameter $\gamma$")
    ax.set_ylabel("allocated share (% of $v(N)$)")
    ax.set_title("The $\\gamma$ lever and its participation limit")
    ax.legend(fontsize=8.4, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 78)
    _save(fig, "fig_gamma_ir")


# --------------------------------------------------------------------------- #
#  FIG -- robustness (beta scale + characteristic-function baseline)
# --------------------------------------------------------------------------- #
def fig_robust(sys):
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # (a) beta scaling
    rb = A.robustness_beta(scales=(0.75, 0.875, 1.0, 1.125, 1.25))
    sc = [r["scale"] for r in rb]
    share = [r["leader_share"] for r in rb]
    poa = [100 * (r["PoA"] - 1) for r in rb]
    a0 = ax[0]
    a0.plot(sc, share, color=C["leader"], lw=2.4, marker="o", ms=5,
            label="leader Stk-Shapley share")
    a0.set_ylim(50, 65)
    a0.set_xlabel(r"cost-rate scaling $\beta_i \to s\,\beta_i$")
    a0.set_ylabel("leader share (%)", color=C["leader"])
    a0.tick_params(axis="y", labelcolor=C["leader"])
    a0.set_title("(a)  Robustness to cost calibration")
    a0b = a0.twinx()
    a0b.plot(sc, poa, color=C["accent"], lw=2.0, ls="--", marker="s", ms=4,
             label="welfare loss (PoA$-1$)")
    a0b.set_ylabel("welfare loss % (PoA$-1$)", color=C["accent"])
    a0b.tick_params(axis="y", labelcolor=C["accent"])
    a0b.grid(False)
    a0.axvspan(0.75, 1.25, color="#999", alpha=0.05)

    # (b) characteristic-function baseline
    rbb = A.robustness_baseline(sys)
    names = [r["baseline"] for r in rbb]
    kap = [r["kappa_L"] for r in rbb]
    lstk = [r["leader_stk"] for r in rbb]
    x = np.arange(len(names))
    a1 = ax[1]
    b = a1.bar(x - 0.2, kap, 0.4, color=C["stk"], edgecolor="k", linewidth=0.6,
               label=r"leader criticality $\kappa_L$")
    a1.set_ylabel(r"$\kappa_L$", color=C["stk"])
    a1.tick_params(axis="y", labelcolor=C["stk"])
    a1.set_xticks(x)
    a1.set_xticklabels([r"$R^{\min}$", "Nash", r"$R^{\max}$"])
    a1.set_xlabel("non-member baseline in $v(S)$")
    a1.set_title("(b)  Robustness to $v(S)$ convention")
    a1.set_ylim(0, 4)
    a1b = a1.twinx()
    a1b.plot(x, lstk, color=C["leader"], lw=2.2, marker="o", ms=6,
             label="leader Stk share")
    a1b.set_ylabel("leader Stk share (%)", color=C["leader"])
    a1b.tick_params(axis="y", labelcolor=C["leader"])
    a1b.set_ylim(50, 65)
    a1b.grid(False)
    for xi, val in zip(x, lstk):
        a1b.text(xi, val + 0.5, f"{val:.0f}%", ha="center", fontsize=8.2,
                 color=C["leader"])
    _save(fig, "fig_robust")


# --------------------------------------------------------------------------- #
#  FIG -- allocation-rule comparison with IR floors
# --------------------------------------------------------------------------- #
def fig_alloc_compare(sys):
    rules, floors, ir = A.allocation_baselines(sys)
    names = list(rules.keys())
    x = np.arange(sys.n)
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    rule_cols = {"Egalitarian": "#AEB6BF", "Proportional": "#5D6D7E",
                 "Shapley": C["stk"], "Stk-Shapley": C["leader"]}
    for k, name in enumerate(names):
        ax.bar(x + (k - 1.5) * w, rules[name], w, label=name,
               color=rule_cols[name], edgecolor="k", linewidth=0.5)
    # IR floors
    for i in range(sys.n):
        ax.hlines(floors[i], x[i] - 2 * w, x[i] + 2 * w, color="k",
                  lw=1.8, ls=":")
    ax.plot([], [], color="k", lw=1.8, ls=":", label="stand-alone floor $v(\\{i\\})$")
    labels = [s.name.replace(" Operating", "\nOperating").replace(" (L)", "\n(Leader)")
              .replace(" / Cell", "\n/ Cell").replace("Electrical ", "")
              .replace("Solar-Array ", "SA-") for s in sys.subs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("allocated value")
    ax.set_title("Allocation rules vs. individual-rationality floors")
    ax.legend(fontsize=8.6, loc="upper right", ncol=1)
    _save(fig, "fig_alloc_compare")


import analysis2 as A2


def fig_methods(sys):
    """Total design cost to hit a common system-reliability target, across
    classic allocation methods vs the cost-optimal allocation."""
    rows = A2.compare_methods(sys, R_target=0.95)
    names = [r["name"] for r in rows]
    costs = [r["cost"] for r in rows]
    cols = ["#AEB6BF", "#5D6D7E", "#B9770E", C["soc"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(names, costs, color=cols, edgecolor="k", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("total design cost to reach $R_{\\mathrm{sys}}=0.95$  (log)")
    ax.set_title("Classic allocation heuristics are cost-inefficient")
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c * 1.15, f"{c:.0f}",
                ha="center", fontsize=9)
    opt = costs[-1]
    ax.text(0.5, 0.92, f"cost-optimal = {opt:.0f};  heuristics cost "
            f"{costs[0]/opt:.1f}$\\times$ to {max(costs)/opt:.0f}$\\times$ more",
            transform=ax.transAxes, ha="center", fontsize=8.6, color="#333")
    _save(fig, "fig_methods")


def fig_manip(sys):
    """Manipulation incentive: a follower's post-transfer share as it misreports
    its cost rate beta. Not flat => the rule is not strategy-proof."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = FOLLOW
    for idx, j in enumerate(sys.F):
        f, share = A2.manipulation_curve(sys, j)
        ax.plot(f, share, color=cols[idx], lw=2.0, marker="o", ms=3.5,
                label=sys.subs[j].name)
    ax.axvline(1.0, color="#333", ls="--", lw=1.2)
    ax.text(1.02, ax.get_ylim()[0] + 0.02, "truthful", fontsize=8.4, color="#333")
    ax.set_xlabel(r"reported cost rate $\beta_j'/\beta_j$ (misreport factor)")
    ax.set_ylabel("follower's allocated share")
    ax.set_title("The transfer is budget-balanced, hence not strategy-proof")
    ax.legend(fontsize=8.4, loc="upper right")
    _save(fig, "fig_manip")


def fig_scale20(sys):
    rows = A2.scalability(ns=range(2, 21))
    ns = [r["n"] for r in rows]
    poa = [r["poa"] for r in rows]
    share = [r["leader_share"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ns, poa, color=C["accent"], lw=2.6, marker="o", ms=4,
            label="price of anarchy")
    ax.set_xlabel("number of series subsystems $n$")
    ax.set_ylabel("price of anarchy $W_{\\mathrm{soc}}/W_{\\mathrm{Nash}}$",
                  color=C["accent"])
    ax.tick_params(axis="y", labelcolor=C["accent"])
    ax.set_title("Scalability: a single leader fails in long chains")
    ax.axhspan(1.0, 1.10, color="#1E8449", alpha=0.08)
    ax.text(14, 1.18, "conceptual-design\nregime (few majors)",
            fontsize=8.0, color="#1E8449", ha="center")
    ax2 = ax.twinx()
    sh_n = [s for s in share if s == s]
    sh_x = [n for n, s in zip(ns, share) if s == s]
    ax2.plot(sh_x, sh_n, color=C["leader"], lw=2.2, ls="--", marker="s", ms=4,
             label="leader Shapley share")
    ax2.set_ylabel("leader Shapley share (%)", color=C["leader"])
    ax2.tick_params(axis="y", labelcolor=C["leader"])
    ax2.grid(False)
    _save(fig, "fig_scale20")


def fig_subsidy(sys):
    """Demonstrated remedy: a Pigouvian follower subsidy recovers the welfare
    gap across chain lengths, unlike the single-leader lever."""
    import analysis3 as A3
    rem = A3.subsidy_remedy(n_list=(4, 8, 12))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    cols = {4: C["leader"], 8: C["stk"], 12: C["soc"]}
    for n, d in rem.items():
        ax.plot(d["s_grid"], 100 * d["rec_sub"], color=cols[n], lw=2.4,
                marker="o", ms=3.5, label=f"subsidy, $n={n}$")
        ax.scatter([0], [100 * d["rec_leader"]], color=cols[n], marker="x",
                   s=70, zorder=5)
    ax.scatter([], [], color="#333", marker="x", s=70,
               label="single leader (at $s=0$)")
    ax.set_xlabel(r"follower subsidy intensity $s$ (0 = Nash, 1 = full Pigouvian)")
    ax.set_ylabel("Nash$\\to$social welfare gap closed (%)")
    ax.set_title("A follower subsidy scales where a single leader cannot")
    ax.axhline(100, color="#999", ls=":", lw=1.0)
    ax.legend(fontsize=8.6, loc="center right")
    ax.set_ylim(0, 108)
    _save(fig, "fig_subsidy")


def fig_sweep(sys):
    """Heatmap: leader's post-transfer share is stable across a wide (alpha,beta)
    grid -- the allocative conclusion is not a calibration artefact."""
    import analysis4 as A4
    a_scales = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
    b_scales = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    Z = np.zeros((len(b_scales), len(a_scales)))
    for i, bs in enumerate(b_scales):
        for j, as_ in enumerate(a_scales):
            r = A4.alpha_beta_sweep(a_scales=[as_], b_scales=[bs])[0]
            Z[i, j] = r["share"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto",
                   vmin=46, vmax=54)
    ax.set_xticks(range(len(a_scales))); ax.set_xticklabels([f"{x:g}" for x in a_scales])
    ax.set_yticks(range(len(b_scales))); ax.set_yticklabels([f"{x:g}" for x in b_scales])
    ax.set_xlabel(r"cost-scale multiplier on $\alpha_i$")
    ax.set_ylabel(r"cost-rate multiplier on $\beta_i$")
    ax.set_title("Leader's allocated share (%) across a wide cost grid")
    for i in range(len(b_scales)):
        for j in range(len(a_scales)):
            ax.text(j, i, f"{Z[i,j]:.1f}", ha="center", va="center",
                    color="w" if Z[i, j] < 50.5 else "k", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax); cb.set_label("leader Stk--Shapley share (%)")
    _save(fig, "fig_sweep")


if __name__ == "__main__":
    sys = satellite_eps()
    fig_theta(sys)
    fig_gamma_ir(sys)
    fig_robust(sys)
    fig_alloc_compare(sys)
    fig_methods(sys)
    fig_manip(sys)
    fig_scale20(sys)
    print("all revision figures generated.")
