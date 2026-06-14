"""
Publication-quality figures for the Stackelberg-Shapley reliability paper.
Each figure is saved to ./figs at 300 dpi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager

from stackelberg_shapley import Subsystem, SeriesSystem, satellite_eps_real as satellite_eps

# ----------------------------------------------------------------------------- #
#  Global publication style
# ----------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# colour-blind-safe palette
C = {
    "leader":  "#C0392B",   # deep red
    "f1":      "#2471A3",   # blue
    "f2":      "#1E8449",   # green
    "f3":      "#B9770E",   # amber
    "nash":    "#7F8C8D",   # grey
    "stk":     "#2471A3",   # blue
    "soc":     "#1E8449",   # green
    "accent":  "#6C3483",   # purple
    "ink":     "#222222",
}
FOLLOW = [C["f1"], C["f2"], C["f3"]]
THETA, GAMMA = 0.70, 0.20
FIG = "figs"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- #
#  FIG 1 -- cost / reliability and marginal cost
# ----------------------------------------------------------------------------- #
def fig_cost(sys):
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    R = np.linspace(0.80, 0.992, 400)
    cols = [C["leader"]] + FOLLOW
    for k, s in enumerate(sys.subs):
        ax[0].plot(R, [s.cost(r) for r in R], color=cols[k], lw=2.0,
                   label=s.name.replace(" (L)", ""))
        ax[1].plot(R, [s.dcost(r) for r in R], color=cols[k], lw=2.0)
    ax[0].set_xlabel(r"subsystem reliability $R_i$")
    ax[0].set_ylabel(r"design cost $C_i(R_i)$  [M\$]")
    ax[0].set_title("(a)  Convex cost--reliability curves")
    ax[0].legend(fontsize=8.5, loc="upper left")
    ax[0].set_ylim(0, 30)
    ax[0].axvspan(0.93, 0.965, color="#999", alpha=0.10)
    ax[0].text(0.9475, 27.5, "operating\nregion", ha="center", va="top",
               fontsize=7.8, color="#555", style="italic")
    ax[1].set_xlabel(r"subsystem reliability $R_i$")
    ax[1].set_ylabel(r"marginal cost $C_i'(R_i)$")
    ax[1].set_title("(b)  Steeply rising marginal cost")
    ax[1].set_yscale("log")
    ax[1].axvspan(0.93, 0.965, color="#999", alpha=0.10)
    for a in ax:
        a.set_xlim(0.80, 0.992)
    _save(fig, "fig_cost")


# ----------------------------------------------------------------------------- #
#  FIG 2 -- per-subsystem provision gap (Nash vs Social): free-riding map
# ----------------------------------------------------------------------------- #
def fig_provision(sys):
    Rn, Ro = sys.solve_nash(), sys.solve_social()
    labels = [s.name.replace(" Operating", "").replace(" (L)", "\n(Leader)").replace(" / Cell","").replace("Electrical ","").replace("Solar-Array ","SA-") for s in sys.subs]
    stakes = [s.pi for s in sys.subs]
    x = np.arange(sys.n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    cols = [C["leader"]] + FOLLOW
    b1 = ax.bar(x-w/2, Rn, w, color="#AEB6BF", edgecolor="k", linewidth=0.6,
                label=r"Nash $R_i$ (decentralised)")
    b2 = ax.bar(x+w/2, Ro, w, color=cols, edgecolor="k", linewidth=0.6,
                label=r"Social-optimum $R_i$")
    # provision-gap arrows
    for i in range(sys.n):
        gap = Ro[i]-Rn[i]
        ax.annotate("", xy=(x[i]+w/2, Ro[i]), xytext=(x[i]-w/2, Rn[i]),
                    arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.3, alpha=0.85))
        ax.text(x[i]-0.06, (Rn[i]+Ro[i])/2, f"$+{gap:.3f}$",
                ha="right", va="center", fontsize=8.4, color=C["accent"])
        ax.text(x[i], 0.838, f"stake $\\pi={stakes[i]:.1f}$", ha="center",
                fontsize=8.2, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel(r"chosen reliability $R_i$")
    ax.set_ylim(0.83, 0.99)
    ax.set_title("Provision gap: free-riding concentrates in low-stake followers", pad=8)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    _save(fig, "fig_provision")


# ----------------------------------------------------------------------------- #
#  FIG 3 -- regime comparison: reliability & welfare (Nash / Stackelberg / Social)
# ----------------------------------------------------------------------------- #
def fig_regimes(sys):
    Rn, Rs, Ro = sys.solve_nash(), sys.solve_stackelberg(theta=THETA), sys.solve_social()
    regimes = ["Nash\n(decentralised)", "Stackelberg\n(leadership)", "Social\noptimum"]
    Rsys = [sys.R_sys(Rn), sys.R_sys(Rs), sys.R_sys(Ro)]
    W = [sys.welfare(Rn), sys.welfare(Rs), sys.welfare(Ro)]
    cols = [C["nash"], C["stk"], C["soc"]]

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.7))
    x = np.arange(3)
    b1 = ax[0].bar(x, Rsys, color=cols, width=0.6, edgecolor="k", linewidth=0.6)
    for r, v in zip(b1, Rsys):
        ax[0].text(r.get_x()+r.get_width()/2, v+0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax[0].set_ylim(0.74, 0.87)
    ax[0].set_ylabel(r"system reliability $R_{\mathrm{sys}}$")
    ax[0].set_xticks(x); ax[0].set_xticklabels(regimes, fontsize=9)
    ax[0].set_title("(a)  System reliability by regime")
    # annotate strategic advantage
    ax[0].annotate("", xy=(1, Rsys[1]), xytext=(0, Rsys[0]),
                   arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.4))
    ax[0].text(0.5, max(Rsys[0], Rsys[1])+0.012,
               r"$\Delta_{\mathrm{strategic}}$"+f"={Rsys[1]-Rsys[0]:.4f}",
               color=C["accent"], ha="center", fontsize=9)

    b2 = ax[1].bar(x, W, color=cols, width=0.6, edgecolor="k", linewidth=0.6)
    for r, v in zip(b2, W):
        ax[1].text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].set_ylim(5.6, 6.2)
    ax[1].set_ylabel(r"total welfare $W=\sum_i u_i$")
    ax[1].set_xticks(x); ax[1].set_xticklabels(regimes, fontsize=9)
    ax[1].set_title("(b)  Welfare and Price of Anarchy")
    poa = W[2]/W[0]
    ax[1].annotate("", xy=(2, W[2]), xytext=(2, W[0]),
                   arrowprops=dict(arrowstyle="<->", color=C["leader"], lw=1.4))
    ax[1].text(2.05, (W[0]+W[2])/2, f"PoA = {poa:.3f}\n({100*(poa-1):.1f}% loss)",
               color=C["leader"], fontsize=8.6, va="center")
    _save(fig, "fig_regimes")


# ----------------------------------------------------------------------------- #
#  FIG 4 -- best-response iteration convergence + leader objective
# ----------------------------------------------------------------------------- #
def fig_convergence(sys):
    Rfin, hist = sys.solve_nash(record=True)
    hist = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    cols = [C["leader"]] + FOLLOW
    labels = [s.name.replace(" (L)", "") for s in sys.subs]
    for i in range(sys.n):
        ax[0].plot(range(len(hist)), hist[:, i], marker="o", ms=4, lw=1.8,
                   color=cols[i], label=labels[i])
    ax[0].set_xlabel("best-response iteration $k$")
    ax[0].set_ylabel(r"reliability $R_i^{(k)}$")
    ax[0].set_title("(a)  Nash best-response convergence")
    ax[0].legend(fontsize=8.3, loc="lower right")
    ax[0].set_xlim(0, min(8, len(hist)-1))

    # leader effective objective over R_L
    RL = np.linspace(sys.subs[sys.L].Rmin, 0.984, 90)
    obj = [sys.leader_objective(r, theta=THETA) for r in RL]
    Rs = sys.solve_stackelberg(theta=THETA)
    ax[1].plot(RL, obj, color=C["accent"], lw=2.2)
    ax[1].scatter(Rs[0], sys.leader_objective(Rs[0], theta=THETA),
                  color=C["leader"], marker="*", s=240, edgecolor="k", zorder=6,
                  label=fr"$R_L^*={Rs[0]:.3f}$")
    ax[1].axvline(sys.solve_nash()[0], color=C["nash"], ls="--", lw=1.3,
                  label=fr"Nash $R_L={sys.solve_nash()[0]:.3f}$")
    ax[1].set_xlabel(r"leader reliability $R_L$")
    ax[1].set_ylabel(r"leader objective $\theta W+(1-\theta)u_L$")
    ax[1].set_title("(b)  Leader's bilevel optimisation")
    ax[1].set_xlim(sys.subs[sys.L].Rmin, 0.984)
    ax[1].set_ylim(2.0, 5.1)
    ax[1].legend(fontsize=8.5, loc="lower center")
    _save(fig, "fig_convergence")


# ----------------------------------------------------------------------------- #
#  FIG 5 -- allocation shares vs gamma (managerial lever)
# ----------------------------------------------------------------------------- #
def fig_gamma(sys):
    phi, v = sys.shapley()
    N = set(range(sys.n))
    kappa = v(N) - v(N - {sys.L})
    tot = phi.sum()
    gammas = np.linspace(0, 1, 60)
    sharesL, sharesF = [], [[] for _ in sys.F]
    for gm in gammas:
        pl = phi[sys.L] + gm*kappa
        sharesL.append(pl/tot)
        for idx, j in enumerate(sys.F):
            sharesF[idx].append((phi[j] - gm*kappa/(sys.n-1))/tot)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(gammas, sharesL, color=C["leader"], lw=2.6, label="Leader (Solar/PCU)")
    fl = [s.name for s in sys.subs]
    for idx, j in enumerate(sys.F):
        ax.plot(gammas, sharesF[idx], color=FOLLOW[idx], lw=2.0, label=fl[j])
    ax.axvline(GAMMA, color=C["ink"], ls=":", lw=1.3)
    ax.text(GAMMA+0.01, 0.62, fr"operating point $\gamma={GAMMA}$",
            rotation=90, va="top", fontsize=8.5)
    ax.fill_between(gammas, 0, 1, where=(gammas <= 0.0), alpha=0)  # noop keeps limits
    ax.set_xlabel(r"strategic-premium parameter $\gamma$")
    ax.set_ylabel("share of allocated system value")
    ax.set_title(r"Allocation shares vs. the managerial lever $\gamma$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.72)
    ax.legend(fontsize=8.6, loc="center left")
    ax.text(0.02, 0.02, r"$\gamma=0$: pure Shapley   $\rightarrow$   "
                        r"$\gamma=1$: full strategic premium",
            transform=ax.transAxes, fontsize=8, style="italic", color="#555")
    _save(fig, "fig_gamma")


# ----------------------------------------------------------------------------- #
#  FIG 6 -- Shapley vs Stackelberg-Shapley allocation
# ----------------------------------------------------------------------------- #
def fig_shapley(sys):
    m = sys.metrics(GAMMA, theta=THETA)
    labels = [s.name.replace(" Operating", "").replace(" (L)", "\n(Leader)").replace(" / Cell","").replace("Electrical ","").replace("Solar-Array ","SA-") for s in sys.subs]
    x = np.arange(sys.n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    b1 = ax.bar(x-w/2, m["phi"], w, color="#AEB6BF", edgecolor="k", linewidth=0.6,
                label=r"Shapley value $\phi_i$")
    cols = [C["leader"]] + FOLLOW
    b2 = ax.bar(x+w/2, m["phi_stk"], w, color=cols, edgecolor="k", linewidth=0.6,
                label=r"Stackelberg--Shapley $\phi_i^{\mathrm{Stk}}$")
    for rects, vals in [(b1, m["phi"]), (b2, m["phi_stk"])]:
        for r, vv in zip(rects, vals):
            ax.text(r.get_x()+r.get_width()/2, vv+0.03, f"{vv:.2f}",
                    ha="center", fontsize=8.2)
    # arrows showing redistribution
    ax.annotate("", xy=(0+w/2, m["phi_stk"][0]-0.05), xytext=(0+w/2, m["phi"][0]+0.05),
                arrowprops=dict(arrowstyle="->", color=C["leader"], lw=1.6))
    ax.text(0.55, (m["phi"][0]+m["phi_stk"][0])/2, "+premium",
            color=C["leader"], fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("allocated system value")
    ax.set_title("Cooperative allocation: strategic premium re-weights the leader")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(m["phi_stk"])*1.18)
    _save(fig, "fig_shapley")


# ----------------------------------------------------------------------------- #
#  FIG 7 -- welfare landscape over (R_L, R_F) with the three operating points
# ----------------------------------------------------------------------------- #
def fig_landscape():
    subs = [Subsystem("Leader", 0.020, 0.090, 3.4, 0.80, 0.999, "leader"),
            Subsystem("Follower", 0.030, 0.075, 1.7, 0.80, 0.999)]
    g = SeriesSystem(subs, 0)
    rl = np.linspace(0.90, 0.978, 200)
    rf = np.linspace(0.90, 0.978, 200)
    RL, RF = np.meshgrid(rl, rf)
    W = np.zeros_like(RL)
    for a in range(RL.shape[0]):
        for b in range(RL.shape[1]):
            W[a, b] = g.welfare([RL[a, b], RF[a, b]])
    Rn = g.solve_nash(); Rs = g.solve_stackelberg(theta=THETA); Ro = g.solve_social()

    fig, ax = plt.subplots(figsize=(5.9, 4.9))
    vmax = W.max()
    vmin = np.percentile(W, 5)
    levels = np.linspace(vmin, vmax, 24)
    cs = ax.contourf(RL, RF, np.clip(W, vmin, vmax), levels=levels,
                     cmap="viridis", alpha=0.95, extend="min")
    ax.contour(RL, RF, np.clip(W, vmin, vmax), levels=levels[::3],
               colors="white", linewidths=0.4, alpha=0.55)
    cb = fig.colorbar(cs, ax=ax); cb.set_label(r"total welfare $W$")
    pts = [(Rn, "Nash", C["nash"], "o"), (Rs, "Stackelberg", C["leader"], "*"),
           (Ro, "Social opt.", "#FFFFFF", "P")]
    for R, name, col, mk in pts:
        ax.scatter(R[0], R[1], color=col, marker=mk,
                   s=240 if mk == "*" else 120, edgecolor="k", zorder=6, label=name)
    # gradient arrow Nash -> Social
    ax.annotate("", xy=(Ro[0], Ro[1]), xytext=(Rn[0], Rn[1]),
                arrowprops=dict(arrowstyle="->", color="w", lw=1.6, alpha=0.9))
    ax.set_xlabel(r"leader reliability $R_L$")
    ax.set_ylabel(r"follower reliability $R_F$")
    ax.set_title("Welfare landscape and operating points")
    ax.legend(loc="lower left", fontsize=8.6)
    _save(fig, "fig_landscape")


# ----------------------------------------------------------------------------- #
#  FIG 8 -- scalability: Price of Anarchy & leader share vs number of subsystems
# ----------------------------------------------------------------------------- #
def fig_scalability():
    ns = list(range(2, 10))
    poa, recov, leadshare = [], [], []
    for n in ns:
        subs = [Subsystem("L", 0.020, 0.090, 3.4, 0.80, 0.999, "leader")]
        for j in range(n-1):
            subs.append(Subsystem(f"F{j}", 0.028, 0.072, 1.5, 0.80, 0.999))
        g = SeriesSystem(subs, 0)
        Rn, Rs, Ro = g.solve_nash(), g.solve_stackelberg(theta=THETA), g.solve_social()
        wn, ws, wo = g.welfare(Rn), g.welfare(Rs), g.welfare(Ro)
        poa.append(wo/wn)
        recov.append(100*(ws-wn)/(wo-wn) if wo > wn else 0)
        phi, v = g.shapley()
        leadshare.append(100*phi[0]/phi.sum())

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax[0].plot(ns, poa, marker="o", color=C["leader"], lw=2.2)
    ax[0].set_xlabel("number of subsystems $n$")
    ax[0].set_ylabel(r"Price of Anarchy  $W_{\mathrm{soc}}/W_{\mathrm{Nash}}$")
    ax[0].set_title("(a)  Decentralisation cost grows with $n$")
    for xi, yi in zip(ns, poa):
        ax[0].annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                       xytext=(0, 7), fontsize=7.5, ha="center")

    ax[1].plot(ns, leadshare, marker="s", color=C["stk"], lw=2.2,
               label="leader Shapley share")
    ax[1].set_xlabel("number of subsystems $n$")
    ax[1].set_ylabel("leader Shapley share  [%]", color=C["stk"])
    ax[1].tick_params(axis="y", labelcolor=C["stk"])
    ax[1].set_title("(b)  Leader prominence vs. system size")
    ax2 = ax[1].twinx()
    ax2.plot(ns, recov, marker="^", color=C["accent"], lw=2.0, ls="--",
             label="welfare gap recovered")
    ax2.set_ylabel("welfare gap recovered  [%]", color=C["accent"])
    ax2.tick_params(axis="y", labelcolor=C["accent"])
    ax2.grid(False)
    lines = ax[1].get_lines()+ax2.get_lines()
    ax[1].legend(lines, [l.get_label() for l in lines], fontsize=8.2, loc="upper right")
    _save(fig, "fig_scalability")


# ----------------------------------------------------------------------------- #
#  FIG 9 -- satellite EPS reliability block diagram (schematic)
# ----------------------------------------------------------------------------- #
def fig_blockdiagram(sys):
    fig, ax = plt.subplots(figsize=(9.0, 2.7))
    ax.axis("off")
    labels = ["Solar-Array\nOperating", "Battery\n/ Cell",
              "Electrical\nDistribution", "Solar-Array\nDeployment"]
    roles = ["LEADER", "follower", "follower", "follower"]
    cols = [C["leader"]] + FOLLOW
    x0, y0, w, h, gap = 0.055, 0.30, 0.165, 0.40, 0.065
    centers = []
    for k in range(4):
        x = x0 + k*(w+gap)
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                             linewidth=1.8, edgecolor="k",
                             facecolor=cols[k], alpha=0.16 if k else 0.24)
        ax.add_patch(box)
        ax.text(x+w/2, y0+h*0.60, labels[k], ha="center", va="center",
                fontsize=10.5, weight="bold")
        ax.text(x+w/2, y0+h*0.20, roles[k], ha="center", va="center",
                fontsize=8.5, style="italic",
                color=cols[k], weight="bold" if k == 0 else "normal")
        centers.append((x, x+w, y0+h/2))
    # series connections
    for k in range(3):
        ax.add_patch(FancyArrowPatch((centers[k][1], centers[k][2]),
                                     (centers[k+1][0], centers[k+1][2]),
                                     arrowstyle="-|>", mutation_scale=16,
                                     lw=1.8, color="#333"))
    # input/output
    ax.add_patch(FancyArrowPatch((0.0, centers[0][2]), (centers[0][0], centers[0][2]),
                                 arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#333"))
    ax.add_patch(FancyArrowPatch((centers[3][1], centers[3][2]), (1.0, centers[3][2]),
                                 arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#333"))
    ax.text(-0.045, centers[0][2]+0.135, "Sun", fontsize=9, ha="center", style="italic")
    ax.text(1.06, centers[3][2]+0.135, "Payload\nbus", fontsize=9, ha="center",
            va="center", style="italic")
    ax.text(0.5, 0.94, r"Series reliability:  $R_{\mathrm{sys}}=\prod_{i=1}^{4} R_i$",
            ha="center", fontsize=11)
    ax.set_xlim(-0.10, 1.14); ax.set_ylim(0, 1)
    _save(fig, "fig_blockdiagram")


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    sys = satellite_eps()
    fig_blockdiagram(sys)
    fig_cost(sys)
    fig_provision(sys)
    fig_regimes(sys)
    fig_convergence(sys)
    fig_gamma(sys)
    fig_shapley(sys)
    fig_landscape()
    fig_scalability()
    print("\nAll figures written to ./figs")
