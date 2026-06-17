"""
Protocol + Pigouvian subsidy on the REAL EPS case: how much of the Nash-to-social gap is recovered by the
    leader alone, and by leader + follower subsidy at several intensities.

Simultaneous misreporting: all followers under-report their cost rate
    beta at once (not one at a time), to probe the complete-information
    assumption A3 more severely.

A second, independent case study (a generic five-subsystem series
    system with different parameters) to show the method transfers beyond
    the single satellite-EPS instance.
"""
import numpy as np
from stackelberg_shapley import Subsystem, SeriesSystem, satellite_eps_real
import analysis3 as A3

THETA, GAMMA = 0.70, 0.20


# --------------------------------------------------------------------------- #
#  (2) Protocol + subsidy on the real EPS case
# --------------------------------------------------------------------------- #
def eps_protocol_plus_subsidy(s_list=(0.0, 0.25, 0.5, 0.75, 1.0)):
    g = satellite_eps_real()
    Wn = g.welfare(g.solve_nash())
    Wo = g.welfare(g.solve_social())
    Wstk = g.welfare(g.solve_stackelberg(theta=THETA))
    leader_alone = (Wstk - Wn) / (Wo - Wn)
    rows = []
    for s in s_list:
        W = g.welfare(A3.subsidy_equilibrium(g, s))
        rows.append((s, (W - Wn) / (Wo - Wn)))
    return leader_alone, rows


# --------------------------------------------------------------------------- #
#  (4) Simultaneous misreporting of beta by ALL followers
# --------------------------------------------------------------------------- #
def simultaneous_misreport(under=0.20):
    """All followers simultaneously under-report beta by factor (1-under);
    the leader is truthful. Recompute Shapley allocation on the REPORTED game
    and report how the leader's post-transfer share moves and whether any
    follower's true individual rationality is still respected."""
    true = satellite_eps_real()
    N = set(range(true.n))
    # truthful allocation
    phi_t, v_t = true.shapley()
    kap_t = v_t(N) - v_t(N - {true.L})
    shareL_true = (phi_t[true.L] + GAMMA * kap_t) / phi_t.sum()

    # reported system: followers shade beta down (claim cheaper improvement)
    subs = []
    for i, s in enumerate(true.subs):
        b = s.beta * (1 - under) if i in true.F else s.beta
        subs.append(Subsystem(s.name, s.alpha, b, s.pi, s.Rmin, s.Rmax, s.role))
    rep = SeriesSystem(subs, leader_index=true.L)
    phi_r, v_r = rep.shapley()
    kap_r = v_r(N) - v_r(N - {rep.L})
    shareL_rep = (phi_r[rep.L] + GAMMA * kap_r) / phi_r.sum()
    # follower aggregate share gain from joint misreport
    foll_true = sum(phi_t[j] - GAMMA * kap_t / (true.n - 1) for j in true.F) / phi_t.sum()
    foll_rep = sum(phi_r[j] - GAMMA * kap_r / (rep.n - 1) for j in rep.F) / phi_r.sum()
    return dict(under=under,
                shareL_true=100 * shareL_true, shareL_rep=100 * shareL_rep,
                foll_gain_pts=100 * (foll_rep - foll_true))


# --------------------------------------------------------------------------- #
#  (5) Second, independent case study: generic 5-subsystem series system
# --------------------------------------------------------------------------- #
def second_case_study():
    """A generic five-subsystem series system with a dominant leader and four
    heterogeneous followers"""
    subs = [
        Subsystem("Dominant unit (L)", 0.030, 0.095, 4.0, 0.90, 0.9995, "leader"),
        Subsystem("Follower A",        0.026, 0.070, 2.2, 0.90, 0.9995),
        Subsystem("Follower B",        0.022, 0.065, 1.7, 0.90, 0.9995),
        Subsystem("Follower C",        0.020, 0.058, 1.3, 0.90, 0.9995),
        Subsystem("Follower D",        0.018, 0.050, 0.9, 0.90, 0.9995),
    ]
    g = SeriesSystem(subs, leader_index=0)
    N = set(range(g.n))
    Rn, Rs, Ro = g.solve_nash(), g.solve_stackelberg(theta=THETA), g.solve_social()
    Wn, Ws, Wo = g.welfare(Rn), g.welfare(Rs), g.welfare(Ro)
    rho, _ = g.br_jacobian_spectral_radius(Rn)
    phi, v = g.shapley(); kap = v(N) - v(N - {g.L})
    shareL0 = 100 * phi[g.L] / phi.sum()
    shareL = 100 * (phi[g.L] + GAMMA * kap) / phi.sum()
    return dict(Rn=g.R_sys(Rn), Rs=g.R_sys(Rs), Ro=g.R_sys(Ro),
                poa=Wo / Wn, rho=rho, kap=kap,
                closed=100 * (Ws - Wn) / (Wo - Wn),
                shareL0=shareL0, shareL=shareL,
                pivotal=bool(phi[g.L] == max(phi)))


if __name__ == "__main__":
    print("=" * 68)
    print("(2) PROTOCOL + SUBSIDY ON THE REAL EPS CASE")
    la, rows = eps_protocol_plus_subsidy()
    print(f"  leader alone closes: {100*la:.1f}%")
    for s, rec in rows:
        tag = "  (= Nash)" if s == 0 else ("  (= social opt)" if s == 1 else "")
        print(f"  + follower subsidy s={s:.2f}: closes {100*rec:5.1f}% of the gap{tag}")

    print("=" * 68)
    print("(4) SIMULTANEOUS MISREPORTING (all followers shade beta down 20%)")
    m = simultaneous_misreport(0.20)
    print(f"  leader share truthful={m['shareL_true']:.1f}%  reported={m['shareL_rep']:.1f}%")
    print(f"  followers' aggregate share gain from joint misreport: "
          f"{m['foll_gain_pts']:+.1f} percentage points")

    print("=" * 68)
    print("(5) SECOND CASE STUDY (generic 5-subsystem series system)")
    c = second_case_study()
    print(f"  R_sys: Nash={c['Rn']:.4f} Stk={c['Rs']:.4f} Social={c['Ro']:.4f}  PoA={c['poa']:.3f}")
    print(f"  contraction rho={c['rho']:.4f}  kappa_L={c['kap']:.3f}  closed={c['closed']:.1f}%")
    print(f"  leader share {c['shareL0']:.1f}% -> {c['shareL']:.1f}%  pivotal={c['pivotal']}")
