"""
Revision analyses for the Stackelberg-Shapley reliability paper.

Adds the quantitative studies requested by reviewers:
  A. theta-sensitivity   : how the leader's accountability weight moves the
                           equilibrium (R_sys, R_L, mean follower R, welfare,
                           leader's final allocation).
  B. gamma design        : the individually-rational range of the transfer
                           parameter, derived from follower participation
                           constraints (replaces the 'arbitrary gamma' charge).
  C. robustness          : sensitivity of the headline metrics to the cost
                           rate beta (+/-25%) and to the characteristic-function
                           baseline convention (rmin / nash / rmax).
  D. allocation baselines: comparison of Stackelberg-Shapley against
                           proportional-to-stake, egalitarian, and plain Shapley,
                           with individual-rationality checks.
  E. contraction         : spectral radius of the best-response Jacobian
                           (verifies the contraction hypothesis of Prop. 1).
"""
import numpy as np
from copy import deepcopy
from stackelberg_shapley import satellite_eps_real as satellite_eps, SeriesSystem, Subsystem

GAMMA = 0.20
THETA = 0.70


# --------------------------------------------------------------------------- #
def theta_sweep(sys, thetas=None, gamma=GAMMA):
    if thetas is None:
        thetas = np.linspace(0.0, 1.0, 21)
    out = dict(theta=thetas, Rsys=[], RL=[], RFmean=[], W=[],
               leader_payoff=[], leader_alloc=[])
    N = set(range(sys.n))
    for th in thetas:
        R = sys.solve_stackelberg(theta=th)
        out["Rsys"].append(sys.R_sys(R))
        out["RL"].append(R[sys.L])
        out["RFmean"].append(np.mean([R[j] for j in sys.F]))
        out["W"].append(sys.welfare(R))
        out["leader_payoff"].append(sys.utility(sys.L, R))
        phi, v = sys.shapley()
        kap = v(N) - v(N - {sys.L})
        out["leader_alloc"].append(phi[sys.L] + gamma * kap)
    for k in out:
        out[k] = np.array(out[k])
    return out


# --------------------------------------------------------------------------- #
def gamma_ir_range(sys):
    """Individually-rational interval [0, gamma_max] for the transfer."""
    N = set(range(sys.n))
    phi, v = sys.shapley()
    kap = v(N) - v(N - {sys.L})
    floors = {j: v({j}) for j in sys.F}
    gmax = 1.0
    binding = None
    for j in sys.F:
        gj = (phi[j] - floors[j]) * (sys.n - 1) / kap
        if gj < gmax:
            gmax, binding = gj, j
    return dict(phi=phi, v=v, kappa_L=kap, floors=floors,
                gamma_max=gmax, binding=binding)


# --------------------------------------------------------------------------- #
def robustness_beta(scales=(0.75, 0.90, 1.0, 1.10, 1.25), gamma=GAMMA, theta=THETA):
    """Perturb every beta by a common scale; report headline metrics."""
    rows = []
    N = None
    for sc in scales:
        sys = satellite_eps()
        for s in sys.subs:
            s.beta *= sc
        N = set(range(sys.n))
        Rn, Rs, Ro = sys.solve_nash(), sys.solve_stackelberg(theta=theta), sys.solve_social()
        phi, v = sys.shapley()
        kap = v(N) - v(N - {sys.L})
        rows.append(dict(
            scale=sc,
            PoA=sys.welfare(Ro) / sys.welfare(Rn),
            delta=sys.R_sys(Rs) - sys.R_sys(Rn),
            kappa_L=kap,
            leader_share=100 * (phi[sys.L] + gamma * kap) / phi.sum(),
        ))
    return rows


def robustness_baseline(sys, gamma=GAMMA):
    N = set(range(sys.n))
    rows = []
    for base in ("rmin", "nash", "rmax"):
        phi, v = sys.shapley(baseline=base)
        kap = v(N) - v(N - {sys.L})
        rows.append(dict(baseline=base, kappa_L=kap,
                         leader_shapley=100 * phi[sys.L] / phi.sum(),
                         leader_stk=100 * (phi[sys.L] + gamma * kap) / phi.sum()))
    return rows


# --------------------------------------------------------------------------- #
def allocation_baselines(sys, gamma=GAMMA):
    """Compare four allocation rules + individual-rationality check."""
    N = set(range(sys.n))
    phi, v = sys.shapley()
    tot = phi.sum()
    kap = v(N) - v(N - {sys.L})
    stakes = np.array([s.pi for s in sys.subs])
    floors = np.array([v({i}) for i in range(sys.n)])

    egal = np.full(sys.n, tot / sys.n)
    prop = tot * stakes / stakes.sum()
    shap = phi.copy()
    stk = phi.copy()
    stk[sys.L] = phi[sys.L] + gamma * kap
    for j in sys.F:
        stk[j] = phi[j] - gamma * kap / (sys.n - 1)

    rules = {"Egalitarian": egal, "Proportional": prop,
             "Shapley": shap, "Stk-Shapley": stk}
    ir = {name: bool(np.all(a >= floors - 1e-9)) for name, a in rules.items()}
    return rules, floors, ir


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sys = satellite_eps()
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("E. CONTRACTION DIAGNOSTIC")
    rho, J = sys.br_jacobian_spectral_radius()
    print(f"   spectral radius of follower BR Jacobian at Nash = {rho:.4f}")
    print(f"   contraction (rho<1) verified: {rho < 1}")

    print("=" * 70)
    print("B. GAMMA INDIVIDUAL-RATIONALITY RANGE")
    g = gamma_ir_range(sys)
    print(f"   kappa_L = {g['kappa_L']:.4f}")
    for j in sys.F:
        print(f"   floor v({{{sys.subs[j].name[:14]}}}) = {g['floors'][j]:.4f}")
    print(f"   gamma_max = {g['gamma_max']:.3f}  (binding: {sys.subs[g['binding']].name})")
    print(f"   chosen gamma = {GAMMA} is IR-feasible: {GAMMA < g['gamma_max']}")

    print("=" * 70)
    print("C1. ROBUSTNESS TO COST RATE beta")
    for r in robustness_beta():
        print(f"   beta x{r['scale']:.2f} | PoA={r['PoA']:.3f} "
              f"| delta={r['delta']:.4f} | kappa_L={r['kappa_L']:.3f} "
              f"| leader share={r['leader_share']:.1f}%")

    print("C2. ROBUSTNESS TO CHARACTERISTIC-FUNCTION BASELINE")
    for r in robustness_baseline(sys):
        print(f"   baseline={r['baseline']:5s} | kappa_L={r['kappa_L']:.3f} "
              f"| leader Shapley={r['leader_shapley']:.1f}% "
              f"| leader Stk={r['leader_stk']:.1f}%")

    print("=" * 70)
    print("D. ALLOCATION-RULE COMPARISON  (IR = all shares >= stand-alone)")
    rules, floors, ir = allocation_baselines(sys)
    print(f"   {'rule':14s} | " + " ".join(f"{s.name[:10]:>10s}" for s in sys.subs)
          + " | IR")
    for name, a in rules.items():
        print(f"   {name:14s} | " + " ".join(f"{x:10.3f}" for x in a)
              + f" | {ir[name]}")
    print(f"   {'floor v(i)':14s} | " + " ".join(f"{x:10.3f}" for x in floors))

    print("=" * 70)
    print("A. THETA SWEEP (sample)")
    t = theta_sweep(sys, thetas=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    for k in range(len(t["theta"])):
        print(f"   theta={t['theta'][k]:.2f} | R_sys={t['Rsys'][k]:.4f} "
              f"| R_L={t['RL'][k]:.4f} | W={t['W'][k]:.4f} "
              f"| leader alloc={t['leader_alloc'][k]:.3f}")
