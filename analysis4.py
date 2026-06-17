"""
1 -- connect the cost parameters to data, two ways:
  (1) An explicit, reproducible *recipe* mapping a subsystem's published
      Weibull reliability to the cost primitives (alpha_i, beta_i).
  (2) A wide two-dimensional sensitivity sweep over (alpha-scale, beta-scale)
      -- an order of magnitude in beta and 0.5x-2x in alpha -- showing the
      headline numbers (~10% gap closed; leader share ~40%->~50%) are stable,
      not artefacts of one hand-picked calibration.

2 -- a concrete risk-aversion table (R_sys, % gap closed, contraction).

The Weibull anchor uses the published fact that space-EPS constituents follow
Weibull reliability with shape ~1 (infant-mortality regime) and that the
Solar-Array Operating element is the dominant culprit (Castet & Saleh 2009b;
Kim, Castet & Saleh 2012).
"""
import numpy as np
from stackelberg_shapley import Subsystem, SeriesSystem, satellite_eps_real

THETA, GAMMA = 0.70, 0.20


# --------------------------------------------------------------------------- #
#  Defect 1(a): an explicit Weibull -> (alpha, beta) calibration recipe
# --------------------------------------------------------------------------- #
def calibrate_from_weibull(weibull, T_design=15.0, c_ref=0.0365,
                           beta_min=0.060, beta_max=0.088):
    """Map published Weibull (shape k_i, scale lambda_i in years) to cost
    primitives by a transparent, hazard-informed recipe.

    Step 1 (baseline reliability).  R0_i = exp(-(T/lambda_i)**k_i): the
    on-orbit reliability at the design life T.

    Step 2 (cost rate beta, hazard-informed).  A constituent that fails faster
    is harder (steeper) to harden, so the cost rate is taken affine in the
    Weibull hazard at the design life,
        h_i(T) = (k_i/lambda_i) (T/lambda_i)^{k_i-1},
        beta_i = beta_min + (beta_max-beta_min) (h_i - h_min)/(h_max - h_min).
    This makes the most failure-prone constituent (the Solar-Array Operating
    element) carry the steepest cost rate, recovering the ordering used in the
    worked example.

    Step 3 (scale alpha).  Fix alpha so the design cost at the baseline equals a
    reference unit cost: alpha_i = c_ref * exp(-beta_i * x(R0_i)),
    x(R)=R/(1-R).  Stakes pi_i come from the published failure-contribution
    shares (Table, Section 7.1) and are not part of the cost calibration."""
    def x(R):
        return R / (1.0 - R)
    hz = []
    for (name, k, lam, pi, role) in weibull:
        hz.append((k / lam) * (T_design / lam) ** (k - 1.0))
    hz = np.array(hz)
    hmin, hmax = hz.min(), hz.max()
    out = []
    for i, (name, k, lam, pi, role) in enumerate(weibull):
        R0 = float(np.exp(-(T_design / lam) ** k))
        beta = beta_min + (beta_max - beta_min) * (hz[i] - hmin) / (hmax - hmin)
        alpha = c_ref * np.exp(-beta * x(R0))
        out.append(dict(name=name, R0=R0, hazard=float(hz[i]),
                        alpha=alpha, beta=beta, pi=pi, role=role))
    return out


def calibrated_system():
    """Apply the recipe to representative EPS Weibull values (shape ~1,
    constituent scales ordered by the published culprit ranking SAO<Battery<
    ED<SAD in increasing reliability)."""
    # (name, Weibull shape k, scale lambda [yr], stake pi, role)
    weibull = [
        ("Solar-Array Operating (L)", 0.90,  95.0, 3.2, "leader"),
        ("Battery / Cell",            0.95, 120.0, 1.9, "follower"),
        ("Electrical Distribution",   1.00, 150.0, 1.5, "follower"),
        ("Solar-Array Deployment",    1.05, 200.0, 1.1, "follower"),
    ]
    cal = calibrate_from_weibull(weibull)
    subs = [Subsystem(c["name"], c["alpha"], c["beta"], c["pi"], 0.90, 0.9995, c["role"])
            for c in cal]
    return SeriesSystem(subs, leader_index=0), cal


# --------------------------------------------------------------------------- #
#  Defect 1(b): wide 2-D sensitivity over (alpha-scale, beta-scale)
# --------------------------------------------------------------------------- #
def alpha_beta_sweep(a_scales=None, b_scales=None):
    """Scale every alpha_i and beta_i by common factors over a wide grid and
    record the gap closed by leadership and the leader's Shapley->Stk share."""
    if a_scales is None:
        a_scales = [0.5, 1.0, 2.0]
    if b_scales is None:
        b_scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]      # an order of magnitude
    base = satellite_eps_real()
    rows = []
    for bs in b_scales:
        for as_ in a_scales:
            subs = [Subsystem(s.name, s.alpha * as_, s.beta * bs, s.pi,
                              s.Rmin, s.Rmax, s.role) for s in base.subs]
            g = SeriesSystem(subs, leader_index=base.L)
            Rn, Rs, Ro = g.solve_nash(), g.solve_stackelberg(theta=THETA), g.solve_social()
            Wn, Ws, Wo = g.welfare(Rn), g.welfare(Rs), g.welfare(Ro)
            phi, v = g.shapley()
            N = set(range(g.n)); kap = v(N) - v(N - {g.L})
            stk = phi.copy(); stk[g.L] = phi[g.L] + GAMMA * kap
            for j in g.F: stk[j] = phi[j] - GAMMA * kap / (g.n - 1)
            rows.append(dict(a=as_, b=bs,
                             closed=100 * (Ws - Wn) / (Wo - Wn),
                             share0=100 * phi[g.L] / phi.sum(),
                             share=100 * stk[g.L] / stk.sum(),
                             poa=Wo / Wn))
    return rows


# --------------------------------------------------------------------------- #
#  Defect 2: risk-aversion table with contraction index
# --------------------------------------------------------------------------- #
def risk_table(etas=(1.0, 0.75, 0.5)):
    import analysis3 as A3
    rows = []
    for eta in etas:
        base = satellite_eps_real()
        sys = A3.RiskAverseSystem(base.subs, leader_index=base.L, eta=eta)
        Rn, Rs, Ro = sys.solve_nash(), sys.solve_stackelberg(theta=THETA), sys.solve_social()
        Wn, Ws, Wo = sys.welfare(Rn), sys.welfare(Rs), sys.welfare(Ro)
        rho, _ = sys.br_jacobian_spectral_radius(Rn)
        rows.append(dict(eta=eta, Rsys=sys.R_sys(Rs),
                         closed=100 * (Ws - Wn) / (Wo - Wn), rho=rho))
    return rows


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("DEFECT 1(a): Weibull -> (alpha,beta) CALIBRATION RECIPE")
    sysc, cal = calibrated_system()
    for c in cal:
        print(f"  {c['name']:26s} R0={c['R0']:.4f}  alpha={c['alpha']:.4g}  beta={c['beta']:.4f}")
    Rn, Rs, Ro = sysc.solve_nash(), sysc.solve_stackelberg(theta=THETA), sysc.solve_social()
    Wn, Ws, Wo = sysc.welfare(Rn), sysc.welfare(Rs), sysc.welfare(Ro)
    phi, v = sysc.shapley(); N = set(range(sysc.n)); kap = v(N) - v(N - {sysc.L})
    print(f"  --> calibrated system: R_sys Nash={sysc.R_sys(Rn):.4f} Stk={sysc.R_sys(Rs):.4f}"
          f" Soc={sysc.R_sys(Ro):.4f}; closed={100*(Ws-Wn)/(Wo-Wn):.1f}%;"
          f" leader {100*phi[sysc.L]/phi.sum():.1f}%->{100*(phi[sysc.L]+GAMMA*kap)/phi.sum():.1f}%")

    print("=" * 70)
    print("DEFECT 1(b): WIDE (alpha,beta) SWEEP  [closed% | leader share% ]")
    rows = alpha_beta_sweep()
    cl = [r["closed"] for r in rows]; sh = [r["share"] for r in rows]
    for r in rows:
        print(f"  beta x{r['b']:<4} alpha x{r['a']:<4} : closed={r['closed']:5.1f}%  "
              f"leader {r['share0']:.1f}%->{r['share']:.1f}%  PoA={r['poa']:.3f}")
    print(f"  RANGE over whole grid: closed in [{min(cl):.1f},{max(cl):.1f}]%, "
          f"leader-Stk share in [{min(sh):.1f},{max(sh):.1f}]%")

    print("=" * 70)
    print("DEFECT 2: RISK-AVERSION TABLE")
    for r in risk_table():
        print(f"  eta={r['eta']:.2f}  R_sys(Stk)={r['Rsys']:.4f}  closed={r['closed']:.1f}%  rho={r['rho']:.4f}")
