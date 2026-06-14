"""
Revision-2 analyses: real-data anchoring and the comparisons reviewers asked for.

Real-data basis (all citable):
  * EPS is the single largest driver of spacecraft unreliability: >25% of all
    on-orbit failures, and ~41% of failures after 5 years
    (Kim, Castet & Saleh, RESS 98(1):55-65, 2012; Wertz & Larson, SMAD).
  * The EPS decomposes into four constituents tracked in the SpaceTrak/Seradata
    on-orbit database (1584 satellites, 1990-2008): Solar-Array Operating (SAO),
    Battery/Cell, Electrical Distribution (ED), Solar-Array Deployment (SAD)
    (Kim et al. 2011/2012; Castet & Saleh, RESS 94(11):1718-1728, 2009).
  * SAO is reported as the dominant culprit (majority of EPS failure events),
    which is our data-grounded justification for making SAO the leader.
  * Space-grade components show Weibull shape ~1.7 and unit costs of tens of M$
    (Mosleh, Dalili & Heydari, IEEE Systems J., 2016).

We map these facts to model primitives, then (A) compare the protocol's
operating point and allocation against classic reliability-allocation methods
(Equal, ARINC, AGREE) and the cost-optimal allocation; (B) test robustness to
the cost *functional form* (exponential vs power-law); (C) quantify the
manipulation incentive (the protocol is budget-balanced, hence not
strategy-proof -- we measure how strong the incentive to misreport is);
(D) decompose the contribution of each protocol stage; (E) push scalability to
n = 20.
"""
import numpy as np
from scipy.optimize import brentq, minimize
from stackelberg_shapley import Subsystem, SeriesSystem

GAMMA = 0.20
THETA = 0.70


# --------------------------------------------------------------------------- #
#  Real-data-anchored EPS case study
# --------------------------------------------------------------------------- #
def satellite_eps_real():
    """Four real EPS constituents in series, leader = Solar-Array Operating.

    Stakes pi_i are assigned in proportion to each constituent's *published
    ranking* as a contributor to EPS unreliability (SAO > Battery > ED > SAD),
    using representative shares consistent with the cited multi-state failure
    analyses. beta_i are kept in the finite small-rate regime (defended in the
    paper: beta=5 would make one subsystem cost ~e^95, exceeding any real
    budget by tens of orders of magnitude, so it is the *unrealistic* choice).
    Reliability bounds reflect mature space-EPS operating ranges."""
    subs = [
        #          name                         alpha   beta    pi    Rmin   Rmax    role
        Subsystem("Solar-Array Operating (L)",  0.024,  0.088,  3.2,  0.90, 0.9995, "leader"),
        Subsystem("Battery / Cell",             0.032,  0.078,  1.9,  0.90, 0.9995),
        Subsystem("Electrical Distribution",    0.027,  0.071,  1.5,  0.90, 0.9995),
        Subsystem("Solar-Array Deployment",     0.020,  0.060,  1.1,  0.90, 0.9995),
    ]
    return SeriesSystem(subs, leader_index=0)


# --------------------------------------------------------------------------- #
#  A. Classic reliability-allocation methods (to hit a common system target)
# --------------------------------------------------------------------------- #
def cost_to_hit(sys, R):
    return sum(s.cost(R[i]) for i, s in enumerate(sys.subs))


def alloc_equal(sys, R_target):
    n = sys.n
    return np.array([R_target ** (1.0 / n)] * n)


def alloc_arinc(sys, R_target, baseline=None):
    """ARINC: unreliability budget split proportional to predicted failure
    rate lambda_i = -ln R_i^base.  R_i = R_target^{w_i}, sum w_i = 1."""
    if baseline is None:
        baseline = sys.solve_nash()
    lam = np.array([-np.log(baseline[i]) for i in range(sys.n)])
    w = lam / lam.sum()
    return np.array([R_target ** w[i] for i in range(sys.n)])


def alloc_agree(sys, R_target, complexity=None):
    """AGREE-style: unreliability split proportional to subsystem complexity
    n_i (module count).  -ln R_i = (n_i/N)(-ln R_target)."""
    if complexity is None:
        # representative module counts for EPS constituents
        complexity = np.array([40.0, 24.0, 16.0, 8.0])[:sys.n]
    w = complexity / complexity.sum()
    return np.array([R_target ** w[i] for i in range(sys.n)])


def alloc_cost_optimal(sys, R_target):
    """Cost-minimal allocation achieving prod R_i = R_target (Lagrangian)."""
    n = sys.n
    logT = np.log(R_target)

    def obj(x):  # x = logits mapped to (Rmin,Rmax)
        R = _logit_to_R(sys, x)
        return cost_to_hit(sys, R)

    def con(x):
        R = _logit_to_R(sys, x)
        return np.sum(np.log(R)) - logT

    from scipy.optimize import NonlinearConstraint
    x0 = np.zeros(n)
    nlc = NonlinearConstraint(con, 0, 0)
    res = minimize(obj, x0, constraints=[nlc], method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    return _logit_to_R(sys, res.x)


def _logit_to_R(sys, x):
    R = np.empty(sys.n)
    for i, s in enumerate(sys.subs):
        lo, hi = s.Rmin, min(s.Rmax, 0.99995)
        R[i] = lo + (hi - lo) / (1.0 + np.exp(-x[i]))
    return R


def compare_methods(sys, R_target=0.95):
    rows = []
    for name, R in [("Equal", alloc_equal(sys, R_target)),
                    ("ARINC", alloc_arinc(sys, R_target)),
                    ("AGREE", alloc_agree(sys, R_target)),
                    ("Cost-optimal", alloc_cost_optimal(sys, R_target))]:
        rows.append(dict(name=name, R=R, Rsys=float(np.prod(R)),
                         cost=cost_to_hit(sys, R)))
    return rows


# --------------------------------------------------------------------------- #
#  B. Functional-form robustness: power-law cost
# --------------------------------------------------------------------------- #
class PowerLawSub(Subsystem):
    """C_i(R) = a * (R/(1-R))^b  -- convex, divergent at 1, different shape."""
    def cost(self, R):
        R = np.clip(R, self.Rmin, min(self.Rmax, 0.999999))
        return self.alpha * (R / (1.0 - R)) ** self.beta

    def dcost(self, R):
        R = np.clip(R, self.Rmin, min(self.Rmax, 0.999999))
        x = R / (1.0 - R)
        dx = 1.0 / (1.0 - R) ** 2
        return self.alpha * self.beta * x ** (self.beta - 1.0) * dx


def powerlaw_eps():
    base = satellite_eps_real()
    subs = []
    for s in base.subs:
        # choose exponents giving comparable operating costs
        subs.append(PowerLawSub(s.name, 0.0016, 1.6, s.pi, s.Rmin, s.Rmax, s.role))
    return SeriesSystem(subs, leader_index=0)


# --------------------------------------------------------------------------- #
#  C. Manipulation: incentive to misreport cost rate beta_j
# --------------------------------------------------------------------------- #
def manipulation_curve(real_sys, j, gamma=GAMMA, factors=None):
    """Follower j reports beta_j' = f * beta_j (over/under-statement).  We
    recompute the Shapley allocation under the *reported* game and record j's
    post-transfer share, holding everyone else truthful.  A flat/decreasing
    curve => little incentive to inflate; an increasing curve => manipulable."""
    if factors is None:
        factors = np.linspace(0.6, 1.6, 11)
    N = set(range(real_sys.n))
    out = []
    true_beta = real_sys.subs[j].beta
    for f in factors:
        subs = []
        for k, s in enumerate(real_sys.subs):
            b = s.beta * f if k == j else s.beta
            subs.append(Subsystem(s.name, s.alpha, b, s.pi, s.Rmin, s.Rmax, s.role))
        g = SeriesSystem(subs, leader_index=real_sys.L)
        phi, v = g.shapley()
        kap = v(N) - v(N - {g.L})
        if j == g.L:
            share = (phi[j] + gamma * kap)
        else:
            share = (phi[j] - gamma * kap / (g.n - 1))
        out.append(share)
    return np.array(factors), np.array(out)


# --------------------------------------------------------------------------- #
#  D. Stage-contribution decomposition
# --------------------------------------------------------------------------- #
def stage_decomposition(sys, theta=THETA):
    Rn = sys.solve_nash()
    Rs = sys.solve_stackelberg(theta=theta)
    Ro = sys.solve_social()
    Wn, Ws, Wo = sys.welfare(Rn), sys.welfare(Rs), sys.welfare(Ro)
    return dict(
        Wn=Wn, Ws=Ws, Wo=Wo,
        stk_gain=Ws - Wn,                  # what leadership (stages 1-2) adds
        residual=Wo - Ws,                  # what leadership cannot reach
        frac_closed=(Ws - Wn) / (Wo - Wn),
    )


# --------------------------------------------------------------------------- #
#  E. Scalability to n = 20
# --------------------------------------------------------------------------- #
def scalability(ns=range(2, 21)):
    rng = np.random.default_rng(7)
    rows = []
    for n in ns:
        subs = [Subsystem("L", 0.024, 0.088, 3.2, 0.90, 0.9995, "leader")]
        for k in range(n - 1):
            subs.append(Subsystem(f"F{k}", 0.028, 0.072,
                                   float(1.0 + 0.6 * rng.random()),
                                   0.90, 0.9995))
        g = SeriesSystem(subs, leader_index=0)
        Rn, Ro = g.solve_nash(), g.solve_social()
        phi, v = g.shapley() if n <= 11 else (None, None)
        poa = g.welfare(Ro) / g.welfare(Rn)
        share = (100 * phi[g.L] / phi.sum()) if phi is not None else np.nan
        rows.append(dict(n=n, poa=poa, leader_share=share))
    return rows


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sys = satellite_eps_real()

    print("=" * 72)
    print("REAL-DATA EPS CASE STUDY (SAO / Battery / ED / SAD)")
    Rn, Rs, Ro = sys.solve_nash(), sys.solve_stackelberg(theta=THETA), sys.solve_social()
    for tag, R in [("Nash", Rn), ("Stackelberg", Rs), ("Social", Ro)]:
        print(f"  {tag:12s} R_sys={np.prod(R):.4f}  W={sys.welfare(R):.4f}  R={R}")
    rho, _ = sys.br_jacobian_spectral_radius()
    print(f"  contraction rho={rho:.4f}")
    m = sys.metrics(GAMMA, theta=THETA)
    print(f"  kappa_L={m['kappa_L']:.3f}  leader share {100*m['phi'][sys.L]/m['phi'].sum():.1f}%"
          f" -> {100*m['phi_stk'][sys.L]/m['phi_stk'].sum():.1f}%")

    print("=" * 72)
    print("A. CLASSIC METHODS vs COST-OPTIMAL (target R_sys=0.95)")
    for r in compare_methods(sys, 0.95):
        print(f"  {r['name']:13s} R_sys={r['Rsys']:.4f}  total cost={r['cost']:.3f}  R={r['R']}")

    print("=" * 72)
    print("B. FUNCTIONAL-FORM ROBUSTNESS (power-law cost)")
    pl = powerlaw_eps()
    d = stage_decomposition(pl)
    print(f"  power-law: frac of welfare gap closed by leadership = {100*d['frac_closed']:.1f}%")
    print(f"  (exponential reference below)")

    print("=" * 72)
    print("C. MANIPULATION INCENTIVE (follower 1 misreports beta)")
    f, share = manipulation_curve(sys, 1)
    print(f"  factor : {f}")
    print(f"  share  : {share}")
    print(f"  truthful share={share[5]:.3f}; max share={share.max():.3f} at f={f[np.argmax(share)]:.2f}")

    print("=" * 72)
    print("D. STAGE DECOMPOSITION (exponential, real case)")
    d = stage_decomposition(sys)
    print(f"  W_nash={d['Wn']:.4f} -> W_stk={d['Ws']:.4f} -> W_social={d['Wo']:.4f}")
    print(f"  leadership closes {100*d['frac_closed']:.1f}% of the Nash->social welfare gap")

    print("=" * 72)
    print("E. SCALABILITY to n=20")
    for r in scalability():
        print(f"  n={r['n']:2d}  PoA={r['poa']:.3f}  leader share={r['leader_share']:.1f}%")
