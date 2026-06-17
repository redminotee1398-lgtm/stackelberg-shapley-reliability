"""
A. Analytical contraction condition (closed form) and its verification.
B. Utility-form robustness: the linear utility is the *expected* value of the
     threshold (indicator) payoff; a risk-averse (concave) variant is tested.
C. A demonstrated remedy for long chains.
D. VCG comparison.
E. gamma_max sensitivity to the characteristic-function baseline.
F. AGREE sensitivity to the complexity weights.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from stackelberg_shapley import Subsystem, SeriesSystem, satellite_eps_real

THETA, GAMMA = 0.70, 0.20


def d2cost(s, R):
    """Second derivative of the exponential cost C=alpha*exp(beta*R/(1-R))."""
    R = float(np.clip(R, s.Rmin, min(s.Rmax, 0.999999)))
    b = s.beta
    g = b * R / (1.0 - R)
    gp = b / (1.0 - R) ** 2          # g'
    gpp = 2.0 * b / (1.0 - R) ** 3   # g''
    e = s.alpha * np.exp(g)
    return e * (gp * gp + gpp)


# --------------------------------------------------------------------------- #
#  A. Analytical contraction bound
# --------------------------------------------------------------------------- #
def contraction_bound(sys, R=None):
    """Closed-form sufficient condition for the follower best-response map to be
    a sup-norm contraction.  For follower j the best response solves
    pi_j * R_sys/R_j = C_j'(R_j); implicit differentiation gives

        dBR_j/dR_k = pi_j * R_sys /(R_j R_k C_j''(R_j))   (k != j),

    so the row sum Lambda_j = sum_{k in F, k!=j} dBR_j/dR_k, and
    Lambda := max_j Lambda_j < 1  ==>  contraction (hence unique equilibrium).
    Returns (Lambda, per-follower rows) and the numerical spectral radius."""
    if R is None:
        R = sys.solve_nash()
    R = np.asarray(R, float)
    Rsys = np.prod(R)
    F = sys.F
    rows = {}
    for j in F:
        s = sys.subs[j]
        c2 = d2cost(s, R[j])
        lam = 0.0
        for k in F:
            if k == j:
                continue
            lam += s.pi * Rsys / (R[j] * R[k] * c2)
        rows[j] = lam
    Lambda = max(rows.values())
    rho, _ = sys.br_jacobian_spectral_radius(R)
    return Lambda, rows, rho


# --------------------------------------------------------------------------- #
#  B. Utility-form robustness (risk-averse / concave benefit)
# --------------------------------------------------------------------------- #
class RiskAverseSystem(SeriesSystem):
    """u_i = pi_i * w(R_sys) - C_i, with w concave (CRRA-style w=R_sys^eta,
    0<eta<=1).  eta=1 recovers the linear (risk-neutral expected-threshold)
    model.  Solvers are overridden to maximise the *actual* concave utility,
    so the equilibria are internally consistent."""
    def __init__(self, subs, leader_index=0, eta=0.5):
        super().__init__(subs, leader_index)
        self.eta = eta

    def utility(self, i, R):
        Rsys = self.R_sys(R)
        return self.subs[i].pi * (Rsys ** self.eta) - self.subs[i].cost(R[i])

    def _maximise_i(self, i, R, stake):
        """1-D maximise stake*w(prod) - C_i over R_i with others fixed."""
        others = float(np.prod([R[k] for k in range(self.n) if k != i]))
        s = self.subs[i]
        lo, hi = s.Rmin, min(s.Rmax, 0.999999)

        def neg(Ri):
            return -(stake * (others * Ri) ** self.eta - s.cost(Ri))
        res = minimize_scalar(neg, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9})
        return float(res.x)

    def best_response(self, i, R):
        return self._maximise_i(i, R, self.subs[i].pi)

    def solve_social(self):
        Pi = sum(s.pi for s in self.subs)
        R = np.array([s.Rmin for s in self.subs], float)
        for _ in range(400):
            Rprev = R.copy()
            for i in range(self.n):
                R[i] = self._maximise_i(i, R, Pi)
            if np.max(np.abs(R - Rprev)) < 1e-9:
                break
        return R


def risk_aversion_scan(etas=(1.0, 0.75, 0.5)):
    rows = []
    for eta in etas:
        base = satellite_eps_real()
        sys = RiskAverseSystem(base.subs, leader_index=base.L, eta=eta)
        Rn, Rs, Ro = sys.solve_nash(), sys.solve_stackelberg(theta=THETA), sys.solve_social()
        Wn, Ws, Wo = sys.welfare(Rn), sys.welfare(Rs), sys.welfare(Ro)
        rows.append(dict(eta=eta,
                         Rn=sys.R_sys(Rn), Rs=sys.R_sys(Rs), Ro=sys.R_sys(Ro),
                         poa=Wo / Wn,
                         closed=(Ws - Wn) / (Wo - Wn) if Wo > Wn else float("nan")))
    return rows


# --------------------------------------------------------------------------- #
#  C. Follower-subsidy remedy (Pigouvian) for long chains
# --------------------------------------------------------------------------- #
def subsidy_equilibrium(sys, s_sub):
    """Nash among ALL teams when each team i receives a per-unit subsidy that
    augments its effective stake from pi_i to pi_i + s_sub*(Pi_tot - pi_i),
    i.e. a Pigouvian transfer internalising a fraction s_sub of the externality
    it imposes on others.  s_sub=0 -> Nash; s_sub=1 -> social optimum."""
    Pi = sum(x.pi for x in sys.subs)
    eff = [x.pi + s_sub * (Pi - x.pi) for x in sys.subs]
    R = np.array([x.Rmin for x in sys.subs], float)
    for _ in range(200):
        Rprev = R.copy()
        for i, x in enumerate(sys.subs):
            others = np.prod([R[k] for k in range(sys.n) if k != i])

            def foc(Ri):
                return eff[i] * others - x.dcost(Ri)
            lo, hi = x.Rmin, min(x.Rmax, 0.999999)
            if foc(lo) <= 0:
                R[i] = lo
            elif foc(hi) >= 0:
                R[i] = hi
            else:
                R[i] = brentq(foc, lo, hi, xtol=1e-10)
        if np.max(np.abs(R - Rprev)) < 1e-9:
            break
    return R


def subsidy_remedy(n_list=(4, 8, 12), s_grid=None):
    """For each chain length, how much of the Nash->social welfare gap a uniform
    follower subsidy recovers, vs the single-leader Stackelberg lever."""
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
        Wn, Wo = g.welfare(g.solve_nash()), g.welfare(g.solve_social())
        Wstk = g.welfare(g.solve_stackelberg(theta=THETA))
        rec_sub = []
        for s_sub in s_grid:
            W = g.welfare(subsidy_equilibrium(g, s_sub))
            rec_sub.append((W - Wn) / (Wo - Wn))
        out[n] = dict(s_grid=s_grid, rec_sub=np.array(rec_sub),
                      rec_leader=(Wstk - Wn) / (Wo - Wn))
    return out


# --------------------------------------------------------------------------- #
#  D. VCG comparison: efficient but budget-imbalanced
# --------------------------------------------------------------------------- #
def vcg_imbalance(sys):
    """Clarke (VCG) payments on the cooperative game with value v(S): each
    player's payment is its marginal externality v(N)-v(N\\{i}) ... summed and
    compared to the efficient surplus v(N) to expose the budget (im)balance."""
    N = set(range(sys.n))
    phi, v = sys.shapley()
    vN = v(N)
    # VCG 'value kept' by i = v(N) - v(N\{i})  (its marginal contribution)
    margins = {i: vN - v(N - {i}) for i in range(sys.n)}
    total_margin = sum(margins.values())
    # In a Groves/VCG scheme the sum of players' retained marginal values
    # generally != v(N): the gap is the budget imbalance.
    return dict(vN=vN, margins=margins, total_margin=total_margin,
                imbalance=total_margin - vN,
                imbalance_pct=100 * (total_margin - vN) / vN)


# --------------------------------------------------------------------------- #
#  E. gamma_max sensitivity to the v(S) baseline
# --------------------------------------------------------------------------- #
def gamma_max_baselines(sys):
    out = {}
    N = set(range(sys.n))
    for base in ("rmin", "nash", "rmax"):
        phi, v = sys.shapley(baseline=base)
        kap = v(N) - v(N - {sys.L})
        gmax = min((sys.n - 1) * (phi[j] - v({j})) / kap for j in sys.F)
        out[base] = dict(kappa_L=kap, gamma_max=gmax)
    return out


# --------------------------------------------------------------------------- #
#  F. AGREE complexity-weight sensitivity
# --------------------------------------------------------------------------- #
def agree_sensitivity(sys, R_target=0.95):
    import analysis2 as A2
    weights = {
        "steep (40,24,16,8)": np.array([40., 24., 16., 8.]),
        "mild (4,3,2,1)":     np.array([4., 3., 2., 1.]),
        "uniform (1,1,1,1)":  np.array([1., 1., 1., 1.]),
    }
    out = {}
    for name, w in weights.items():
        R = A2.alloc_agree(sys, R_target, complexity=w[:sys.n])
        out[name] = A2.cost_to_hit(sys, R)
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sys = satellite_eps_real()

    print("=" * 72)
    print("A. ANALYTICAL CONTRACTION BOUND")
    Lam, rows, rho = contraction_bound(sys)
    print(f"   analytical Lambda = {Lam:.4f}  (sufficient: <1)")
    print(f"   numerical rho     = {rho:.4f}")
    print(f"   per-follower rows : {[round(v,4) for v in rows.values()]}")

    print("=" * 72)
    print("B. UTILITY-FORM ROBUSTNESS (risk aversion eta)")
    for r in risk_aversion_scan():
        print(f"   eta={r['eta']:.2f}  Rsys: N={r['Rn']:.4f} S={r['Rs']:.4f} O={r['Ro']:.4f}"
              f"  PoA={r['poa']:.3f}  closed={100*r['closed']:.1f}%")

    print("=" * 72)
    print("C. FOLLOWER-SUBSIDY REMEDY")
    rem = subsidy_remedy()
    for n, d in rem.items():
        half = d['rec_sub'][len(d['s_grid'])//2]
        print(f"   n={n:2d}  leader-only recovers {100*d['rec_leader']:5.1f}%  |"
              f"  subsidy s=0.5 recovers {100*half:5.1f}%  |  s=1.0 -> {100*d['rec_sub'][-1]:.0f}%")

    print("=" * 72)
    print("D. VCG BUDGET IMBALANCE")
    vc = vcg_imbalance(sys)
    print(f"   v(N)={vc['vN']:.3f}  sum of marginal values={vc['total_margin']:.3f}"
          f"  imbalance={vc['imbalance']:.3f} ({vc['imbalance_pct']:.1f}% of v(N))")

    print("=" * 72)
    print("E. gamma_max SENSITIVITY TO v(S) BASELINE")
    for b, d in gamma_max_baselines(sys).items():
        print(f"   {b:5s}: kappa_L={d['kappa_L']:.3f}  gamma_max={d['gamma_max']:.3f}")

    print("=" * 72)
    print("F. AGREE COMPLEXITY-WEIGHT SENSITIVITY")
    for name, c in agree_sensitivity(sys).items():
        print(f"   {name:22s} total cost = {c:.1f}")
