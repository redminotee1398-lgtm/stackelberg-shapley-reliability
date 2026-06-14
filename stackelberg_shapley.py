"""
Stackelberg-Shapley Protocol for Reliability Optimization and Resource Allocation
in the Conceptual Design of Complex Engineering Systems.

This module implements the full four-stage protocol:
  Stage 1 (Stackelberg) : a pivotal *leader* subsystem commits first.
  Stage 2 (Nash)        : *follower* subsystems play a simultaneous Nash game.
  Stage 3 (Shapley)     : the cooperative surplus is divided by the Shapley value.
  Stage 4 (Allocation)  : a gamma-controlled strategic premium re-weights the
                          leader's share (efficiency-preserving).

Modelling stance
----------------
A series system's reliability  R_sys = prod_i R_i  is a *public good* shared by
the subsystem design teams: every team benefits from R_sys but each pays only its
own reliability-growth cost.  This produces classic free-riding and under-provision
at the Nash equilibrium.  Because the reliabilities are *strategic complements*
(d^2 u_i / dR_i dR_j > 0), a first-moving leader that commits to a higher
reliability pulls the followers upward, raising welfare -- the "strategic advantage".

Author: prepared for submission to IEEE Trans. Reliability / RESS.
"""

import itertools
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# ----------------------------------------------------------------------------- #
#  1.  MODEL PRIMITIVES
# ----------------------------------------------------------------------------- #


class Subsystem:
    """A single subsystem (design team) of a series reliability system."""

    def __init__(self, name, alpha, beta, pi, Rmin, Rmax, role="follower"):
        self.name = name
        self.alpha = alpha      # cost scale
        self.beta = beta        # cost growth (small -> finite, convex)
        self.pi = pi            # stake / value derived from system reliability
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.role = role

    # exponential reliability-cost function  C_i(R) = a * exp(b * R/(1-R))
    def cost(self, R):
        R = np.clip(R, self.Rmin, min(self.Rmax, 0.999999))
        return self.alpha * np.exp(self.beta * R / (1.0 - R))

    def dcost(self, R):
        R = np.clip(R, self.Rmin, min(self.Rmax, 0.999999))
        g = self.beta * R / (1.0 - R)
        gp = self.beta / (1.0 - R) ** 2
        return self.alpha * np.exp(g) * gp


class SeriesSystem:
    """Series reliability system + Stackelberg-Shapley solver."""

    def __init__(self, subsystems, leader_index=0):
        self.subs = subsystems
        self.n = len(subsystems)
        self.L = leader_index
        self.F = [i for i in range(self.n) if i != leader_index]

    # ---- reliability & welfare ------------------------------------------- #
    def R_sys(self, R):
        return float(np.prod(R))

    def utility(self, i, R):
        Rsys = self.R_sys(R)
        return self.subs[i].pi * Rsys - self.subs[i].cost(R[i])

    def welfare(self, R):
        return sum(self.utility(i, R) for i in range(self.n))

    # ---- best response of a single player -------------------------------- #
    def best_response(self, i, R):
        """Maximise u_i over R_i with others fixed:  C_i'(R_i) = pi_i * prod_{j!=i} R_j."""
        P_minus_i = np.prod([R[j] for j in range(self.n) if j != i])
        sub = self.subs[i]
        target = sub.pi * P_minus_i
        lo, hi = sub.Rmin, min(sub.Rmax, 0.999999)
        # C_i' is strictly increasing; solve C_i'(R)=target, else clip to bounds
        if sub.dcost(lo) >= target:
            return lo
        if sub.dcost(hi) <= target:
            return hi
        return brentq(lambda R: sub.dcost(R) - target, lo, hi, xtol=1e-10)

    # ---- Nash equilibrium among a set of players (others fixed) ---------- #
    def nash(self, players, R0, fixed=None, tol=1e-9, max_iter=500, record=False):
        """Best-response iteration over `players`; entries not in `players`
        keep their value from `fixed` (or R0)."""
        R = np.array(R0, dtype=float)
        if fixed is not None:
            for k, v in fixed.items():
                R[k] = v
        history = [R.copy()]
        for _ in range(max_iter):
            R_new = R.copy()
            for i in players:
                R_new[i] = self.best_response(i, R_new)
            history.append(R_new.copy())
            if np.max(np.abs(R_new - R)) < tol:
                R = R_new
                break
            R = R_new
        return (R, history) if record else R

    # ---- regime solvers --------------------------------------------------- #
    def solve_nash(self, record=False):
        R0 = np.array([s.Rmin for s in self.subs], dtype=float)
        return self.nash(list(range(self.n)), R0, record=record)

    def solve_social(self):
        """Social optimum: every team internalises the *total* stake sum(pi)."""
        Pi = sum(s.pi for s in self.subs)
        R = np.array([s.Rmin for s in self.subs], dtype=float)
        for _ in range(500):
            R_new = R.copy()
            for i in range(self.n):
                P = np.prod([R_new[j] for j in range(self.n) if j != i])
                sub = self.subs[i]
                target = Pi * P
                lo, hi = sub.Rmin, min(sub.Rmax, 0.999999)
                if sub.dcost(lo) >= target:
                    R_new[i] = lo
                elif sub.dcost(hi) <= target:
                    R_new[i] = hi
                else:
                    R_new[i] = brentq(lambda r: sub.dcost(r) - target, lo, hi, xtol=1e-10)
            if np.max(np.abs(R_new - R)) < 1e-9:
                R = R_new
                break
            R = R_new
        return R

    def followers_nash(self, RL, R_warm=None):
        """Nash equilibrium of the followers, given leader reliability RL fixed."""
        R0 = R_warm.copy() if R_warm is not None else np.array([s.Rmin for s in self.subs])
        R0[self.L] = RL
        return self.nash(self.F, R0, fixed={self.L: RL})

    def leader_objective(self, RL, theta=0.0):
        """Leader's objective with system-accountability weight theta in [0,1]:
        theta=0 -> purely self-interested (classic Stackelberg);
        theta=1 -> leader fully internalises system welfare (architectural keystone)."""
        R = self.followers_nash(RL)
        return theta * self.welfare(R) + (1.0 - theta) * self.utility(self.L, R)

    def solve_stackelberg(self, theta=0.6):
        """Leader optimises its (accountability-weighted) objective anticipating
        the followers' Nash response."""
        lo, hi = self.subs[self.L].Rmin, min(self.subs[self.L].Rmax, 0.999999)
        res = minimize_scalar(lambda RL: -self.leader_objective(RL, theta),
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-7})
        RL = res.x
        R = self.followers_nash(RL)
        return R

    # ---- contraction diagnostic ------------------------------------------ #
    def br_jacobian_spectral_radius(self, R=None, eps=1e-6):
        """Spectral radius of the followers' best-response Jacobian at R.
        rho < 1  <=>  the best-response map is a local contraction, so the
        follower Nash equilibrium is unique and best-response iteration
        converges geometrically (verifies the hypothesis of Prop. 1)."""
        if R is None:
            R = self.solve_nash()
        R = np.array(R, dtype=float)
        F = self.F
        m = len(F)

        def br_vec(Rin):
            out = np.empty(m)
            for k, i in enumerate(F):
                out[k] = self.best_response(i, Rin)
            return out

        J = np.zeros((m, m))
        base = br_vec(R)
        for c, j in enumerate(F):
            Rp = R.copy()
            Rp[j] += eps
            bp = br_vec(Rp)
            J[:, c] = (bp - base) / eps
        rho = max(abs(np.linalg.eigvals(J)))
        return float(rho), J

    # ---- characteristic function & Shapley ------------------------------- #
    def _baseline_R(self, baseline):
        """Reliability vector used for *non-members* of a coalition.
        baseline in {'rmin','nash','rmax'} -- supports robustness analysis of
        the characteristic-function convention (reviewer concern)."""
        if baseline == "rmin":
            return np.array([s.Rmin for s in self.subs], dtype=float)
        if baseline == "rmax":
            return np.array([min(s.Rmax, 0.999999) for s in self.subs], dtype=float)
        if baseline == "nash":
            return self.solve_nash()
        raise ValueError(baseline)

    def coalition_value(self, S, baseline="rmin"):
        """v(S): members of S cooperatively maximise their joint benefit while
        non-members sit at a fixed `baseline` reliability.  Only members'
        stakes are captured by S.  The baseline is a *convention*; we expose it
        so the allocative conclusions can be stress-tested against it."""
        S = list(S)
        if not S:
            return 0.0
        Pi_S = sum(self.subs[i].pi for i in S)
        Rbase = self._baseline_R(baseline)
        base_out = np.prod([Rbase[j] for j in range(self.n) if j not in S]) \
            if len(S) < self.n else 1.0

        # cooperative optimum over members (block best-response; concave enough)
        R = np.array([self.subs[i].Rmin for i in range(self.n)], dtype=float)
        for _ in range(500):
            R_new = R.copy()
            for i in S:
                P_in = np.prod([R_new[j] for j in S if j != i])
                sub = self.subs[i]
                target = Pi_S * P_in * base_out
                lo, hi = sub.Rmin, min(sub.Rmax, 0.999999)
                if sub.dcost(lo) >= target:
                    R_new[i] = lo
                elif sub.dcost(hi) <= target:
                    R_new[i] = hi
                else:
                    R_new[i] = brentq(lambda r: sub.dcost(r) - target, lo, hi, xtol=1e-10)
            if np.max(np.abs(R_new - R)) < 1e-10:
                R = R_new
                break
            R = R_new
        prod_in = np.prod([R[i] for i in S])
        benefit = Pi_S * prod_in * base_out - sum(self.subs[i].cost(R[i]) for i in S)
        return max(benefit, 0.0)

    def shapley(self, baseline="rmin"):
        n = self.n
        phi = np.zeros(n)
        players = list(range(n))
        # exact Shapley via subset enumeration (n is small in design problems)
        vcache = {}

        def v(S):
            key = frozenset(S)
            if key not in vcache:
                vcache[key] = self.coalition_value(S, baseline=baseline)
            return vcache[key]

        for i in players:
            others = [p for p in players if p != i]
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    w = (math.factorial(len(S)) *
                         math.factorial(n - len(S) - 1) /
                         math.factorial(n))
                    phi[i] += w * (v(set(S) | {i}) - v(set(S)))
        return phi, v

    # ---- three strategic metrics ----------------------------------------- #
    def metrics(self, gamma, theta=0.6):
        R_nash = self.solve_nash()
        R_stk = self.solve_stackelberg(theta=theta)
        phi, v = self.shapley()

        # Metric 1: system strategic advantage
        delta_strategic = self.R_sys(R_stk) - self.R_sys(R_nash)
        # Metric 2: leader criticality (Shapley-type)
        N = set(range(self.n))
        kappa_L = v(N) - v(N - {self.L})
        # Stackelberg-Shapley allocation (efficiency preserving)
        phi_stk = phi.copy()
        phi_stk[self.L] = phi[self.L] + gamma * kappa_L
        for j in self.F:
            phi_stk[j] = phi[j] - gamma * kappa_L / (self.n - 1)
        # Metric 3: leader entitlement (premium)
        psi_L = phi_stk[self.L] - phi[self.L]

        return dict(R_nash=R_nash, R_stk=R_stk, phi=phi, phi_stk=phi_stk,
                    delta_strategic=delta_strategic, kappa_L=kappa_L, psi_L=psi_L,
                    v=v)


# ----------------------------------------------------------------------------- #
#  2.  CASE STUDY:  SATELLITE ELECTRICAL POWER SYSTEM (EPS)
# ----------------------------------------------------------------------------- #
def satellite_eps():
    """Four series subsystems of a satellite electrical power system.
    The Solar-Array / Power-Conditioning unit is the pivotal LEADER:
    it is the single power source feeding all downstream units and carries
    the largest mission stake."""
    subs = [
        #          name                      alpha   beta    pi     Rmin   Rmax   role
        Subsystem("Solar Array & PCU (L)",   0.020,  0.090,  3.4,  0.80, 0.999, "leader"),
        Subsystem("Battery Assembly",        0.030,  0.075,  1.7,  0.80, 0.999),
        Subsystem("Power Distribution",      0.028,  0.070,  1.5,  0.80, 0.999),
        Subsystem("Avionics Power I/F",      0.025,  0.065,  1.3,  0.80, 0.999),
    ]
    return SeriesSystem(subs, leader_index=0)


def satellite_eps_real():
    """Data-grounded satellite EPS, decomposed into the four constituents
    tracked in the SpaceTrak/Seradata on-orbit failure database and analysed by
    Castet & Saleh (RESS 94(11), 2009) and Kim, Castet & Saleh (RESS 98(1),
    2012): Solar-Array Operating (SAO), Battery/Cell, Electrical Distribution
    (ED), and Solar-Array Deployment (SAD).  EPS is the largest single driver of
    spacecraft unreliability (>25% of all on-orbit failures; ~41% after 5 yr),
    and SAO is the dominant culprit among EPS constituents -- the empirical
    basis for designating SAO the leader.  Stakes pi_i follow the published
    contribution ranking SAO > Battery > ED > SAD; cost rates beta_i are in the
    finite small-rate regime; reliability bounds reflect mature space-EPS
    operating ranges."""
    subs = [
        #          name                          alpha   beta    pi    Rmin   Rmax    role
        Subsystem("Solar-Array Operating (L)",   0.024,  0.088,  3.2,  0.90, 0.9995, "leader"),
        Subsystem("Battery / Cell",              0.032,  0.078,  1.9,  0.90, 0.9995),
        Subsystem("Electrical Distribution",     0.027,  0.071,  1.5,  0.90, 0.9995),
        Subsystem("Solar-Array Deployment",      0.020,  0.060,  1.1,  0.90, 0.9995),
    ]
    return SeriesSystem(subs, leader_index=0)


if __name__ == "__main__":
    sys = satellite_eps()
    GAMMA = 0.35
    THETA = 0.6

    print("=" * 74)
    print(" STACKELBERG-SHAPLEY RELIABILITY PROTOCOL -- Satellite EPS case study")
    print("=" * 74)

    R_nash = sys.solve_nash()
    R_stk = sys.solve_stackelberg(theta=THETA)
    R_soc = sys.solve_social()

    def row(tag, R):
        print(f"{tag:14s} | R_sys={sys.R_sys(R):.5f} | W={sys.welfare(R):7.4f} | "
              + " ".join(f"{r:.4f}" for r in R))

    print("\n-- Reliability regimes -------------------------------------------")
    print(f"{'regime':14s} |   R_sys   |    W    | per-subsystem R_i")
    row("Nash", R_nash)
    row("Stackelberg", R_stk)
    row("Social opt.", R_soc)

    paoR = sys.R_sys(R_stk) - sys.R_sys(R_nash)
    poa = sys.welfare(R_soc) / sys.welfare(R_nash)
    print(f"\nStrategic advantage  Delta_strategic = {paoR:.5f}"
          f"  ({100*paoR/sys.R_sys(R_nash):.2f}% reliability gain)")
    print(f"Price of Anarchy (W_social/W_nash)   = {poa:.4f}")

    m = sys.metrics(GAMMA, theta=THETA)
    print("\n-- Cooperative allocation (gamma = %.2f) --------------------------" % GAMMA)
    print(f"{'subsystem':22s} | Shapley phi_i | Stk-Shapley | share%")
    for i, s in enumerate(sys.subs):
        print(f"{s.name:22s} | {m['phi'][i]:11.4f}   | {m['phi_stk'][i]:9.4f}  "
              f" | {100*m['phi_stk'][i]/m['phi_stk'].sum():5.1f}")
    print(f"{'TOTAL':22s} | {m['phi'].sum():11.4f}   | {m['phi_stk'].sum():9.4f}")

    print("\n-- Three strategic metrics ---------------------------------------")
    print(f"Delta_strategic (system advantage) = {m['delta_strategic']:.5f}")
    print(f"kappa_L         (leader criticality)= {m['kappa_L']:.5f}")
    print(f"psi_L           (leader entitlement)= {m['psi_L']:.5f}")
    sh0 = m['phi'][sys.L] / m['phi'].sum()
    sh1 = m['phi_stk'][sys.L] / m['phi_stk'].sum()
    print(f"Leader share: {100*sh0:.1f}%  ->  {100*sh1:.1f}%  (Shapley -> Stk-Shapley)")
