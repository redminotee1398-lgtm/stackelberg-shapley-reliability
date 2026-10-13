#!/usr/bin/env python3
"""Regression checks for the numerical values reported in the synchronized manuscript."""
import math
import numpy as np
from stackelberg_shapley import satellite_eps_real
import analysis3 as A3
import analysis5 as A5

THETA = 0.70
GAMMA = 0.20

def close(name, got, expected, atol=5e-6):
    if not np.isclose(got, expected, atol=atol, rtol=0):
        raise AssertionError(f"{name}: got {got:.10g}, expected {expected:.10g} +/- {atol}")
    print(f"PASS  {name:46s} {got:.8f}")

sys = satellite_eps_real()
Rn = sys.solve_nash()
Rs = sys.solve_stackelberg(theta=THETA)
Ro = sys.solve_social()
Wn, Ws, Wo = sys.welfare(Rn), sys.welfare(Rs), sys.welfare(Ro)

close("Nash R_sys", sys.R_sys(Rn), 0.7722847412)
close("Stackelberg R_sys", sys.R_sys(Rs), 0.7815708259)
close("Social-optimum R_sys", sys.R_sys(Ro), 0.8480909299)
close("Nash welfare", Wn, 5.6263814775)
close("Stackelberg welfare", Ws, 5.6565236061)
close("Social-optimum welfare", Wo, 5.9321853250)
close("Leader-only welfare gap closed", 100*(Ws-Wn)/(Wo-Wn), 9.8566872, atol=2e-4)

m = sys.metrics(GAMMA, theta=THETA)
close("Leader criticality kappa_L", m["kappa_L"], 2.7501417628)
close("Plain-Shapley leader share (%)", 100*m["phi"][sys.L]/m["phi"].sum(), 40.4283882, atol=2e-4)
close("Premium leader share (%)", 100*m["phi_stk"][sys.L]/m["phi_stk"].sum(), 49.7003230, atol=2e-4)

rho, _ = sys.br_jacobian_spectral_radius(Rn)
close("Local BR Jacobian spectral radius", rho, 0.04433, atol=2e-4)
Lam, rows, rho2 = A3.contraction_bound(sys)
close("Global boxwise contraction bound", Lam, 0.41050142, atol=2e-5)
if not Lam < 1:
    raise AssertionError("Global contraction bound must be < 1")

bases = A3.gamma_max_baselines(sys)
close("gamma_max under Rmin projection", bases["rmin"]["gamma_max"], 0.24548154, atol=2e-5)
close("gamma_max under Nash projection", bases["nash"]["gamma_max"], 0.08005209, atol=2e-5)
if not bases["rmax"]["gamma_max"] <= 0:
    raise AssertionError("Rmax projection should admit no positive common premium in this instance")
print(f"PASS  {'Rmax positive common premium':46s} none")

leader_alone, rows = A5.eps_protocol_plus_subsidy()
expected = {0.00: 9.8566872, 0.25: 71.2982586, 0.50: 90.6980700, 0.75: 97.8433620, 1.00: 99.5596487}
for s, frac in rows:
    close(f"Leader + follower-only subsidy s={s:.2f} (%)", 100*frac, expected[round(float(s),2)], atol=3e-4)

# Synthetic Figure-15 curves: check endpoints and s=0.5 for deterministic seed/scaling.
rem = A3.subsidy_remedy(n_list=(4, 8, 12), s_grid=np.linspace(0,1,11))
checks = {
    4: (9.18, 90.54, 99.59),
    8: (5.70, 93.31, 99.84),
    12: (5.05, 94.55, 99.90),
}
for n,(v0,v5,v1) in checks.items():
    arr = 100*rem[n]["rec_sub"]
    close(f"Synthetic n={n}: s=0 (%)", arr[0], v0, atol=0.02)
    close(f"Synthetic n={n}: s=0.5 (%)", arr[5], v5, atol=0.02)
    close(f"Synthetic n={n}: s=1 (%)", arr[-1], v1, atol=0.02)

print("\nALL SYNCHRONIZED NUMERICAL CHECKS PASSED")
