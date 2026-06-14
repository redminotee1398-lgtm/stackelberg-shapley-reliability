# Stackelberg–Shapley Protocol for Reliability Allocation

Reproducibility code for the paper

> **A Stackelberg–Shapley Protocol for Individually-Rational Reliability Allocation in the Conceptual Design of Series Systems**

This repository contains the complete Python implementation, analysis modules, and figure-generation scripts. Every table and figure in the paper can be reproduced from a single entry point.

---

## What the paper does

In the conceptual design of a complex **series system** (e.g. a satellite Electrical Power System), each subsystem is built by a separate, self-interested team that controls its own reliability budget yet shares in the reliability of the integrated product. Because the series reliability

```
R_sys = ∏ R_i
```

benefits every team but is paid for locally, it behaves as a **public good**: decentralised choices cause free-riding and under-provision relative to the system optimum.

The paper introduces a four-stage **Stackelberg–Shapley protocol**:

1. A pivotal subsystem (the **leader**) moves first under a system-accountability weight `θ`.
2. The remaining teams reach a **Nash equilibrium** in a public-goods game.
3. The **Shapley value** of an induced cooperative characteristic function is computed.
4. A `γ`-parameterised, **budget-balanced transfer** rewards the leader in proportion to its measured criticality `κ_L = v(N) − v(N\{L})`, bounded by follower individual rationality.

The case study is anchored in published on-orbit failure statistics for the satellite EPS (Castet & Saleh 2009; Kim, Castet & Saleh 2012), with cost parameters set by a transparent **Weibull-informed calibration**.

**Headline, transparently reported findings:** a single leader closes only ≈10% of the Nash-to-social welfare gap; the rule is individually rational but *not* strategy-proof; the protocol's principal value is **allocative** (it lifts the pivotal team's fair share from 40.4% to 49.7% while keeping every team individually rational and the budget exactly balanced); and a Pigouvian follower subsidy recovers 80–95% of the gap in long chains, complementing the protocol.

---

## Repository structure

| File | Contents |
|------|----------|
| `stackelberg_shapley.py` | Core model: `Subsystem`, `SeriesSystem`, Nash / Stackelberg / social solvers, Shapley value, best-response Jacobian, the real satellite-EPS instance |
| `analysis.py` | θ-sweep, γ individual-rationality range, β-robustness, baseline robustness, allocation baselines |
| `analysis2.py` | Classic allocation methods (Equal, ARINC, AGREE, cost-optimal) and functional-form (power-law) robustness |
| `analysis3.py` | Analytical contraction bound, AGREE complexity-weight sensitivity, VCG-style budget deficit, risk-aversion, Pigouvian follower-subsidy remedy |
| `analysis4.py` | Weibull-informed cost calibration and the wide (α, β) sensitivity sweep |
| `analysis5.py` | Protocol + subsidy on the real EPS case, simultaneous misreporting, and a second independent case study |
| `figures.py`, `figures2.py` | All 18 figures (300 dpi) |

---

## Reproducing the results

Requires Python 3.10+ with `numpy`, `scipy`, and `matplotlib`.

```bash
# install dependencies
pip install numpy scipy matplotlib

# core numerical results
python stackelberg_shapley.py     # baseline EPS instance and regimes
python analysis.py                # theta, gamma-IR, robustness
python analysis2.py               # classic methods, functional-form robustness
python analysis3.py               # contraction, VCG, risk aversion, subsidy
python analysis4.py               # Weibull calibration + (alpha,beta) sweep
python analysis5.py               # protocol+subsidy on EPS, joint misreport, 2nd case

# regenerate all figures
python figures.py
python figures2.py
```

Each script prints the numbers that appear in the corresponding tables of the paper; figures are written as PNGs.

---

## Key results at a glance (satellite-EPS case)

| Quantity | Value |
|----------|-------|
| Nash system reliability | 0.772 |
| Social-optimum system reliability | 0.848 |
| Price of anarchy | 1.054 |
| Best-response contraction radius ρ | 0.044 |
| Leader criticality κ_L | 2.750 |
| Gap closed by leader alone | 9.9% |
| Gap closed by leader + subsidy (s = 0.5) | 89.5% |
| Leader fair share: plain Shapley → Stackelberg–Shapley | 40.4% → 49.7% |

---

## Citation

If you use this code, please cite the paper (full bibliographic details to be added on publication).

## License

Released under the [MIT License](LICENSE).
