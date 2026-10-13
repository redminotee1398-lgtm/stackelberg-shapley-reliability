# Stackelberg-Shapley protocol for decentralized reliability allocation

Python implementation and analysis scripts associated with the manuscript
**"A Stackelberg-Shapley Protocol for Individually-Rational Reliability Allocation in the Conceptual Design of Series Systems."**

## Scope of the repository

The code implements the public-good reliability model, simultaneous Nash and
accountability-weighted Stackelberg regimes, the induced characteristic-function
game and Shapley allocation, the participation-constrained leader premium,
robustness/calibration experiments, fixed-target allocation comparisons,
synthetic scalability tests, and the corrected **follower-only subsidy combined
with the accountable Stackelberg leader**.

The satellite Electrical Power System (EPS) illustration is **data-informed**:
published reliability evidence motivates the subsystem decomposition and the
choice of Solar-Array Operating as the leader.  The normalized stake and
reliability-hardening cost parameters remain modelling inputs rather than a
fully empirical economic calibration.

## Important interpretation notes

- The Jacobian spectral radius `rho ~= 0.044` is a **local convergence diagnostic**.
  The manuscript's global sufficient uniqueness certificate is the conservative
  boxwise sup-norm contraction bound `Lambda_box ~= 0.411` implemented in
  `analysis3.py`.
- The cost-rate perturbation experiment is a **reporting-parameter sensitivity
  analysis**, not a strategy-proofness test.
- The fixed-target "AGREE" calculation available in the legacy code does not
  contain all mission-time and importance inputs required for a full AGREE
  implementation; it is therefore reported as a **complexity-weighted proxy**.
- Subsidy results are **gross engineering welfare recovery before financing and
  administrative costs**.  The corrected experiment subsidizes followers only
  and retains the accountable Stackelberg leader.

## Repository structure

| File | Main purpose |
|---|---|
| `stackelberg_shapley.py` | Core model, Nash/Stackelberg/social solvers, characteristic function, Shapley value, EPS case |
| `analysis.py` | Accountability sweep, participation range, calibration/baseline sensitivity, allocation comparisons |
| `analysis2.py` | Fixed-target allocation comparisons, power-law robustness, reporting-input sensitivity, scalability |
| `analysis3.py` | Global contraction bound, concave valuation, follower-only subsidy, marginal-contribution budget diagnostic, baseline sensitivity |
| `analysis4.py` | Weibull-informed calibration and wide cost-parameter sweep |
| `analysis5.py` | Real-EPS leader+subsidy table, joint input perturbation, second synthetic case |
| `figures.py`, `figures2.py` | Computational figure-generation scripts |
| `verify_results.py` | Numerical regression checks for the headline manuscript values |

## Environment

Python 3.10+ with:

```bash
pip install numpy scipy matplotlib
```

## Running the analyses

```bash
python stackelberg_shapley.py
python analysis.py
python analysis2.py
python analysis3.py
python analysis4.py
python analysis5.py
python verify_results.py

# computational figures
python figures.py
python figures2.py
```

Figure files are written to `./figs`; the directory is created automatically.

## Headline numerical checks for the EPS illustration

| Quantity | Verified value |
|---|---:|
| Nash system reliability | 0.7722847 |
| Accountable Stackelberg system reliability (`theta=0.70`) | 0.7815708 |
| Social-optimum system reliability | 0.8480909 |
| Nash welfare | 5.6263815 |
| Stackelberg welfare | 5.6565236 |
| Social-optimum welfare | 5.9321853 |
| Welfare gap closed by leader alone | 9.8567% |
| Local follower-BR Jacobian spectral radius | 0.0443 |
| Global boxwise contraction bound | 0.4105 |
| Leader criticality (`Rmin` projection) | 2.75014 |
| Leader share, Shapley -> premium allocation (`gamma=0.20`) | 40.43% -> 49.70% |
| `gamma_max`, `Rmin` projection | 0.24548 |
| `gamma_max`, Nash projection | 0.08005 |
| Positive common premium under `Rmax` projection | none |
| Leader + follower-only subsidy, `s=0.25` | 71.30% gross gap recovery |
| Leader + follower-only subsidy, `s=0.50` | 90.70% gross gap recovery |
| Leader + follower-only subsidy, `s=0.75` | 97.84% gross gap recovery |
| Leader + follower-only subsidy, `s=1.00` | 99.56% gross gap recovery |

## Data/code statement

The repository contains the Python source code used for the numerical analyses
reported in the manuscript.  Numerical reproducibility verifies the
implementation of the stated model; it does not by itself validate the modelling
assumptions or turn normalized parameters into measured economic data.

## Citation

Please cite the manuscript; full bibliographic details can be added after publication.
