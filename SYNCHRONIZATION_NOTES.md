# AEJ synchronization corrections

This package addresses the numerical/reproducibility issues identified in the code audit.

## Manuscript changes

- Table 3: Nash-projection `gamma_max` corrected from 0.075 to 0.080; feasible leader share updated from 44.8% to 45.0%.
- Figure 11: the hard-coded 92% residual reliability gap is replaced by the calculated value, 85.1% at `theta=1`.
- Section 6.4/Table 10/Figure 15: the experiment is now genuinely **accountable Stackelberg leader + follower-only subsidy**.
  - leader only: 9.9%
  - `s=0.25`: 71.3%
  - `s=0.50`: 90.7%
  - `s=0.75`: 97.8%
  - `s=1.00`: 99.6%
- Data availability wording now states that the repository contains the Python source code used for the reported analyses; it no longer says that post-hoc code was written to "regenerate" the study.

## Repository changes applied by `apply_repository_fixes.py`

1. `stackelberg_shapley.py`
   - clarifies that `rho` is a local Jacobian diagnostic, not a global uniqueness proof;
   - command-line execution now uses `satellite_eps_real()`, `theta=0.70`, `gamma=0.20`.
2. `analysis.py`
   - relabels the spectral-radius result as a local diagnostic.
3. `analysis2.py`
   - reinterprets the beta-report experiment as reporting-parameter sensitivity;
   - labels the incomplete AGREE calculation as a complexity-weighted proxy.
4. `analysis3.py`
   - replaces the pointwise contraction quantity with the conservative global boxwise bound (about 0.4105);
   - implements follower-only subsidy;
   - implements accountable Stackelberg leader + follower-only subsidy for the manuscript experiment.
5. `analysis5.py`
   - Table-10 computation now calls the combined leader+follower-subsidy solver.
6. `figures.py`
   - creates the `figs/` output directory automatically.
7. `figures2.py`
   - Figure 11 residual percentage is calculated, not hard-coded;
   - Figure 10 is labeled as reporting sensitivity, not a strategy-proofness test;
   - Figure 15 uses the corrected combined subsidy experiment;
   - subsidy and cost-grid figures are called from `__main__`.
8. `README.md`
   - removes overclaims and updates all headline values and interpretations.

## Verification

After patching the repository, copy `verify_results.py` into the repository root and run:

```bash
python verify_results.py
```

The script contains regression assertions for the central Nash, Stackelberg, social-optimum, Shapley, contraction, participation, and subsidy results.
