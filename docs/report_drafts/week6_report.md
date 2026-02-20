# Report draft 2 — Week 6 (STAT 4830)

## Problem

Same core question: beat equal-weight on Polymarket with differentiable domain constraints, now with a cleaner data slice and reproducible `script/polymarket_week8_pipeline.py` orchestration.

## Approach

- Category-aware sampling and stricter binary Yes/No filtering.
- Frozen-vs-online evaluation toggle for defensible holdout reporting.
- Continued grid search on constrained optimizer; documentation of lookahead fixes in the data alignment path.

## Results

Week 8-scale iteration (early): expanded categories; baseline Sortino improved on some slices, but constrained optimizer still **did not** reliably beat equal-weight on holdout Sortino / drawdown. Covariance diagnostics suggested residual concentration in correlated risk.

## Next steps

Replace Sortino-ratio gradient target with a **mean–downside surrogate**, increase inner steps per window, reduce uniform-mix floor, and migrate toward **projected-simplex** weights for sparse corners.

*Archived milestone text — see root `report.md` for the full final paper.*
