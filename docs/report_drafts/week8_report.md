# Report draft 3 — Week 8 (STAT 4830)

## Problem

Hidden **event clustering**: many contracts move on a few latent drivers. We test whether domain-capped optimization improves risk-adjusted returns versus a **category-equal** baseline.

## Approach

- `script/polymarket_week8_pipeline.py`: build → baseline → constrained experiments → figures.
- Balanced selection across many tag categories; equal **total** weight per category at baseline.
- Iteration figures: `figures/week8_iteration_*`.

## Results (representative)

- Larger universe (order ~80 markets / categories in reporting runs).
- Baseline: improved drawdown profile versus tiny early prototype.
- Constrained: still struggled to beat baseline on holdout Sortino when tuned fairly; variance-ratio diagnostics flagged over-concentration in correlated shocks.

## Next steps

Week 9+ optimizer overhaul (mean–downside objective, Optuna tuning, macro integration experiments documented in `docs/development_log.md`).

*Archived milestone text — see root `report.md` for the full final paper.*
