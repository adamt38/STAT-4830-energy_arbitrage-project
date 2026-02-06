# Report draft 1 — Week 4 (STAT 4830)

## Problem (snapshot)

Optimize a Polymarket contract portfolio for risk-adjusted return with explicit penalties on domain concentration, versus an equal-weight baseline.

## Approach (snapshot)

- Data: Gamma API + CLOB history via `src/polymarket_data.py`.
- Baseline: `src/baseline.py` (equal weight, Sortino, drawdown, domain exposure).
- Optimizer: `src/constrained_optimizer.py` — rolling-window updates, Sortino-style objective with domain penalty, small grid over learning rate / λ / window.

## Results (snapshot)

Prototype slice (~19 markets): baseline Sortino ~0.07; best constrained grid point **underperformed** on Sortino and showed much worse drawdown than baseline — establishing that the first penalty settings were unstable.

## Next steps (snapshot)

Expand universe, improve tag/domain balance, and tighten evaluation (walk-forward, less uniform-mix dilution).

*Archived milestone text — see root `report.md` for the full final paper.*
