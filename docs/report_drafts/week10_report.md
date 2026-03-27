# Report draft 4 — Week 10 (STAT 4830)

## Problem

After extensive MVO / Sortino-family experiments, equal-weight remained a strong null. We reframed the question toward **geometric growth** and binary payoffs.

## Approach

- **Kelly + dynamic Gaussian copula** (`src/kelly_copula_optimizer.py`, `script/polymarket_week10_kelly_pipeline.py`): expected log-wealth, MLP-emitting correlation state, straight-through Bernoulli samples, Adam-OGD on the simplex.
- Retained 40-market balanced universe and walk-forward holdout discipline from Week 8+ pipelines.

## Results (directional)

- K10-family runs (A/B/C) established that **K10C** (full stack) could improve **gross** log-wealth versus baseline on the main holdout window — with caveats on turnover and fees left for Round 7 post-hocs.

## Next steps

Fee re-ranking, fractional-Kelly α diagnostic, circular-block bootstrap on Δ log-wealth; optional GPU sweeps (K10E/K10F) for fee- and DD-aware training.

*Archived milestone text — see root `report.md` for the full final paper.*
