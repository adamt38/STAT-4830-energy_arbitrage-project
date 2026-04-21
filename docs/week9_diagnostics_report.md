# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week13_S5`
- constrained artifact stem: `week13_S5_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5634`
- holdout steps: `1409`
- objective: `1.1`-var / `2.8`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0755 | 0.0843 | +0.0088 |
| Max drawdown | -17.3682% | -18.8047% | -1.4365% |
| Mean return | 0.00028744 | 0.00032649 | +0.00003905 |
| Volatility | 0.00731503 | 0.00830711 | +0.00099209 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.2%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0261 | 0.0909 |
| Open | Mean return | 0.00011755 | 0.00037560 |
| Open | Volatility | 0.00822882 | 0.01046951 |
| Open | Max drawdown (subset) | -13.0319% | -12.8598% |
| Closed | Sortino | 0.0895 | 0.0827 |
| Closed | Mean return | 0.00032517 | 0.00031559 |
| Closed | Volatility | 0.00709564 | 0.00774549 |
| Closed | Max drawdown (subset) | -16.7868% | -16.4467% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0486`
- baseline max drawdown (full): `-17.9351%`

## Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 23.6% of total return
**Biggest domain:** `colombia-election` — 32.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.108447 | 23.6% | 0.0477 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.086227 | 18.7% | 0.0497 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.072141 | 15.7% | 0.0509 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067507 | 14.7% | 0.0497 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.065172 | 14.2% | 0.0497 |
| 6 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.046446 | 10.1% | 0.0511 |
| 7 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.045027 | 9.8% | 0.0509 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.044667 | -9.7% | 0.0493 |
| 9 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.040633 | 8.8% | 0.0495 |
| 10 | Will Allen Waters be the Republican nominee for Senate in Rhode Island? | `rhode-island-primary` | 0.036683 | 8.0% | 0.0506 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.151399 | 32.9% |
| 2 | `nba-champion` | 0.149080 | 32.4% |
| 3 | `serie-a` | 0.072141 | 15.7% |
| 4 | `uefa-europa-league` | 0.046446 | 10.1% |
| 5 | `airdrops` | 0.045027 | 9.8% |
| 6 | `rhode-island-primary` | 0.036683 | 8.0% |
| 7 | `roland-garros` | -0.036179 | -7.9% |
| 8 | `ligue-1` | 0.035965 | 7.8% |
| 9 | `hockey` | -0.031296 | -6.8% |
| 10 | `stanley-cup` | -0.028683 | -6.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3929 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1685 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |
| Will the Portland Trail Blazers win the 2026 NBA Finals? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0919 |
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0408 |

## Correlation and Risk Structure
- category count: `15`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.0613`
- top eigenvalue share: `0.8332`
- variance ratio constrained vs baseline: `0.6902`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0088)
- [ ] Constrained holdout drawdown better than baseline (-1.4365%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 32.9%)
