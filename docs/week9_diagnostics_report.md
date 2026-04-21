# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week13_S1`
- constrained artifact stem: `week13_S1_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5829`
- holdout steps: `1458`
- objective: `1.5`-var / `2.3`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0607 | 0.0756 | +0.0149 |
| Max drawdown | -22.6707% | -21.4885% | +1.1822% |
| Mean return | 0.00022821 | 0.00028901 | +0.00006080 |
| Volatility | 0.00712312 | 0.00815020 | +0.00102708 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0201 | 0.0813 |
| Open | Mean return | 0.00008766 | 0.00032676 |
| Open | Volatility | 0.00797581 | 0.01011669 |
| Open | Max drawdown (subset) | -14.7134% | -14.2337% |
| Closed | Sortino | 0.0719 | 0.0743 |
| Closed | Mean return | 0.00025972 | 0.00028055 |
| Closed | Volatility | 0.00691716 | 0.00764018 |
| Closed | Max drawdown (subset) | -20.8604% | -19.5117% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0507`
- baseline max drawdown (full): `-22.6707%`

## Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 33.0% of total return
**Biggest domain:** `nba-champion` — 42.2% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.138857 | 33.0% | 0.0498 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.088946 | 21.1% | 0.0500 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.069612 | 16.5% | 0.0505 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.066373 | 15.8% | 0.0500 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066094 | 15.7% | 0.0500 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.044396 | 10.5% | 0.0503 |
| 7 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.043158 | 10.2% | 0.0503 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.042090 | -10.0% | 0.0497 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041213 | -9.8% | 0.0497 |
| 10 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.039106 | 9.3% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `nba-champion` | 0.177963 | 42.2% |
| 2 | `colombia-election` | 0.155040 | 36.8% |
| 3 | `serie-a` | 0.069612 | 16.5% |
| 4 | `airdrops` | 0.044396 | 10.5% |
| 5 | `uefa-europa-league` | 0.043158 | 10.2% |
| 6 | `roland-garros` | -0.038179 | -9.1% |
| 7 | `primaries` | -0.035069 | -8.3% |
| 8 | `awards` | 0.033077 | 7.8% |
| 9 | `hockey` | -0.030481 | -7.2% |
| 10 | `ligue-1` | 0.025160 | 6.0% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3929 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1686 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |
| Will the Portland Trail Blazers win the 2026 NBA Finals? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0936 |
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0416 |

## Correlation and Risk Structure
- category count: `15`
- avg abs category correlation: `0.0024`
- max abs category correlation: `0.0613`
- top eigenvalue share: `0.8198`
- variance ratio constrained vs baseline: `0.6922`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0149)
- [x] Constrained holdout drawdown better than baseline (+1.1822%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 42.2%)
