# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week13_S4_seed3`
- constrained artifact stem: `week13_S4_seed3_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5829`
- holdout steps: `1458`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0607 | 0.0763 | +0.0156 |
| Max drawdown | -22.6707% | -21.5841% | +1.0866% |
| Mean return | 0.00022821 | 0.00028994 | +0.00006173 |
| Volatility | 0.00712312 | 0.00814067 | +0.00101755 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0201 | 0.0859 |
| Open | Mean return | 0.00008766 | 0.00034134 |
| Open | Volatility | 0.00797581 | 0.01018735 |
| Open | Max drawdown (subset) | -14.7134% | -14.2422% |
| Closed | Sortino | 0.0719 | 0.0740 |
| Closed | Mean return | 0.00025972 | 0.00027842 |
| Closed | Volatility | 0.00691716 | 0.00760660 |
| Closed | Max drawdown (subset) | -20.8604% | -19.5489% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0507`
- baseline max drawdown (full): `-22.6707%`

## Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 33.5% of total return
**Biggest domain:** `nba-champion` — 42.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.141591 | 33.5% | 0.0498 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.087563 | 20.7% | 0.0499 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.069342 | 16.4% | 0.0501 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067352 | 15.9% | 0.0499 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066702 | 15.8% | 0.0499 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.043646 | 10.3% | 0.0501 |
| 7 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.042857 | 10.1% | 0.0501 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.042109 | -10.0% | 0.0499 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041298 | -9.8% | 0.0499 |
| 10 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.039846 | 9.4% | 0.0499 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `nba-champion` | 0.181437 | 42.9% |
| 2 | `colombia-election` | 0.154265 | 36.5% |
| 3 | `serie-a` | 0.069342 | 16.4% |
| 4 | `airdrops` | 0.043646 | 10.3% |
| 5 | `uefa-europa-league` | 0.042857 | 10.1% |
| 6 | `roland-garros` | -0.038201 | -9.0% |
| 7 | `primaries` | -0.035531 | -8.4% |
| 8 | `awards` | 0.033172 | 7.8% |
| 9 | `hockey` | -0.030213 | -7.1% |
| 10 | `ligue-1` | 0.026054 | 6.2% |

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
- variance ratio constrained vs baseline: `0.6931`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0156)
- [x] Constrained holdout drawdown better than baseline (+1.0866%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 42.9%)
