# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_G`
- constrained artifact stem: `week9_G`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5315`
- holdout steps: `1329`
- objective: `1.5`-var / `2.6`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.2237 | 0.2576 | +0.0339 |
| Max drawdown | -22.4972% | -21.2629% | +1.2343% |
| Mean return | 0.00110812 | 0.00126077 | +0.00015265 |
| Volatility | 0.02407253 | 0.02449667 | +0.00042414 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `14.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.4254 | 0.5845 |
| Open | Mean return | 0.00188330 | 0.00266327 |
| Open | Volatility | 0.02008508 | 0.02986860 |
| Open | Max drawdown (subset) | -7.4674% | -7.5653% |
| Closed | Sortino | 0.1935 | 0.2060 |
| Closed | Mean return | 0.00097482 | 0.00101960 |
| Closed | Volatility | 0.02469096 | 0.02344080 |
| Closed | Max drawdown (subset) | -22.4972% | -21.2629% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0830`
- baseline max drawdown (full): `-22.4972%`

## Attribution — What Drove Returns

**Biggest single market:** Will Hubert Hurkacz win the 2026 Men's US Open? (`tennis`) — 38.7% of total return
**Biggest domain:** `tennis` — 38.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Hubert Hurkacz win the 2026 Men's US Open? | `tennis` | 0.648849 | 38.7% | 0.0488 |
| 2 | Starmer out by April 30, 2026? | `grooming-gangs` | 0.577141 | 34.4% | 0.0489 |
| 3 | Will Wesley Hunt win the 2026 Texas Republican Primary? | `republican-primary` | 0.124235 | 7.4% | 0.0502 |
| 4 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.093485 | 5.6% | 0.0500 |
| 5 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.087921 | 5.2% | 0.0502 |
| 6 | Will Zachary Shrewsbury be the Democratic nominee for Senate in West Virginia? | `west-virginia-primary` | 0.084901 | 5.1% | 0.0502 |
| 7 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | 0.070438 | 4.2% | 0.0502 |
| 8 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.050558 | 3.0% | 0.0502 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041636 | -2.5% | 0.0501 |
| 10 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.040084 | 2.4% | 0.0502 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `tennis` | 0.648849 | 38.7% |
| 2 | `grooming-gangs` | 0.614195 | 36.7% |
| 3 | `republican-primary` | 0.124235 | 7.4% |
| 4 | `colombia-election` | 0.087921 | 5.2% |
| 5 | `west-virginia-primary` | 0.084901 | 5.1% |
| 6 | `nba-champion` | 0.083795 | 5.0% |
| 7 | `primaries` | 0.070438 | 4.2% |
| 8 | `airdrops` | 0.050558 | 3.0% |
| 9 | `ligue-1` | -0.041636 | -2.5% |
| 10 | `uefa-europa-league` | 0.040084 | 2.4% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will the Phoenix Suns win the NBA Western Conference Finals? | Will the Los Angeles Kings win the Western Conference? | 0.1397 |
| Will Zachary Shrewsbury be the Democratic nominee for Senate in West Virginia? | Will the Democrats win the South Dakota Senate race in 2026? | 0.1335 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.1233 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Los Angeles Kings win the Western Conference? | 0.0513 |
| Will the Democrats win the South Dakota Senate race in 2026? | Will Claude 5 be released by April 30, 2026? | 0.0138 |

## Correlation and Risk Structure
- category count: `18`
- avg abs category correlation: `0.0016`
- max abs category correlation: `0.0645`
- top eigenvalue share: `0.7314`
- variance ratio constrained vs baseline: `0.8605`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0339)
- [x] Constrained holdout drawdown better than baseline (+1.2343%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0645)
- [x] No single domain dominates returns (top domain share: 38.7%)
