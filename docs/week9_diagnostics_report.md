# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_A`
- constrained artifact stem: `week9_A`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8382`
- holdout steps: `2096`
- objective: `1.1`-var / `2.3`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0387 | -0.0209 | -0.0597 |
| Max drawdown | -8.0707% | -13.6616% | -5.5910% |
| Mean return | 0.00005719 | -0.00002817 | -0.00008536 |
| Volatility | 0.00380155 | 0.00268330 | -0.00111825 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.1%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1183 | 0.0134 |
| Open | Mean return | 0.00018203 | 0.00001500 |
| Open | Volatility | 0.00516096 | 0.00383377 |
| Open | Max drawdown (subset) | -3.9132% | -4.8751% |
| Closed | Sortino | 0.0238 | -0.0260 |
| Closed | Mean return | 0.00003495 | -0.00003586 |
| Closed | Volatility | 0.00350392 | 0.00242150 |
| Closed | Max drawdown (subset) | -6.5584% | -12.8567% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0565`
- baseline max drawdown (full): `-9.2357%`

## Attribution — What Drove Returns

**Biggest single market:** Will Daniel Quintero win the 2026 Colombian presidential election? (`colombia-election`) — -84.1% of total return
**Biggest domain:** `colombia-election` — -84.1% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.049656 | -84.1% | 0.0282 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.042481 | 72.0% | 0.0137 |
| 3 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.039861 | -67.5% | 0.0180 |
| 4 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.026140 | -44.3% | 0.0233 |
| 5 | Will the number of Republican Senate members who retire in 2026 be exactly 5? | `congress` | -0.021611 | 36.6% | 0.0170 |
| 6 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | -0.019790 | 33.5% | 0.0168 |
| 7 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.016929 | 28.7% | 0.0164 |
| 8 | Epstein client list released by June 30? | `epstein` | -0.016512 | 28.0% | 0.0173 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.016127 | 27.3% | 0.0290 |
| 10 | Will Hyperliquid dip to $8 by December 31, 2026? | `crypto-prices` | -0.011973 | 20.3% | 0.0229 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.049656 | -84.1% |
| 2 | `claude-5` | -0.042481 | 72.0% |
| 3 | `felix` | 0.039861 | -67.5% |
| 4 | `fifa-world-cup` | 0.026140 | -44.3% |
| 5 | `congress` | -0.021611 | 36.6% |
| 6 | `best-of-2025` | -0.019790 | 33.5% |
| 7 | `economic-policy` | -0.016929 | 28.7% |
| 8 | `epstein` | -0.016512 | 28.0% |
| 9 | `2026-fifa-world-cup` | -0.016127 | 27.3% |
| 10 | `crypto-prices` | -0.011973 | 20.3% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7555 |
| Felix Protocol FDV above $300M one day after launch? | Will 2026 be the third-hottest year on record? | 0.1911 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will the number of Republican Senate members who retire in 2026 be exactly 5? | -0.1032 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.0981 |
| Will 2026 be the third-hottest year on record? | Opensea FDV above $2B one day after launch? | 0.0911 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0024`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8596`
- variance ratio constrained vs baseline: `0.4760`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0597)
- [ ] Constrained holdout drawdown better than baseline (-5.5910%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [x] No single domain dominates returns (top domain share: -84.1%)
