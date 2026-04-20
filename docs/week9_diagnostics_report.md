# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_B`
- constrained artifact stem: `week9_B`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `7691`
- holdout steps: `1923`
- objective: `1.0`-var / `2.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0997 | 0.0997 | +0.0000 |
| Max drawdown | -6.0264% | -6.0264% | +0.0000% |
| Mean return | 0.00014660 | 0.00014660 | -0.00000000 |
| Volatility | 0.00465710 | 0.00465710 | -0.00000000 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `14.5%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.2066 | 0.2066 |
| Open | Mean return | 0.00032272 | 0.00032272 |
| Open | Volatility | 0.00554201 | 0.00554201 |
| Open | Max drawdown (subset) | -3.7338% | -3.7338% |
| Closed | Sortino | 0.0803 | 0.0803 |
| Closed | Mean return | 0.00011683 | 0.00011683 |
| Closed | Volatility | 0.00448968 | 0.00448968 |
| Closed | Max drawdown (subset) | -6.2968% | -6.2968% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0430`
- baseline max drawdown (full): `-9.5967%`

## Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 47.1% of total return
**Biggest domain:** `best-of-2025` — 47.1% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.132736 | 47.1% | 0.0250 |
| 2 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.084806 | 30.1% | 0.0250 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033333 | 11.8% | 0.0250 |
| 4 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.023806 | 8.4% | 0.0250 |
| 5 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.018366 | -6.5% | 0.0250 |
| 6 | Will Claude 5 be released by April 30, 2026? | `claude-5` | 0.017920 | 6.4% | 0.0250 |
| 7 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.015015 | 5.3% | 0.0250 |
| 8 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.010173 | 3.6% | 0.0250 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.010000 | -3.5% | 0.0250 |
| 10 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.010000 | -3.5% | 0.0250 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.132736 | 47.1% |
| 2 | `felix` | 0.084806 | 30.1% |
| 3 | `colombia-election` | 0.033333 | 11.8% |
| 4 | `airdrops` | 0.023806 | 8.4% |
| 5 | `economic-policy` | -0.018366 | -6.5% |
| 6 | `claude-5` | 0.017920 | 6.4% |
| 7 | `colorado-midterm` | 0.015015 | 5.3% |
| 8 | `comex-silver-futures` | 0.010173 | 3.6% |
| 9 | `2026-fifa-world-cup` | -0.010000 | -3.5% |
| 10 | `california-midterm` | -0.010000 | -3.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8457 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7529 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7341 |
| Felix Protocol FDV above $300M one day after launch? | Will 2026 be the third-hottest year on record? | 0.1477 |
| Will the Republicans win the Colorado Senate race in 2026? | Will 2026 be the third-hottest year on record? | 0.1322 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0034`
- max abs category correlation: `0.2653`
- top eigenvalue share: `0.4675`
- variance ratio constrained vs baseline: `1.0000`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (+0.0000)
- [ ] Constrained holdout drawdown better than baseline (+0.0000%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2653)
- [x] No single domain dominates returns (top domain share: 47.1%)
