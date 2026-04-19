# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8_F`
- constrained artifact stem: `week8_F`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8635`
- holdout steps: `2159`
- objective: `0.8`-var / `2.1`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0534 | 0.0332 | -0.0202 |
| Max drawdown | -7.7391% | -8.4621% | -0.7230% |
| Mean return | 0.00006838 | 0.00004123 | -0.00002715 |
| Volatility | 0.00349438 | 0.00318700 | -0.00030738 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1667 | 0.0790 |
| Open | Mean return | 0.00023384 | 0.00008982 |
| Open | Volatility | 0.00509108 | 0.00349463 |
| Open | Max drawdown (subset) | -3.3679% | -2.4729% |
| Closed | Sortino | 0.0306 | 0.0258 |
| Closed | Mean return | 0.00003842 | 0.00003243 |
| Closed | Volatility | 0.00311811 | 0.00312798 |
| Closed | Max drawdown (subset) | -7.8938% | -8.4922% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0643`
- baseline max drawdown (full): `-8.7121%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 85.8% of total return
**Biggest domain:** `felix` — 85.8% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.076410 | 85.8% | 0.0263 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.039632 | -44.5% | 0.0256 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033359 | 37.5% | 0.0252 |
| 4 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.027701 | 31.1% | 0.0255 |
| 5 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.026502 | -29.8% | 0.0247 |
| 6 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.017270 | 19.4% | 0.0252 |
| 7 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.013550 | 15.2% | 0.0249 |
| 8 | Will J.D. Vance announce a presidential run before 2027? | `celebrities` | -0.010918 | -12.3% | 0.0247 |
| 9 | MegaETH market cap (FDV) >$3B one day after launch? | `airdrops` | 0.010722 | 12.0% | 0.0253 |
| 10 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.010008 | -11.2% | 0.0247 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.076410 | 85.8% |
| 2 | `claude-5` | -0.039632 | -44.5% |
| 3 | `colombia-election` | 0.033359 | 37.5% |
| 4 | `best-of-2025` | 0.027701 | 31.1% |
| 5 | `economic-policy` | -0.026502 | -29.8% |
| 6 | `colorado-midterm` | 0.017270 | 19.4% |
| 7 | `fifa-world-cup` | 0.013550 | 15.2% |
| 8 | `celebrities` | -0.010918 | -12.3% |
| 9 | `airdrops` | 0.010722 | 12.0% |
| 10 | `2026-fifa-world-cup` | -0.010008 | -11.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8594 |
| Will the Republicans win the Colorado Senate race in 2026? | Will SpaceX not IPO by December 31, 2027? | -0.1852 |
| Felix Protocol FDV above $300M one day after launch? | Will SpaceX not IPO by December 31, 2027? | -0.1602 |
| Will Claude 5 be released by April 30, 2026? | Will Daniel Quintero win the 2026 Colombian presidential election? | 0.1003 |
| Will J.D. Vance announce a presidential run before 2027? | Will SpaceX not IPO by December 31, 2027? | -0.0505 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0027`
- max abs category correlation: `0.2661`
- top eigenvalue share: `0.8578`
- variance ratio constrained vs baseline: `1.0483`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0202)
- [ ] Constrained holdout drawdown better than baseline (-0.7230%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2661)
- [ ] No single domain dominates returns (top domain share: 85.8%)
