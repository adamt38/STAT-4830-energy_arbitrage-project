# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_C`
- constrained artifact stem: `week9_C_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8619`
- holdout steps: `2155`
- objective: `1.9`-var / `2.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1240 | 0.1280 | +0.0040 |
| Max drawdown | -8.1956% | -8.3810% | -0.1854% |
| Mean return | 0.00019803 | 0.00016291 | -0.00003511 |
| Volatility | 0.00507946 | 0.00528384 | +0.00020438 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `14.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0667 | 0.1759 |
| Open | Mean return | 0.00014238 | 0.00015724 |
| Open | Volatility | 0.00548862 | 0.00392487 |
| Open | Max drawdown (subset) | -6.8498% | -2.6505% |
| Closed | Sortino | 0.1398 | 0.1235 |
| Closed | Mean return | 0.00020759 | 0.00016389 |
| Closed | Volatility | 0.00500573 | 0.00548356 |
| Closed | Max drawdown (subset) | -7.2057% | -6.7245% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0427`
- baseline max drawdown (full): `-8.1956%`

## Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 48.3% of total return
**Biggest domain:** `best-of-2025` — 48.3% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.169407 | 48.3% | 0.0214 |
| 2 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.125484 | 35.7% | 0.0170 |
| 3 | Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.078185 | 22.3% | 0.0176 |
| 4 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.049883 | 14.2% | 0.0308 |
| 5 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.047372 | -13.5% | 0.0125 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.027973 | 8.0% | 0.0265 |
| 7 | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | `economic-policy` | -0.018186 | -5.2% | 0.0205 |
| 8 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.012663 | -3.6% | 0.0260 |
| 9 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.009700 | 2.8% | 0.0160 |
| 10 | Will Bernie endorse Antonio Delgado for NY-Gov by Nov 2 2026 ET? | `bernie-sanders` | -0.009571 | -2.7% | 0.0231 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.169407 | 48.3% |
| 2 | `felix` | 0.125484 | 35.7% |
| 3 | `basketball` | 0.078185 | 22.3% |
| 4 | `colombia-election` | 0.049883 | 14.2% |
| 5 | `claude-5` | -0.047372 | -13.5% |
| 6 | `airdrops` | 0.027973 | 8.0% |
| 7 | `economic-policy` | -0.018186 | -5.2% |
| 8 | `california-midterm` | -0.012663 | -3.6% |
| 9 | `comex-silver-futures` | 0.009700 | 2.8% |
| 10 | `bernie-sanders` | -0.009571 | -2.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8669 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.0981 |
| Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? | Will Bitcoin outperform Gold in 2026? | 0.0320 |
| Will Silver (SI) hit (HIGH) $250 by end of June? | Will Bernie endorse Antonio Delgado for NY-Gov by Nov 2 2026 ET? | 0.0132 |
| Will Claude 5 be released by April 30, 2026? | Will Gold (GC) hit (HIGH) $6,500 by end of June? | 0.0103 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0029`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.4246`
- variance ratio constrained vs baseline: `0.6031`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0040)
- [ ] Constrained holdout drawdown better than baseline (-0.1854%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [x] No single domain dominates returns (top domain share: 48.3%)
