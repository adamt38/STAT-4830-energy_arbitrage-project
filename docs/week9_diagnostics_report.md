# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_L`
- constrained artifact stem: `week9_L`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8042`
- holdout steps: `2011`
- objective: `1.3`-var / `2.1`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0394 | 0.0223 | -0.0172 |
| Max drawdown | -7.3094% | -9.9122% | -2.6028% |
| Mean return | 0.00004098 | 0.00003764 | -0.00000334 |
| Volatility | 0.00211065 | 0.00295713 | +0.00084648 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1159 | 0.0302 |
| Open | Mean return | 0.00013204 | 0.00006171 |
| Open | Volatility | 0.00314199 | 0.00325264 |
| Open | Max drawdown (subset) | -3.0654% | -6.4851% |
| Closed | Sortino | 0.0207 | 0.0202 |
| Closed | Mean return | 0.00002100 | 0.00003236 |
| Closed | Volatility | 0.00180649 | 0.00288819 |
| Closed | Max drawdown (subset) | -8.0442% | -12.3133% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0138`
- baseline max drawdown (full): `-7.3531%`

## Attribution — What Drove Returns

**Biggest single market:** Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? (`basketball`) — 64.6% of total return
**Biggest domain:** `basketball` — 64.6% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.048879 | 64.6% | 0.0464 |
| 2 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.043926 | -58.0% | 0.0512 |
| 3 | Will Victor Wembanyama win the 2025–2026 NBA MVP? | `awards` | 0.038573 | 51.0% | 0.0513 |
| 4 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.033604 | 44.4% | 0.0377 |
| 5 | Jeffrey Epstein confirmed to be alive before 2027? | `epstein` | -0.014915 | -19.7% | 0.0513 |
| 6 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.014042 | 18.6% | 0.0108 |
| 7 | Will Bitcoin outperform Gold in 2026? | `bitcoin` | 0.010267 | 13.6% | 0.0376 |
| 8 | Will Claude 5 be released by May 31, 2026? | `claude-5` | -0.010017 | -13.2% | 0.0362 |
| 9 | Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | `congress` | 0.009457 | 12.5% | 0.0514 |
| 10 | Will China invade Taiwan by end of 2026? | `foreign-policy` | -0.006570 | -8.7% | 0.0513 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `basketball` | 0.048879 | 64.6% |
| 2 | `economic-policy` | -0.043926 | -58.0% |
| 3 | `awards` | 0.038573 | 51.0% |
| 4 | `airdrops` | 0.033604 | 44.4% |
| 5 | `epstein` | -0.014915 | -19.7% |
| 6 | `colombia-election` | 0.014042 | 18.6% |
| 7 | `bitcoin` | 0.010267 | 13.6% |
| 8 | `claude-5` | -0.010017 | -13.2% |
| 9 | `congress` | 0.009457 | 12.5% |
| 10 | `foreign-policy` | -0.006570 | -8.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | -0.2492 |
| Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | Will Somaliland join the Abraham Accords before 2027? | 0.1430 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will Claude 5 be released by May 31, 2026? | -0.0628 |
| Will Somaliland join the Abraham Accords before 2027? | Will Bitcoin have the best performance in 2026? | 0.0297 |
| Will China invade Taiwan by end of 2026? | Will Somaliland join the Abraham Accords before 2027? | 0.0111 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0013`
- max abs category correlation: `0.0740`
- top eigenvalue share: `0.1507`
- variance ratio constrained vs baseline: `2.4786`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0172)
- [ ] Constrained holdout drawdown better than baseline (-2.6028%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0740)
- [ ] No single domain dominates returns (top domain share: 64.6%)
