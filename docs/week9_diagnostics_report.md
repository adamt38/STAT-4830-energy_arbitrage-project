# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_L_smoke`
- constrained artifact stem: `week9_L_smoke`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `7697`
- holdout steps: `1925`
- objective: `2.0`-var / `2.6`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0238 | -0.0102 | -0.0340 |
| Max drawdown | -7.4737% | -7.7247% | -0.2510% |
| Mean return | 0.00002935 | -0.00001498 | -0.00004433 |
| Volatility | 0.00238599 | 0.00252013 | +0.00013415 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `16.4%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0576 | 0.0030 |
| Open | Mean return | 0.00007709 | 0.00000582 |
| Open | Volatility | 0.00339620 | 0.00349960 |
| Open | Max drawdown (subset) | -3.9059% | -5.9680% |
| Closed | Sortino | 0.0165 | -0.0141 |
| Closed | Mean return | 0.00002001 | -0.00001905 |
| Closed | Volatility | 0.00213297 | 0.00227978 |
| Closed | Max drawdown (subset) | -7.9658% | -9.1910% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0402`
- baseline max drawdown (full): `-8.1013%`

## Attribution — What Drove Returns

**Biggest single market:** Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? (`economic-policy`) — 156.9% of total return
**Biggest domain:** `economic-policy` — 156.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.045238 | 156.9% | 0.0350 |
| 2 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.036827 | -127.7% | 0.0188 |
| 3 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.017016 | -59.0% | 0.0320 |
| 4 | Will Saudi Aramco be the largest company in the world by market cap on December 31? | `business` | -0.012581 | 43.6% | 0.0396 |
| 5 | Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | `congress` | 0.011205 | -38.9% | 0.0410 |
| 6 | Will Trump acquire Greenland before 2027? | `foreign-policy` | -0.009165 | 31.8% | 0.0247 |
| 7 | Will Bernie endorse James Talarico for TX-Sen by Nov 2 2026 ET? | `bernie-sanders` | -0.008789 | 30.5% | 0.0246 |
| 8 | Will Bernie Sanders announce a Presidential run before 2027? | `celebrities` | -0.008516 | 29.5% | 0.0290 |
| 9 | Will Somaliland join the Abraham Accords before 2027? | `abraham-accords` | 0.007212 | -25.0% | 0.0445 |
| 10 | Will Charlotte rank #1 among girl names on the SSA’s official list for 2025? | `best-of-2025` | -0.007078 | 24.6% | 0.0044 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `economic-policy` | -0.045238 | 156.9% |
| 2 | `colombia-election` | 0.036827 | -127.7% |
| 3 | `basketball` | 0.017016 | -59.0% |
| 4 | `business` | -0.012581 | 43.6% |
| 5 | `congress` | 0.011205 | -38.9% |
| 6 | `foreign-policy` | -0.009165 | 31.8% |
| 7 | `bernie-sanders` | -0.008789 | 30.5% |
| 8 | `celebrities` | -0.008516 | 29.5% |
| 9 | `abraham-accords` | 0.007212 | -25.0% |
| 10 | `best-of-2025` | -0.007078 | 24.6% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | -0.2240 |
| Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | Will Somaliland join the Abraham Accords before 2027? | 0.1430 |
| Will Trump acquire Greenland before 2027? | Will Bernie endorse James Talarico for TX-Sen by Nov 2 2026 ET? | 0.0551 |
| Will Trump acquire Greenland before 2027? | Extended FDV above $300M one day after launch? | 0.0548 |
| Will Janet Mills be the Democratic nominee for Senate in Maine? | Will the Republicans win the Colorado Senate race in 2026? | -0.0171 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0015`
- max abs category correlation: `0.1035`
- top eigenvalue share: `0.8749`
- variance ratio constrained vs baseline: `0.1752`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0340)
- [ ] Constrained holdout drawdown better than baseline (-0.2510%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1035)
- [ ] No single domain dominates returns (top domain share: 156.9%)
