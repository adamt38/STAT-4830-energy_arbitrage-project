# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_F`
- constrained artifact stem: `week9_F`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8447`
- holdout steps: `2112`
- objective: `1.0`-var / `2.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0571 | 0.0513 | -0.0058 |
| Max drawdown | -9.4627% | -9.2857% | +0.1770% |
| Mean return | 0.00007693 | 0.00006739 | -0.00000954 |
| Volatility | 0.00357437 | 0.00343809 | -0.00013628 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0242 | 0.0057 |
| Open | Mean return | 0.00003608 | 0.00000824 |
| Open | Volatility | 0.00403716 | 0.00366953 |
| Open | Max drawdown (subset) | -6.2110% | -6.2359% |
| Closed | Sortino | 0.0641 | 0.0608 |
| Closed | Mean return | 0.00008452 | 0.00007838 |
| Closed | Volatility | 0.00348153 | 0.00339323 |
| Closed | Max drawdown (subset) | -6.6919% | -6.4899% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0637`
- baseline max drawdown (full): `-10.7215%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 54.9% of total return
**Biggest domain:** `felix` — 54.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.078141 | 54.9% | 0.0240 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.049766 | 35.0% | 0.0262 |
| 3 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.017799 | -12.5% | 0.0238 |
| 4 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.014324 | 10.1% | 0.0239 |
| 5 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.012098 | 8.5% | 0.0243 |
| 6 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.011834 | -8.3% | 0.0246 |
| 7 | Jeffrey Epstein confirmed to be alive before 2027? | `epstein` | -0.010236 | -7.2% | 0.0245 |
| 8 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.009526 | -6.7% | 0.0263 |
| 9 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.009510 | -6.7% | 0.0255 |
| 10 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.008468 | 5.9% | 0.0241 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.078141 | 54.9% |
| 2 | `colombia-election` | 0.049766 | 35.0% |
| 3 | `economic-policy` | -0.017799 | -12.5% |
| 4 | `best-of-2025` | 0.014324 | 10.1% |
| 5 | `comex-silver-futures` | 0.012098 | 8.5% |
| 6 | `congress` | -0.011834 | -8.3% |
| 7 | `epstein` | -0.010236 | -7.2% |
| 8 | `2026-fifa-world-cup` | -0.009526 | -6.7% |
| 9 | `california-midterm` | -0.009510 | -6.7% |
| 10 | `colorado-midterm` | 0.008468 | 5.9% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8670 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7796 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7737 |
| Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | Will Trump acquire Greenland before 2027? | 0.1760 |
| Will Trump acquire Greenland before 2027? | Extended FDV above $300M one day after launch? | 0.0941 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0028`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8384`
- variance ratio constrained vs baseline: `0.9241`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0058)
- [x] Constrained holdout drawdown better than baseline (+0.1770%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 54.9%)
