# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week11_L`
- constrained artifact stem: `week11_L_macro_both`
- min history days used after backoff: `24.0`
- market count: `25`
- tuning steps: `6724`
- holdout steps: `1681`
- objective: `1.7`-var / `2.9`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1123 | 0.0600 | -0.0523 |
| Max drawdown | -22.3789% | -24.8857% | -2.5068% |
| Mean return | 0.00038644 | 0.00019040 | -0.00019604 |
| Volatility | 0.01189919 | 0.00984945 | -0.00204974 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.3781 | 0.3633 |
| Open | Mean return | 0.00113242 | 0.00109855 |
| Open | Volatility | 0.01982562 | 0.01974648 |
| Open | Max drawdown (subset) | -10.6618% | -9.9417% |
| Closed | Sortino | 0.0640 | -0.0016 |
| Closed | Mean return | 0.00022570 | -0.00000528 |
| Closed | Volatility | 0.00934135 | 0.00580353 |
| Closed | Max drawdown (subset) | -21.4939% | -24.2960% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0847`
- baseline max drawdown (full): `-22.3789%`

## Attribution — What Drove Returns

**Biggest single market:** Starmer out by April 30, 2026? (`grooming-gangs`) — 121.6% of total return
**Biggest domain:** `grooming-gangs` — 121.6% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Starmer out by April 30, 2026? | `grooming-gangs` | 0.389308 | 121.6% | 0.0354 |
| 2 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.108790 | 34.0% | 0.0418 |
| 3 | Will Zachary Shrewsbury be the Democratic nominee for Senate in West Virginia? | `west-virginia-primary` | 0.083050 | 25.9% | 0.0417 |
| 4 | Will Jack Draper win the 2026 Men's French Open? | `roland-garros` | -0.056022 | -17.5% | 0.0370 |
| 5 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | -0.049808 | -15.6% | 0.0338 |
| 6 | Will there be fewer than 5 earthquakes of magnitude 7.0 or higher worldwide in 2026? | `natural-disasters` | -0.047372 | -14.8% | 0.0391 |
| 7 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.041385 | 12.9% | 0.0403 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.039208 | -12.3% | 0.0415 |
| 9 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.037557 | -11.7% | 0.0404 |
| 10 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.037468 | -11.7% | 0.0365 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `grooming-gangs` | 0.389308 | 121.6% |
| 2 | `league-of-legends` | 0.108790 | 34.0% |
| 3 | `roland-garros` | -0.083575 | -26.1% |
| 4 | `west-virginia-primary` | 0.083050 | 25.9% |
| 5 | `economic-policy` | -0.063605 | -19.9% |
| 6 | `basketball` | -0.049808 | -15.6% |
| 7 | `natural-disasters` | -0.047372 | -14.8% |
| 8 | `nba-champion` | 0.041385 | 12.9% |
| 9 | `new-mexico-primary` | -0.039208 | -12.3% |
| 10 | `primaries` | -0.037557 | -11.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | -0.2116 |
| Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | Taylor Swift pregnant before marriage? | -0.1236 |
| Will Connie Chan receive the most votes in the CA-11 primary? | USD.AI FDV above $300M one day after launch? | 0.0758 |
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Taylor Swift pregnant before marriage? | -0.0548 |
| Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | 0.0230 |

## Correlation and Risk Structure
- category count: `23`
- avg abs category correlation: `0.0010`
- max abs category correlation: `0.0804`
- top eigenvalue share: `0.7010`
- variance ratio constrained vs baseline: `0.7771`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0523)
- [ ] Constrained holdout drawdown better than baseline (-2.5068%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0804)
- [ ] No single domain dominates returns (top domain share: 121.6%)
