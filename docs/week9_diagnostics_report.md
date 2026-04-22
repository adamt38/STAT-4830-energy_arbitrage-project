# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week13_A_seed7`
- constrained artifact stem: `week13_A_seed7_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5543`
- holdout steps: `1386`
- objective: `1.8`-var / `2.3`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1914 | 0.1749 | -0.0166 |
| Max drawdown | -20.3348% | -20.8947% | -0.5599% |
| Mean return | 0.00084384 | 0.00073885 | -0.00010500 |
| Volatility | 0.02369963 | 0.02048609 | -0.00321354 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.1%`
- Holdout steps marked exog-stale: `0.3%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | -0.1177 | -0.1074 |
| Open | Mean return | -0.00044583 | -0.00041521 |
| Open | Volatility | 0.00523988 | 0.00500733 |
| Open | Max drawdown (subset) | -12.1990% | -12.0429% |
| Closed | Sortino | 0.2491 | 0.2311 |
| Closed | Mean return | 0.00112905 | 0.00099406 |
| Closed | Volatility | 0.02606459 | 0.02250746 |
| Closed | Max drawdown (subset) | -20.3348% | -20.8947% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0582`
- baseline max drawdown (full): `-20.3348%`

## Attribution — What Drove Returns

**Biggest single market:** Will Hubert Hurkacz win the 2026 Men's US Open? (`tennis`) — 60.0% of total return
**Biggest domain:** `tennis` — 60.0% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Hubert Hurkacz win the 2026 Men's US Open? | `tennis` | 0.614438 | 60.0% | 0.0492 |
| 2 | Will the Orlando Magic win the 2026 NBA Finals? | `nba-champion` | 0.186399 | 18.2% | 0.0500 |
| 3 | Will Jamal Murray win the 2025–2026 NBA Clutch Player of the Year? | `basketball` | 0.148768 | 14.5% | 0.0499 |
| 4 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.070419 | 6.9% | 0.0501 |
| 5 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.066594 | 6.5% | 0.0500 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.049058 | 4.8% | 0.0501 |
| 7 | Will Real Madrid win the 2025–26 La Liga? | `la-liga` | 0.047986 | 4.7% | 0.0500 |
| 8 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041517 | -4.1% | 0.0500 |
| 9 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.039259 | -3.8% | 0.0501 |
| 10 | Will Taylor Fritz win the 2026 Men's French Open? | `roland-garros` | -0.038290 | -3.7% | 0.0501 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `tennis` | 0.614438 | 60.0% |
| 2 | `nba-champion` | 0.186399 | 18.2% |
| 3 | `basketball` | 0.148768 | 14.5% |
| 4 | `serie-a` | 0.070419 | 6.9% |
| 5 | `airdrops` | 0.049058 | 4.8% |
| 6 | `new-mexico-primary` | -0.039259 | -3.8% |
| 7 | `roland-garros` | -0.038290 | -3.7% |
| 8 | `primaries` | -0.036088 | -3.5% |
| 9 | `comex-silver-futures` | 0.035469 | 3.5% |
| 10 | `league-of-legends` | -0.033277 | -3.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will BRION win the LCK 2026 season playoffs? | Will the Los Angeles Kings win the Western Conference? | 0.2546 |
| Will the Los Angeles Kings win the Western Conference? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.1534 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0995 |
| Will Rennes win the 2025–26 French Ligue 1? | Will BRION win the LCK 2026 season playoffs? | 0.0970 |
| Will the Orlando Magic win the 2026 NBA Finals? | Will Silver (SI) hit (HIGH) $200 by end of June? | 0.0674 |

## Correlation and Risk Structure
- category count: `17`
- avg abs category correlation: `0.0029`
- max abs category correlation: `0.0490`
- top eigenvalue share: `0.5579`
- variance ratio constrained vs baseline: `0.8381`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0166)
- [ ] Constrained holdout drawdown better than baseline (-0.5599%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0490)
- [ ] No single domain dominates returns (top domain share: 60.0%)
