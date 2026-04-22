# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week13_A_seed42`
- constrained artifact stem: `week13_A_seed42_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5543`
- holdout steps: `1386`
- objective: `1.2`-var / `2.9`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1914 | 0.1742 | -0.0173 |
| Max drawdown | -20.3348% | -20.8654% | -0.5307% |
| Mean return | 0.00084384 | 0.00073606 | -0.00010779 |
| Volatility | 0.02369963 | 0.02036019 | -0.00333944 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.1%`
- Holdout steps marked exog-stale: `0.3%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | -0.1177 | -0.1077 |
| Open | Mean return | -0.00044583 | -0.00041692 |
| Open | Volatility | 0.00523988 | 0.00500056 |
| Open | Max drawdown (subset) | -12.1990% | -12.0794% |
| Closed | Sortino | 0.2491 | 0.2305 |
| Closed | Mean return | 0.00112905 | 0.00099103 |
| Closed | Volatility | 0.02606459 | 0.02236787 |
| Closed | Max drawdown (subset) | -20.3348% | -20.8654% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0582`
- baseline max drawdown (full): `-20.3348%`

## Attribution — What Drove Returns

**Biggest single market:** Will Hubert Hurkacz win the 2026 Men's US Open? (`tennis`) — 59.7% of total return
**Biggest domain:** `tennis` — 59.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Hubert Hurkacz win the 2026 Men's US Open? | `tennis` | 0.608825 | 59.7% | 0.0491 |
| 2 | Will the Orlando Magic win the 2026 NBA Finals? | `nba-champion` | 0.187576 | 18.4% | 0.0501 |
| 3 | Will Jamal Murray win the 2025–2026 NBA Clutch Player of the Year? | `basketball` | 0.148873 | 14.6% | 0.0500 |
| 4 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.069980 | 6.9% | 0.0501 |
| 5 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.066526 | 6.5% | 0.0499 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.049672 | 4.9% | 0.0501 |
| 7 | Will Real Madrid win the 2025–26 La Liga? | `la-liga` | 0.047931 | 4.7% | 0.0500 |
| 8 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041499 | -4.1% | 0.0499 |
| 9 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.039255 | -3.8% | 0.0501 |
| 10 | Will Taylor Fritz win the 2026 Men's French Open? | `roland-garros` | -0.038293 | -3.8% | 0.0501 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `tennis` | 0.608825 | 59.7% |
| 2 | `nba-champion` | 0.187576 | 18.4% |
| 3 | `basketball` | 0.148873 | 14.6% |
| 4 | `serie-a` | 0.069980 | 6.9% |
| 5 | `airdrops` | 0.049672 | 4.9% |
| 6 | `new-mexico-primary` | -0.039255 | -3.8% |
| 7 | `roland-garros` | -0.038293 | -3.8% |
| 8 | `primaries` | -0.036131 | -3.5% |
| 9 | `comex-silver-futures` | 0.035489 | 3.5% |
| 10 | `league-of-legends` | -0.033277 | -3.3% |

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
- [ ] Constrained holdout Sortino beats baseline (-0.0173)
- [ ] Constrained holdout drawdown better than baseline (-0.5307%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0490)
- [ ] No single domain dominates returns (top domain share: 59.7%)
