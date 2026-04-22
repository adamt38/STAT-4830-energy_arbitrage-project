# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_G_seed123`
- constrained artifact stem: `week9_G_seed123`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5108`
- holdout steps: `1277`
- objective: `1.3`-var / `2.1`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1706 | 0.1623 | -0.0083 |
| Max drawdown | -17.3202% | -16.5094% | +0.8108% |
| Mean return | 0.00074131 | 0.00069449 | -0.00004682 |
| Volatility | 0.02359566 | 0.02106170 | -0.00253396 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `19.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | -0.0275 | -0.0093 |
| Open | Mean return | -0.00011383 | -0.00003945 |
| Open | Volatility | 0.00649590 | 0.00734656 |
| Open | Max drawdown (subset) | -13.2507% | -13.4809% |
| Closed | Sortino | 0.2146 | 0.2024 |
| Closed | Mean return | 0.00094228 | 0.00086697 |
| Closed | Volatility | 0.02602824 | 0.02313016 |
| Closed | Max drawdown (subset) | -16.2241% | -13.8960% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0664`
- baseline max drawdown (full): `-17.3202%`

## Attribution — What Drove Returns

**Biggest single market:** Will Hubert Hurkacz win the 2026 Men's US Open? (`tennis`) — 72.8% of total return
**Biggest domain:** `tennis` — 72.8% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Hubert Hurkacz win the 2026 Men's US Open? | `tennis` | 0.646075 | 72.8% | 0.0486 |
| 2 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.092447 | 10.4% | 0.0453 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.087694 | 9.9% | 0.0504 |
| 4 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.079978 | 9.0% | 0.0515 |
| 5 | Will the Pittsburgh Penguins win the Eastern Conference? | `hockey` | -0.067654 | -7.6% | 0.0514 |
| 6 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.065085 | 7.3% | 0.0473 |
| 7 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.047804 | 5.4% | 0.0519 |
| 8 | Will Taylor Fritz win the 2026 Men's French Open? | `roland-garros` | -0.038133 | -4.3% | 0.0510 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.037404 | -4.2% | 0.0482 |
| 10 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.037270 | -4.2% | 0.0516 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `tennis` | 0.646075 | 72.8% |
| 2 | `nba-champion` | 0.123697 | 13.9% |
| 3 | `colombia-election` | 0.087694 | 9.9% |
| 4 | `serie-a` | 0.079978 | 9.0% |
| 5 | `hockey` | -0.067654 | -7.6% |
| 6 | `airdrops` | 0.047804 | 5.4% |
| 7 | `roland-garros` | -0.038133 | -4.3% |
| 8 | `primaries` | -0.037270 | -4.2% |
| 9 | `league-of-legends` | -0.034465 | -3.9% |
| 10 | `new-mexico-primary` | 0.032907 | 3.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3903 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1638 |
| Will the Pittsburgh Penguins win the Eastern Conference? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.1616 |
| Will the Pittsburgh Penguins win the Eastern Conference? | MegaETH market cap (FDV) >$2B one day after launch? | 0.1317 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |

## Correlation and Risk Structure
- category count: `17`
- avg abs category correlation: `0.0035`
- max abs category correlation: `0.0855`
- top eigenvalue share: `0.5100`
- variance ratio constrained vs baseline: `0.7767`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0083)
- [x] Constrained holdout drawdown better than baseline (+0.8108%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0855)
- [ ] No single domain dominates returns (top domain share: 72.8%)
