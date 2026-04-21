# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week11_K`
- constrained artifact stem: `week11_K_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6424`
- holdout steps: `1606`
- objective: `2.0`-var / `2.7`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0618 | 0.0615 | -0.0003 |
| Max drawdown | -29.1169% | -28.0285% | +1.0884% |
| Mean return | 0.00024762 | 0.00023892 | -0.00000871 |
| Volatility | 0.00821342 | 0.00817748 | -0.00003594 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.2%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1301 | 0.1817 |
| Open | Mean return | 0.00037162 | 0.00048149 |
| Open | Volatility | 0.00655537 | 0.00790220 |
| Open | Max drawdown (subset) | -5.6269% | -5.7454% |
| Closed | Sortino | 0.0528 | 0.0461 |
| Closed | Mean return | 0.00022189 | 0.00018858 |
| Closed | Volatility | 0.00851692 | 0.00823256 |
| Closed | Max drawdown (subset) | -27.8410% | -26.8757% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0579`
- baseline max drawdown (full): `-29.1169%`

## Attribution — What Drove Returns

**Biggest single market:** Will Ninjas in Pyjamas win the LPL 2026 season? (`league-of-legends`) — 36.2% of total return
**Biggest domain:** `colombia-election` — 40.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.139033 | 36.2% | 0.0500 |
| 2 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.088711 | 23.1% | 0.0501 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.088668 | 23.1% | 0.0498 |
| 4 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.076466 | 19.9% | 0.0500 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066740 | 17.4% | 0.0499 |
| 6 | Will Jack Draper win the 2026 Men's French Open? | `roland-garros` | -0.064986 | -16.9% | 0.0499 |
| 7 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.064338 | 16.8% | 0.0500 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.048544 | -12.7% | 0.0500 |
| 9 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044878 | -11.7% | 0.0500 |
| 10 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.044845 | -11.7% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.155408 | 40.5% |
| 2 | `league-of-legends` | 0.139033 | 36.2% |
| 3 | `roland-garros` | -0.103212 | -26.9% |
| 4 | `serie-a` | 0.088711 | 23.1% |
| 5 | `nba-champion` | 0.076466 | 19.9% |
| 6 | `taylor-swift` | 0.064338 | 16.8% |
| 7 | `new-mexico-primary` | -0.048544 | -12.7% |
| 8 | `ufc` | -0.044878 | -11.7% |
| 9 | `primaries` | -0.044845 | -11.7% |
| 10 | `uefa-europa-league` | 0.043142 | 11.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Jack Draper win the 2026 Men's French Open? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0926 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Starmer out by June 30, 2026? | -0.0098 |
| Will Petr Yan fight Pedro Munhoz next? | Will Connie Chan receive the most votes in the CA-11 primary? | 0.0059 |
| Will Connie Chan receive the most votes in the CA-11 primary? | Starmer out by June 30, 2026? | 0.0020 |
| Will AC Milan win the 2025–26 Serie A league? | Will Jack Draper win the 2026 Men's French Open? | 0.0018 |

## Correlation and Risk Structure
- category count: `18`
- avg abs category correlation: `0.0006`
- max abs category correlation: `0.0342`
- top eigenvalue share: `0.8864`
- variance ratio constrained vs baseline: `0.8180`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0003)
- [x] Constrained holdout drawdown better than baseline (+1.0884%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0342)
- [x] No single domain dominates returns (top domain share: 40.5%)
