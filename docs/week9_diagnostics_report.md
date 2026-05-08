# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week11_I`
- constrained artifact stem: `week11_I_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6197`
- holdout steps: `1550`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0963 | 0.1040 | +0.0077 |
| Max drawdown | -29.6971% | -28.5396% | +1.1575% |
| Mean return | 0.00033985 | 0.00036985 | +0.00003000 |
| Volatility | 0.00921816 | 0.00940606 | +0.00018790 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.2606 | 0.3216 |
| Open | Mean return | 0.00054150 | 0.00067592 |
| Open | Volatility | 0.00609932 | 0.00788247 |
| Open | Max drawdown (subset) | -3.4018% | -3.2368% |
| Closed | Sortino | 0.0794 | 0.0811 |
| Closed | Mean return | 0.00029845 | 0.00030702 |
| Closed | Volatility | 0.00973507 | 0.00968804 |
| Closed | Max drawdown (subset) | -29.4053% | -28.2444% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0369`
- baseline max drawdown (full): `-29.6971%`

## Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 34.7% of total return
**Biggest domain:** `best-of-2025` — 34.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.198840 | 34.7% | 0.0498 |
| 2 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.138570 | 24.2% | 0.0500 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.092983 | 16.2% | 0.0501 |
| 4 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.086616 | 15.1% | 0.0499 |
| 5 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.075581 | 13.2% | 0.0500 |
| 6 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066231 | 11.6% | 0.0499 |
| 7 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.062327 | 10.9% | 0.0500 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.048438 | -8.4% | 0.0500 |
| 9 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.047060 | -8.2% | 0.0500 |
| 10 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044579 | -7.8% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.198840 | 34.7% |
| 2 | `colombia-election` | 0.152847 | 26.7% |
| 3 | `league-of-legends` | 0.138570 | 24.2% |
| 4 | `serie-a` | 0.092983 | 16.2% |
| 5 | `nba-champion` | 0.075581 | 13.2% |
| 6 | `taylor-swift` | 0.062327 | 10.9% |
| 7 | `new-mexico-primary` | -0.048438 | -8.4% |
| 8 | `primaries` | -0.047060 | -8.2% |
| 9 | `ufc` | -0.044579 | -7.8% |
| 10 | `ligue-1` | -0.041397 | -7.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0974 |
| Will Taylor Fritz win the 2026 Men's French Open? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0338 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0105 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Starmer out by June 30, 2026? | -0.0098 |
| Will Connie Chan receive the most votes in the CA-11 primary? | Will Petr Yan fight Pedro Munhoz next? | 0.0062 |

## Correlation and Risk Structure
- category count: `19`
- avg abs category correlation: `0.0011`
- max abs category correlation: `0.0474`
- top eigenvalue share: `0.3273`
- variance ratio constrained vs baseline: `0.9421`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0077)
- [x] Constrained holdout drawdown better than baseline (+1.1575%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0474)
- [x] No single domain dominates returns (top domain share: 34.7%)
