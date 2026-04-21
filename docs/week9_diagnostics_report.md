# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week12_Q`
- constrained artifact stem: `week12_Q_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6264`
- holdout steps: `1566`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0751 | 0.0808 | +0.0057 |
| Max drawdown | -35.0992% | -32.3445% | +2.7548% |
| Mean return | 0.00027988 | 0.00029933 | +0.00001945 |
| Volatility | 0.00921683 | 0.00922723 | +0.00001040 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `16.9%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0295 | 0.0351 |
| Open | Mean return | 0.00008447 | 0.00011493 |
| Open | Volatility | 0.00594989 | 0.00731658 |
| Open | Max drawdown (subset) | -4.6859% | -5.0436% |
| Closed | Sortino | 0.0824 | 0.0890 |
| Closed | Mean return | 0.00031969 | 0.00033689 |
| Closed | Volatility | 0.00974850 | 0.00956933 |
| Closed | Max drawdown (subset) | -32.8653% | -30.1273% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0356`
- baseline max drawdown (full): `-35.0992%`

## Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 42.7% of total return
**Biggest domain:** `best-of-2025` — 42.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.200155 | 42.7% | 0.0500 |
| 2 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.090422 | 19.3% | 0.0506 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.085220 | 18.2% | 0.0485 |
| 4 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.075996 | 16.2% | 0.0503 |
| 5 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.065505 | 14.0% | 0.0486 |
| 6 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.061985 | 13.2% | 0.0503 |
| 7 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.056711 | 12.1% | 0.0486 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.051520 | -11.0% | 0.0503 |
| 9 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044805 | -9.6% | 0.0502 |
| 10 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.043362 | 9.3% | 0.0506 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.200155 | 42.7% |
| 2 | `colombia-election` | 0.141931 | 30.3% |
| 3 | `serie-a` | 0.090422 | 19.3% |
| 4 | `nba-champion` | 0.075996 | 16.2% |
| 5 | `taylor-swift` | 0.061985 | 13.2% |
| 6 | `new-mexico-primary` | -0.051520 | -11.0% |
| 7 | `ufc` | -0.044805 | -9.6% |
| 8 | `airdrops` | 0.043362 | 9.3% |
| 9 | `primaries` | -0.042036 | -9.0% |
| 10 | `uefa-europa-league` | 0.040708 | 8.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0971 |
| Will Kash Patel leave the Trump administration before 2027? | SCOTUS accepts sports event contract case by December 31, 2026? | 0.0353 |
| Will Taylor Fritz win the 2026 Men's French Open? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0338 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0105 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | Will Kash Patel leave the Trump administration before 2027? | -0.0087 |

## Correlation and Risk Structure
- category count: `18`
- avg abs category correlation: `0.0012`
- max abs category correlation: `0.0473`
- top eigenvalue share: `0.3440`
- variance ratio constrained vs baseline: `0.9269`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0057)
- [x] Constrained holdout drawdown better than baseline (+2.7548%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0473)
- [x] No single domain dominates returns (top domain share: 42.7%)
