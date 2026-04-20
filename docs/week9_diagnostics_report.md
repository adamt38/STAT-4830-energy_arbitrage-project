# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_E`
- constrained artifact stem: `week9_E_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8464`
- holdout steps: `2116`
- objective: `1.0`-var / `2.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0499 | 0.0280 | -0.0219 |
| Max drawdown | -9.7124% | -8.9104% | +0.8020% |
| Mean return | 0.00006825 | 0.00003857 | -0.00002967 |
| Volatility | 0.00357983 | 0.00319567 | -0.00038415 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.5%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0327 | -0.0136 |
| Open | Mean return | 0.00004805 | -0.00001761 |
| Open | Volatility | 0.00402228 | 0.00290437 |
| Open | Max drawdown (subset) | -5.5899% | -5.4472% |
| Closed | Sortino | 0.0534 | 0.0351 |
| Closed | Mean return | 0.00007196 | 0.00004892 |
| Closed | Volatility | 0.00349225 | 0.00324635 |
| Closed | Max drawdown (subset) | -6.5558% | -7.2237% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0565`
- baseline max drawdown (full): `-9.7124%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 86.8% of total return
**Biggest domain:** `felix` — 86.8% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.070849 | 86.8% | 0.0233 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.053924 | 66.1% | 0.0242 |
| 3 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.016686 | -20.4% | 0.0217 |
| 4 | Will the number of Republican Senate members who retire in 2026 be exactly 5? | `congress` | -0.014602 | -17.9% | 0.0222 |
| 5 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.013469 | 16.5% | 0.0236 |
| 6 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.011512 | -14.1% | 0.0262 |
| 7 | Will Toni Atkins win the California Governor Election in 2026? | `california-midterm` | -0.010405 | -12.7% | 0.0256 |
| 8 | Jeffrey Epstein confirmed to be alive before 2027? | `epstein` | -0.009097 | -11.1% | 0.0230 |
| 9 | Will Hyperliquid dip to $8 by December 31, 2026? | `crypto-prices` | -0.008676 | -10.6% | 0.0241 |
| 10 | MegaETH market cap (FDV) >$3B one day after launch? | `airdrops` | 0.007718 | 9.5% | 0.0255 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.070849 | 86.8% |
| 2 | `colombia-election` | 0.053924 | 66.1% |
| 3 | `economic-policy` | -0.016686 | -20.4% |
| 4 | `congress` | -0.014602 | -17.9% |
| 5 | `fifa-world-cup` | 0.013469 | 16.5% |
| 6 | `2026-fifa-world-cup` | -0.011512 | -14.1% |
| 7 | `california-midterm` | -0.010405 | -12.7% |
| 8 | `epstein` | -0.009097 | -11.1% |
| 9 | `crypto-prices` | -0.008676 | -10.6% |
| 10 | `airdrops` | 0.007718 | 9.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8669 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7555 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7498 |
| Will the number of Republican Senate members who retire in 2026 be exactly 5? | Will Claude 5 be released by April 30, 2026? | 0.0825 |
| Will the number of Republican Senate members who retire in 2026 be exactly 5? | Opensea FDV above $2B one day after launch? | -0.0197 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0026`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8496`
- variance ratio constrained vs baseline: `0.8963`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0219)
- [x] Constrained holdout drawdown better than baseline (+0.8020%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 86.8%)
