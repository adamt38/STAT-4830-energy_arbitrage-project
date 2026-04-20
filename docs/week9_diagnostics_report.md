# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_D`
- constrained artifact stem: `week9_D_macro_explicit`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8628`
- holdout steps: `2157`
- objective: `1.9`-var / `2.4`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0962 | 0.0555 | -0.0407 |
| Max drawdown | -8.7653% | -13.5571% | -4.7919% |
| Mean return | 0.00013075 | 0.00007085 | -0.00005990 |
| Volatility | 0.00429574 | 0.00384035 | -0.00045538 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0589 | -0.1094 |
| Open | Mean return | 0.00008886 | -0.00013405 |
| Open | Volatility | 0.00410650 | 0.00143417 |
| Open | Max drawdown (subset) | -4.7309% | -4.8177% |
| Closed | Sortino | 0.1037 | 0.0831 |
| Closed | Mean return | 0.00013813 | 0.00010694 |
| Closed | Volatility | 0.00432816 | 0.00412005 |
| Closed | Max drawdown (subset) | -7.1228% | -9.9985% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0422`
- baseline max drawdown (full): `-8.8591%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 74.7% of total return
**Biggest domain:** `felix` — 74.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.114095 | 74.7% | 0.0166 |
| 2 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.075298 | 49.3% | 0.0226 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.068018 | 44.5% | 0.0235 |
| 4 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.047632 | -31.2% | 0.0139 |
| 5 | Epstein client list released by June 30? | `epstein` | -0.029359 | -19.2% | 0.0191 |
| 6 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.025013 | 16.4% | 0.0230 |
| 7 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.020254 | -13.3% | 0.0167 |
| 8 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.017066 | -11.2% | 0.0143 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.014146 | -9.3% | 0.0247 |
| 10 | Will Toni Atkins win the California Governor Election in 2026? | `california-midterm` | -0.012418 | -8.1% | 0.0236 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.114095 | 74.7% |
| 2 | `best-of-2025` | 0.075298 | 49.3% |
| 3 | `colombia-election` | 0.068018 | 44.5% |
| 4 | `claude-5` | -0.047632 | -31.2% |
| 5 | `epstein` | -0.029359 | -19.2% |
| 6 | `fifa-world-cup` | 0.025013 | 16.4% |
| 7 | `congress` | -0.020254 | -13.3% |
| 8 | `economic-policy` | -0.017066 | -11.2% |
| 9 | `2026-fifa-world-cup` | -0.014146 | -9.3% |
| 10 | `california-midterm` | -0.012418 | -8.1% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7796 |
| Opensea FDV above $2B one day after launch? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.3181 |
| Felix Protocol FDV above $300M one day after launch? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.2884 |
| Will Claude 5 be released by April 30, 2026? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | 0.0301 |
| Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | MegaETH market cap (FDV) >$3B one day after launch? | 0.0186 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.4510`
- variance ratio constrained vs baseline: `0.6589`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0407)
- [ ] Constrained holdout drawdown better than baseline (-4.7919%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 74.7%)
