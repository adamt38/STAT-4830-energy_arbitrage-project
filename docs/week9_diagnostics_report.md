# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8_D`
- constrained artifact stem: `week8_D_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8678`
- holdout steps: `2170`
- objective: `1.6`-var / `1.8`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0464 | 0.0258 | -0.0206 |
| Max drawdown | -7.9225% | -8.1648% | -0.2423% |
| Mean return | 0.00006175 | 0.00004021 | -0.00002154 |
| Volatility | 0.00349382 | 0.00337768 | -0.00011613 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0394 | -0.0974 |
| Open | Mean return | 0.00005626 | -0.00014508 |
| Open | Volatility | 0.00399671 | 0.00160941 |
| Open | Max drawdown (subset) | -4.9693% | -5.5903% |
| Closed | Sortino | 0.0478 | 0.0469 |
| Closed | Mean return | 0.00006275 | 0.00007368 |
| Closed | Volatility | 0.00339504 | 0.00360477 |
| Closed | Max drawdown (subset) | -7.8388% | -5.8697% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0629`
- baseline max drawdown (full): `-9.4604%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 101.0% of total return
**Biggest domain:** `felix` — 101.0% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.088156 | 101.0% | 0.0252 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.046818 | -53.7% | 0.0250 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033129 | 38.0% | 0.0248 |
| 4 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.027925 | -32.0% | 0.0247 |
| 5 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.020669 | -23.7% | 0.0248 |
| 6 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.019961 | 22.9% | 0.0250 |
| 7 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.017521 | 20.1% | 0.0245 |
| 8 | Will Saudi Aramco be the largest company in the world by market cap on June 30? | `big-tech` | 0.010750 | 12.3% | 0.0252 |
| 9 | MegaETH market cap (FDV) >$3B one day after launch? | `airdrops` | 0.010714 | 12.3% | 0.0254 |
| 10 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.009995 | -11.5% | 0.0246 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.088156 | 101.0% |
| 2 | `claude-5` | -0.046818 | -53.7% |
| 3 | `colombia-election` | 0.033129 | 38.0% |
| 4 | `economic-policy` | -0.027925 | -32.0% |
| 5 | `congress` | -0.020669 | -23.7% |
| 6 | `colorado-midterm` | 0.019961 | 22.9% |
| 7 | `fifa-world-cup` | 0.017521 | 20.1% |
| 8 | `big-tech` | 0.010750 | 12.3% |
| 9 | `airdrops` | 0.010714 | 12.3% |
| 10 | `2026-fifa-world-cup` | -0.009995 | -11.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8594 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7505 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7484 |
| Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | Opensea FDV above $2B one day after launch? | -0.3013 |
| Will the Republicans win the Colorado Senate race in 2026? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.2767 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0029`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8418`
- variance ratio constrained vs baseline: `1.0176`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0206)
- [ ] Constrained holdout drawdown better than baseline (-0.2423%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 101.0%)
