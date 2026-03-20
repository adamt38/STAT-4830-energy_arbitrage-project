# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `45682`
- holdout steps: `11421`
- objective: `0.6`-var / `1.6`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0327 | 0.0187 | -0.0140 |
| Max drawdown | -5.6839% | -6.1521% | -0.4681% |
| Mean return | 0.00002722 | 0.00001727 | -0.00000996 |
| Volatility | 0.00153098 | 0.00158175 | +0.00005077 |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0133`
- baseline max drawdown (full): `-5.9775%`

## Attribution — What Drove Returns

**Biggest single market:** Will Anthropic’s market cap be less than $100B at market close on IPO day? (`anthropic`) — 34.7% of total return
**Biggest domain:** `anthropic` — 34.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Anthropic’s market cap be less than $100B at market close on IPO day? | `anthropic` | 0.068420 | 34.7% | 0.0254 |
| 2 | Will Stuttgart win the 2025–26 Bundesliga? | `bundesliga` | 0.033621 | 17.0% | 0.0252 |
| 3 | Will Giannis Antetokounmpo win the 2025–2026 NBA MVP? | `awards` | 0.031335 | 15.9% | 0.0251 |
| 4 | Will Chad Bianco win the California Governor Election in 2026? | `california-midterm` | 0.021931 | 11.1% | 0.0251 |
| 5 | Will the Fed decrease interest rates by 50+ bps after the March 2026 meeting? | `economic-policy` | 0.019130 | 9.7% | 0.0250 |
| 6 | Will Meituan have the best AI model at the end of March 2026? | `gemini-3` | 0.017274 | 8.8% | 0.0250 |
| 7 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | -0.016693 | -8.5% | 0.0249 |
| 8 | Felix Protocol FDV above $2B one day after launch? | `felix` | 0.015947 | 8.1% | 0.0251 |
| 9 | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | `brazil` | -0.009365 | -4.7% | 0.0249 |
| 10 | Extended FDV above $300M one day after launch? | `extended` | 0.008800 | 4.5% | 0.0250 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `anthropic` | 0.068420 | 34.7% |
| 2 | `bundesliga` | 0.033621 | 17.0% |
| 3 | `awards` | 0.031335 | 15.9% |
| 4 | `california-midterm` | 0.021931 | 11.1% |
| 5 | `economic-policy` | 0.019130 | 9.7% |
| 6 | `gemini-3` | 0.017274 | 8.8% |
| 7 | `colombia-election` | -0.016693 | -8.5% |
| 8 | `felix` | 0.015947 | 8.1% |
| 9 | `brazil` | -0.009365 | -4.7% |
| 10 | `extended` | 0.008800 | 4.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $2B one day after launch? | Will Rand Paul announce a presidential run before 2027? | 0.0597 |
| MicroStrategy sells any Bitcoin by March 31, 2026? | Will Gold have the best performance in 2026? | 0.0483 |
| Felix Protocol FDV above $2B one day after launch? | Will Claude 5 be released by March 31, 2026? | -0.0417 |
| Will Stuttgart win the 2025–26 Bundesliga? | MicroStrategy sells any Bitcoin by March 31, 2026? | 0.0321 |
| Will Stuttgart win the 2025–26 Bundesliga? | Will Chad Bianco win the California Governor Election in 2026? | 0.0294 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0009`
- max abs category correlation: `0.0923`
- top eigenvalue share: `0.3483`
- variance ratio constrained vs baseline: `1.0057`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0140)
- [ ] Constrained holdout drawdown better than baseline (-0.4681%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0923)
- [x] No single domain dominates returns (top domain share: 34.7%)
