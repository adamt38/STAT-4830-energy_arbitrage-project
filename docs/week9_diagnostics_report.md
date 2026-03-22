# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `44168`
- holdout steps: `11042`
- objective: `1.2`-var / `1.2`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0791 | 0.0700 | -0.0091 |
| Max drawdown | -7.1565% | -7.3414% | -0.1849% |
| Mean return | 0.00007378 | 0.00005993 | -0.00001385 |
| Volatility | 0.00336451 | 0.00296682 | -0.00039769 |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0177`
- baseline max drawdown (full): `-7.1565%`

## Attribution — What Drove Returns

**Biggest single market:** Will Databricks’ market cap be $250B or greater at market close on IPO day? (`databricks`) — 41.7% of total return
**Biggest domain:** `databricks` — 41.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Databricks’ market cap be $250B or greater at market close on IPO day? | `databricks` | 0.275660 | 41.7% | 0.0228 |
| 2 | Will the Fed increase interest rates by 25+ bps after the March 2026 meeting? | `economic-policy` | 0.125770 | 19.0% | 0.0247 |
| 3 | Will Giannis Antetokounmpo win the 2025–2026 NBA MVP? | `awards` | 0.067983 | 10.3% | 0.0249 |
| 4 | Will Anthropic’s market cap be less than $100B at market close on IPO day? | `anthropic` | 0.061193 | 9.2% | 0.0250 |
| 5 | Will Chad Bianco win the California Governor Election in 2026? | `california-midterm` | 0.056206 | 8.5% | 0.0250 |
| 6 | Will Stuttgart win the 2025–26 Bundesliga? | `bundesliga` | 0.043709 | 6.6% | 0.0250 |
| 7 | MicroStrategy sells any Bitcoin by March 31, 2026? | `2025-predictions` | 0.038356 | 5.8% | 0.0249 |
| 8 | Will Rand Paul announce a presidential run before 2027? | `celebrities` | 0.026856 | 4.1% | 0.0252 |
| 9 | Will Claude 5 be released by March 31, 2026? | `claude-5` | -0.020991 | -3.2% | 0.0250 |
| 10 | Will Meituan have the best AI model at the end of March 2026? | `gemini-3` | -0.017004 | -2.6% | 0.0250 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `databricks` | 0.275660 | 41.7% |
| 2 | `economic-policy` | 0.125770 | 19.0% |
| 3 | `awards` | 0.067983 | 10.3% |
| 4 | `anthropic` | 0.061193 | 9.2% |
| 5 | `california-midterm` | 0.056206 | 8.5% |
| 6 | `bundesliga` | 0.043709 | 6.6% |
| 7 | `2025-predictions` | 0.038356 | 5.8% |
| 8 | `celebrities` | 0.026856 | 4.1% |
| 9 | `claude-5` | -0.020991 | -3.2% |
| 10 | `gemini-3` | -0.017004 | -2.6% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Stuttgart win the 2025–26 Bundesliga? | Will Gold have the best performance in 2026? | 0.0693 |
| Will Rand Paul announce a presidential run before 2027? | Felix Protocol FDV above $2B one day after launch? | 0.0687 |
| Epstein client list released by June 30? | Will AfD win the most seats in the 2026 Berlin state elections? | 0.0398 |
| MicroStrategy sells any Bitcoin by March 31, 2026? | Will Gold have the best performance in 2026? | -0.0144 |
| Will Chad Bianco win the California Governor Election in 2026? | MicroStrategy sells any Bitcoin by March 31, 2026? | -0.0137 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0008`
- max abs category correlation: `0.0350`
- top eigenvalue share: `0.3937`
- variance ratio constrained vs baseline: `0.9345`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0091)
- [ ] Constrained holdout drawdown better than baseline (-0.1849%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0350)
- [x] No single domain dominates returns (top domain share: 41.7%)
