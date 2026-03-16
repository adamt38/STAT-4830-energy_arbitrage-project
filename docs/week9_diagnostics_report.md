# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `10360`
- holdout steps: `2590`
- objective: `0.0`-var / `0.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0434 | 0.0434 | -0.0000 |
| Max drawdown | -5.4878% | -5.4935% | -0.0057% |
| Mean return | 0.00005888 | 0.00005889 | +0.00000001 |
| Volatility | 0.00255322 | 0.00255359 | +0.00000037 |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0213`
- baseline max drawdown (full): `-8.8416%`

## Attribution — What Drove Returns

**Biggest single market:** Will Anthropic’s market cap be less than $100B at market close on IPO day? (`anthropic`) — 31.7% of total return
**Biggest domain:** `anthropic` — 31.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Anthropic’s market cap be less than $100B at market close on IPO day? | `anthropic` | 0.048391 | 31.7% | 0.0250 |
| 2 | Will the Fed increase interest rates by 25+ bps after the March 2026 meeting? | `economic-policy` | 0.039936 | 26.2% | 0.0250 |
| 3 | Will the Republican Party hold exactly 54 Senate seats after the 2026 midterm elections? | `congress` | 0.038866 | 25.5% | 0.0250 |
| 4 | Will Steve Hilton win the California Governor Election in 2026? | `california-midterm` | 0.028938 | 19.0% | 0.0250 |
| 5 | Will Leverkusen win the 2025–26 Bundesliga? | `bundesliga` | 0.024736 | 16.2% | 0.0250 |
| 6 | Will Tyrese Maxey win the 2025–2026 NBA MVP? | `awards` | -0.019572 | -12.8% | 0.0250 |
| 7 | Will Mistral have the best AI model at the end of March 2026? | `gemini-3` | -0.016667 | -10.9% | 0.0250 |
| 8 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | -0.016652 | -10.9% | 0.0250 |
| 9 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.015094 | 9.9% | 0.0250 |
| 10 | Will Lance Stroll be the 2026 F1 Drivers' Champion? | `formula1` | 0.009996 | 6.6% | 0.0250 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `anthropic` | 0.048391 | 31.7% |
| 2 | `economic-policy` | 0.039936 | 26.2% |
| 3 | `congress` | 0.038866 | 25.5% |
| 4 | `california-midterm` | 0.028938 | 19.0% |
| 5 | `bundesliga` | 0.024736 | 16.2% |
| 6 | `awards` | -0.019572 | -12.8% |
| 7 | `gemini-3` | -0.016667 | -10.9% |
| 8 | `colombia-election` | -0.016652 | -10.9% |
| 9 | `best-of-2025` | 0.015094 | 9.9% |
| 10 | `formula1` | 0.009996 | 6.6% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| MicroStrategy sells any Bitcoin by June 30, 2026? | Extended FDV above $500M one day after launch? | 0.1385 |
| Opensea FDV above $500M one day after launch? | Will Bitmine announce that it holds more than 7M ETH before 2027? | 0.1194 |
| Will Tyrese Maxey win the 2025–2026 NBA MVP? | MicroStrategy sells any Bitcoin by June 30, 2026? | -0.0954 |
| Will Bitcoin have the best performance in 2026? | Opensea FDV above $500M one day after launch? | 0.0423 |
| Will Anthropic’s market cap be less than $100B at market close on IPO day? | Will the Republican Party hold exactly 54 Senate seats after the 2026 midterm elections? | 0.0279 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0020`
- max abs category correlation: `0.1440`
- top eigenvalue share: `0.3519`
- variance ratio constrained vs baseline: `0.9984`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0000)
- [ ] Constrained holdout drawdown better than baseline (-0.0057%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1440)
- [x] No single domain dominates returns (top domain share: 31.7%)
