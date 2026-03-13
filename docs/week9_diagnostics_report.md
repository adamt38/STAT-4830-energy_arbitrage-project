# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8`
- min history days used after backoff: `24.0`
- market count: `260`
- tuning steps: `12680`
- holdout steps: `3170`
- objective: `0.0`-var / `0.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.3297 | 0.3210 | -0.0087 |
| Max drawdown | -1.5619% | -1.7010% | -0.1391% |
| Mean return | 0.00017278 | 0.00016916 | -0.00000362 |
| Volatility | 0.00223399 | 0.00210974 | -0.00012425 |

## Full-Series Baseline Reference
- baseline sortino (full): `0.2101`
- baseline max drawdown (full): `-3.5834%`

## Attribution — What Drove Returns

**Biggest single market:** Will the Edmonton Oilers win the 2025–2026 NHL Presidents' Trophy? (`other`) — 24.9% of total return
**Biggest domain:** `other` — 34.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Edmonton Oilers win the 2025–2026 NHL Presidents' Trophy? | `other` | 0.133567 | 24.9% | 0.0038 |
| 2 | Will the Winnipeg Jets win the 2025–2026 NHL Presidents' Trophy? | `other` | 0.051513 | 9.6% | 0.0038 |
| 3 | Will Claude 5 be released by March 15, 2026? | `claude-5` | 0.029008 | 5.4% | 0.0038 |
| 4 | Will William rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.018461 | 3.4% | 0.0038 |
| 5 | Will the Republican Party win the MI-08 House seat? | `michigan-midterm` | 0.016447 | 3.1% | 0.0039 |
| 6 | Will Discord’s market cap be between $15B and $20B at market close on IPO day? | `tech` | 0.016034 | 3.0% | 0.0038 |
| 7 | Will OpenAI launch a new consumer hardware product by March 31, 2026? | `sam-altman` | 0.013066 | 2.4% | 0.0038 |
| 8 | Will David Belliard win the Paris mayor election? | `mayoral-elections` | 0.012179 | 2.3% | 0.0038 |
| 9 | Will the Republican Party win the MN-02 House seat? | `minnesota-midterm` | 0.011557 | 2.2% | 0.0039 |
| 10 | Will the 10-year treasury yield hit 5.0% by March 31? | `jerome-powell` | 0.011212 | 2.1% | 0.0038 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `other` | 0.185080 | 34.5% |
| 2 | `claude-5` | 0.030694 | 5.7% |
| 3 | `best-of-2025` | 0.029114 | 5.4% |
| 4 | `tech` | 0.022343 | 4.2% |
| 5 | `jerome-powell` | 0.019250 | 3.6% |
| 6 | `michigan-midterm` | 0.016697 | 3.1% |
| 7 | `bundesliga` | 0.014193 | 2.6% |
| 8 | `serie-a` | 0.014062 | 2.6% |
| 9 | `natural-disasters` | 0.013221 | 2.5% |
| 10 | `sam-altman` | 0.012434 | 2.3% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Discord’s market cap be between $15B and $20B at market close on IPO day? | Reya FDV above $200M one day after launch? | -0.0558 |
| Will the Republican Party win the MI-08 House seat? | Will the 10-year treasury yield hit 5.0% by March 31? | 0.0525 |
| Will the Republican Party win the MN-02 House seat? | Will Hoffenheim win the 2025–26 Bundesliga? | -0.0303 |
| Will OpenAI launch a new consumer hardware product by March 31, 2026? | Will Atalanta win the 2025–26 Serie A league? | 0.0280 |
| Will the Republican Party win the MN-02 House seat? | Will the Nashville Predators win the Western Conference? | -0.0264 |

## Correlation and Risk Structure
- category count: `118`
- avg abs category correlation: `0.0027`
- max abs category correlation: `0.1885`
- top eigenvalue share: `0.2313`
- variance ratio constrained vs baseline: `1.0010`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0087)
- [ ] Constrained holdout drawdown better than baseline (-0.1391%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1885)
- [x] No single domain dominates returns (top domain share: 34.5%)
