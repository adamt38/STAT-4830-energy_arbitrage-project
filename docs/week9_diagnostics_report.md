# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8_B`
- constrained artifact stem: `week8_B`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8340`
- holdout steps: `2085`
- objective: `1.0`-var / `2.3`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0405 | 0.0063 | -0.0342 |
| Max drawdown | -9.0698% | -13.7333% | -4.6635% |
| Mean return | 0.00006136 | 0.00001079 | -0.00005057 |
| Volatility | 0.00378400 | 0.00371290 | -0.00007110 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1521 | 0.1186 |
| Open | Mean return | 0.00026148 | 0.00017693 |
| Open | Volatility | 0.00540253 | 0.00416246 |
| Open | Max drawdown (subset) | -5.1308% | -4.4775% |
| Closed | Sortino | 0.0177 | -0.0105 |
| Closed | Mean return | 0.00002614 | -0.00001844 |
| Closed | Volatility | 0.00341963 | 0.00362725 |
| Closed | Max drawdown (subset) | -9.0698% | -13.7333% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0662`
- baseline max drawdown (full): `-10.8395%`

## Attribution — What Drove Returns

**Biggest single market:** Will Dortmund win the 2025–26 Bundesliga? (`bundesliga`) — -290.3% of total return
**Biggest domain:** `bundesliga` — -290.3% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Dortmund win the 2025–26 Bundesliga? | `bundesliga` | -0.065319 | -290.3% | 0.0246 |
| 2 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.057785 | 256.8% | 0.0247 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033938 | 150.8% | 0.0249 |
| 4 | Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | `bernie-sanders` | -0.025092 | -111.5% | 0.0245 |
| 5 | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | `economic-policy` | -0.022482 | -99.9% | 0.0244 |
| 6 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.022170 | -98.5% | 0.0250 |
| 7 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.017807 | 79.1% | 0.0248 |
| 8 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.017758 | 78.9% | 0.0244 |
| 9 | Will Gold (GC) hit (HIGH) $8,000 by end of June? | `comex-gold-futures` | 0.013258 | 58.9% | 0.0251 |
| 10 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.012419 | -55.2% | 0.0243 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `bundesliga` | -0.065319 | -290.3% |
| 2 | `felix` | 0.057785 | 256.8% |
| 3 | `colombia-election` | 0.033938 | 150.8% |
| 4 | `bernie-sanders` | -0.025092 | -111.5% |
| 5 | `economic-policy` | -0.022482 | -99.9% |
| 6 | `2026-fifa-world-cup` | -0.022170 | -98.5% |
| 7 | `colorado-midterm` | 0.017807 | 79.1% |
| 8 | `best-of-2025` | 0.017758 | 78.9% |
| 9 | `comex-gold-futures` | 0.013258 | 58.9% |
| 10 | `claude-5` | -0.012419 | -55.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8594 |
| Will Somaliland join the Abraham Accords before 2027? | Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | -0.1860 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.1002 |
| Will Dortmund win the 2025–26 Bundesliga? | Will Gold (GC) hit (HIGH) $8,000 by end of June? | -0.0313 |
| Will Microsoft be the largest company in the world by market cap on December 31? | Will any presidential candidate win outright in the first round of the Brazil election? | 0.0283 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0023`
- max abs category correlation: `0.2661`
- top eigenvalue share: `0.8483`
- variance ratio constrained vs baseline: `0.9590`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0342)
- [ ] Constrained holdout drawdown better than baseline (-4.6635%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2661)
- [x] No single domain dominates returns (top domain share: -290.3%)
