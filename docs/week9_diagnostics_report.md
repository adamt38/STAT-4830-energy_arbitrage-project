# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8_A`
- constrained artifact stem: `week8_A`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8339`
- holdout steps: `2085`
- objective: `0.7`-var / `2.2`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0642 | 0.0539 | -0.0103 |
| Max drawdown | -8.9604% | -9.1526% | -0.1922% |
| Mean return | 0.00010922 | 0.00008162 | -0.00002761 |
| Volatility | 0.00436554 | 0.00357510 | -0.00079044 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.4%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0816 | 0.3615 |
| Open | Mean return | 0.00017396 | 0.00028612 |
| Open | Volatility | 0.00554517 | 0.00450950 |
| Open | Max drawdown (subset) | -6.8821% | -2.3323% |
| Closed | Sortino | 0.0605 | 0.0276 |
| Closed | Mean return | 0.00009744 | 0.00004440 |
| Closed | Volatility | 0.00411456 | 0.00337604 |
| Closed | Max drawdown (subset) | -8.6729% | -9.1526% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0679`
- baseline max drawdown (full): `-10.2520%`

## Attribution — What Drove Returns

**Biggest single market:** Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? (`basketball`) — 100.8% of total return
**Biggest domain:** `basketball` — 100.8% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.171558 | 100.8% | 0.0149 |
| 2 | Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | `bernie-sanders` | -0.048192 | -28.3% | 0.0196 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.045047 | 26.5% | 0.0279 |
| 4 | MegaETH market cap (FDV) >$3B one day after launch? | `airdrops` | 0.019479 | 11.4% | 0.0248 |
| 5 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.019167 | -11.3% | 0.0209 |
| 6 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.016666 | 9.8% | 0.0241 |
| 7 | Will Toni Atkins win the California Governor Election in 2026? | `california-midterm` | -0.013054 | -7.7% | 0.0278 |
| 8 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.012793 | -7.5% | 0.0285 |
| 9 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.011538 | 6.8% | 0.0205 |
| 10 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | -0.009812 | -5.8% | 0.0208 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `basketball` | 0.171558 | 100.8% |
| 2 | `bernie-sanders` | -0.048192 | -28.3% |
| 3 | `colombia-election` | 0.045047 | 26.5% |
| 4 | `airdrops` | 0.019479 | 11.4% |
| 5 | `economic-policy` | -0.019167 | -11.3% |
| 6 | `fifa-world-cup` | 0.016666 | 9.8% |
| 7 | `california-midterm` | -0.013054 | -7.7% |
| 8 | `2026-fifa-world-cup` | -0.012793 | -7.5% |
| 9 | `comex-silver-futures` | 0.011538 | 6.8% |
| 10 | `best-of-2025` | -0.009812 | -5.8% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | Will Somaliland join the Abraham Accords before 2027? | -0.1890 |
| Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | Will Trump acquire Greenland before 2027? | 0.1777 |
| Extended FDV above $300M one day after launch? | Will Trump acquire Greenland before 2027? | 0.1009 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.1003 |
| Will Silver (SI) hit (HIGH) $250 by end of June? | Will Microsoft be the largest company in the world by market cap on December 31? | -0.0875 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0017`
- max abs category correlation: `0.1132`
- top eigenvalue share: `0.8341`
- variance ratio constrained vs baseline: `0.6718`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0103)
- [ ] Constrained holdout drawdown better than baseline (-0.1922%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1132)
- [ ] No single domain dominates returns (top domain share: 100.8%)
