# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week11_M`
- constrained artifact stem: `week11_M_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8225`
- holdout steps: `2057`
- objective: `2.0`-var / `2.7`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0179 | -0.0301 | -0.0480 |
| Max drawdown | -6.1161% | -17.3875% | -11.2714% |
| Mean return | 0.00002363 | -0.00004465 | -0.00006828 |
| Volatility | 0.00247649 | 0.00321885 | +0.00074236 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.2%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0270 | 0.1266 |
| Open | Mean return | 0.00003636 | 0.00014865 |
| Open | Volatility | 0.00320848 | 0.00609221 |
| Open | Max drawdown (subset) | -4.8760% | -4.8782% |
| Closed | Sortino | 0.0159 | -0.0551 |
| Closed | Mean return | 0.00002099 | -0.00008483 |
| Closed | Volatility | 0.00229520 | 0.00218868 |
| Closed | Max drawdown (subset) | -6.4708% | -16.9407% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0558`
- baseline max drawdown (full): `-8.4628%`

## Attribution — What Drove Returns

**Biggest single market:** Will Daniel Quintero win the 2026 Colombian presidential election? (`colombia-election`) — -104.5% of total return
**Biggest domain:** `colombia-election` — -104.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.095970 | -104.5% | 0.0314 |
| 2 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | -0.056405 | 61.4% | 0.0189 |
| 3 | Will Victor Wembanyama win the 2025–2026 NBA MVP? | `awards` | -0.035883 | 39.1% | 0.0192 |
| 4 | Epstein client list released by June 30? | `epstein` | -0.033025 | 36.0% | 0.0186 |
| 5 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | -0.027789 | 30.3% | 0.0156 |
| 6 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.027783 | -30.3% | 0.0218 |
| 7 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.027067 | 29.5% | 0.0198 |
| 8 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.017783 | 19.4% | 0.0145 |
| 9 | Will Saudi Aramco be the largest company in the world by market cap on December 31? | `business` | -0.009556 | 10.4% | 0.0187 |
| 10 | Will Bernie endorse Dan Osborn for NE-Sen by Nov 2 2026 ET? | `bernie-sanders` | -0.009158 | 10.0% | 0.0177 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.095970 | -104.5% |
| 2 | `basketball` | -0.056405 | 61.4% |
| 3 | `awards` | -0.035883 | 39.1% |
| 4 | `epstein` | -0.033025 | 36.0% |
| 5 | `best-of-2025` | -0.027789 | 30.3% |
| 6 | `fifa-world-cup` | 0.027783 | -30.3% |
| 7 | `congress` | -0.027067 | 29.5% |
| 8 | `economic-policy` | -0.017783 | 19.4% |
| 9 | `business` | -0.009556 | 10.4% |
| 10 | `bernie-sanders` | -0.009158 | 10.0% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | Will Trump acquire Greenland before 2027? | 0.1705 |
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | -0.1587 |
| Epstein client list released by June 30? | Will Saudi Aramco be the largest company in the world by market cap on December 31? | 0.0629 |
| Will Trump acquire Greenland before 2027? | Extended FDV above $300M one day after launch? | 0.0614 |
| MegaETH market cap (FDV) >$3B one day after launch? | Will Trump acquire Greenland before 2027? | -0.0586 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0012`
- max abs category correlation: `0.1009`
- top eigenvalue share: `0.9447`
- variance ratio constrained vs baseline: `0.4078`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0480)
- [ ] Constrained holdout drawdown better than baseline (-11.2714%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1009)
- [x] No single domain dominates returns (top domain share: -104.5%)
