# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week8_C`
- constrained artifact stem: `week8_C_macro_explicit`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `7868`
- holdout steps: `1968`
- objective: `1.0`-var / `2.3`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0812 | 0.0215 | -0.0597 |
| Max drawdown | -9.1099% | -9.5640% | -0.4541% |
| Mean return | 0.00015022 | 0.00002890 | -0.00012131 |
| Volatility | 0.00479357 | 0.00344995 | -0.00134362 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.6%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0357 | 0.0903 |
| Open | Mean return | 0.00008204 | 0.00016183 |
| Open | Volatility | 0.00562186 | 0.00671266 |
| Open | Max drawdown (subset) | -8.1314% | -5.1209% |
| Closed | Sortino | 0.0928 | 0.0035 |
| Closed | Mean return | 0.00016282 | 0.00000433 |
| Closed | Volatility | 0.00462415 | 0.00240204 |
| Closed | Max drawdown (subset) | -8.0609% | -7.1420% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0699`
- baseline max drawdown (full): `-12.8811%`

## Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 171.2% of total return
**Biggest domain:** `felix` — 171.2% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.097371 | 171.2% | 0.0250 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.047117 | -82.8% | 0.0242 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033397 | 58.7% | 0.0247 |
| 4 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.032402 | -57.0% | 0.0248 |
| 5 | Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | `bernie-sanders` | -0.029216 | -51.4% | 0.0244 |
| 6 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.028183 | 49.5% | 0.0243 |
| 7 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.017229 | 30.3% | 0.0249 |
| 8 | Will Gold (GC) hit (HIGH) $8,000 by end of June? | `comex-gold-futures` | 0.011864 | 20.9% | 0.0252 |
| 9 | Will J.D. Vance announce a presidential run before 2027? | `celebrities` | -0.011186 | -19.7% | 0.0247 |
| 10 | Will Toni Atkins win the California Governor Election in 2026? | `california-midterm` | -0.010123 | -17.8% | 0.0248 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.097371 | 171.2% |
| 2 | `claude-5` | -0.047117 | -82.8% |
| 3 | `colombia-election` | 0.033397 | 58.7% |
| 4 | `economic-policy` | -0.032402 | -57.0% |
| 5 | `bernie-sanders` | -0.029216 | -51.4% |
| 6 | `best-of-2025` | 0.028183 | 49.5% |
| 7 | `colorado-midterm` | 0.017229 | 30.3% |
| 8 | `comex-gold-futures` | 0.011864 | 20.9% |
| 9 | `celebrities` | -0.011186 | -19.7% |
| 10 | `california-midterm` | -0.010123 | -17.8% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8588 |
| Will Somaliland join the Abraham Accords before 2027? | Will the number of Democratic House members who retire in 2026 be between 24 and 27 inclusive? | -0.1860 |
| Will Claude 5 be released by April 30, 2026? | Will Daniel Quintero win the 2026 Colombian presidential election? | 0.0997 |
| Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | MegaETH market cap (FDV) >$3B one day after launch? | 0.0274 |
| Will Claude 5 be released by April 30, 2026? | Will OpenAI acquire Pinterest in 2026? | -0.0214 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.2653`
- top eigenvalue share: `0.8258`
- variance ratio constrained vs baseline: `0.9551`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0597)
- [ ] Constrained holdout drawdown better than baseline (-0.4541%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2653)
- [ ] No single domain dominates returns (top domain share: 171.2%)
