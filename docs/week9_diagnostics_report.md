# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_G_seed42`
- constrained artifact stem: `week9_G_seed42`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6078`
- holdout steps: `1520`
- objective: `1.6`-var / `2.4`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0169 | 0.0236 | +0.0067 |
| Max drawdown | -29.0667% | -26.0608% | +3.0059% |
| Mean return | 0.00007957 | 0.00009904 | +0.00001947 |
| Volatility | 0.00859280 | 0.00814467 | -0.00044813 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `16.2%`
- Holdout steps marked exog-stale: `100.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | -0.1056 | -0.0692 |
| Open | Mean return | -0.00045060 | -0.00027320 |
| Open | Volatility | 0.00750661 | 0.00838189 |
| Open | Max drawdown (subset) | -11.0185% | -9.0293% |
| Closed | Sortino | 0.0380 | 0.0403 |
| Closed | Mean return | 0.00018194 | 0.00017092 |
| Closed | Volatility | 0.00878340 | 0.00809610 |
| Closed | Max drawdown (subset) | -27.3608% | -24.2090% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0645`
- baseline max drawdown (full): `-29.0667%`

## Attribution — What Drove Returns

**Biggest single market:** Will AC Milan win the 2025–26 Serie A league? (`serie-a`) — 59.6% of total return
**Biggest domain:** `colombia-election` — 102.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.089751 | 59.6% | 0.0501 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.087543 | 58.2% | 0.0499 |
| 3 | Will there be fewer than 5 earthquakes of magnitude 7.0 or higher worldwide in 2026? | `natural-disasters` | -0.085929 | -57.1% | 0.0501 |
| 4 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.071877 | 47.7% | 0.0500 |
| 5 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067594 | 44.9% | 0.0499 |
| 6 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066827 | 44.4% | 0.0499 |
| 7 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.060435 | 40.1% | 0.0500 |
| 8 | USD.AI FDV above $6B one day after launch? | `usdptai` | -0.055964 | -37.2% | 0.0501 |
| 9 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.046224 | -30.7% | 0.0501 |
| 10 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.043295 | 28.8% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.154370 | 102.5% |
| 2 | `serie-a` | 0.089751 | 59.6% |
| 3 | `natural-disasters` | -0.085929 | -57.1% |
| 4 | `roland-garros` | -0.072360 | -48.1% |
| 5 | `best-of-2025` | 0.071877 | 47.7% |
| 6 | `basketball` | 0.060435 | 40.1% |
| 7 | `usdptai` | -0.055964 | -37.2% |
| 8 | `primaries` | -0.046224 | -30.7% |
| 9 | `nba-champion` | 0.043295 | 28.8% |
| 10 | `uefa-europa-league` | 0.040672 | 27.0% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0412 |
| Will Taylor Fritz win the 2026 Men's French Open? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0338 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will the Ottawa Senators win the Eastern Conference? | -0.0263 |
| Will Nott'm Forest win the 2025-26 UEFA Europa League? | Will Daniil Medvedev win the 2026 Men's French Open? | -0.0203 |
| Will Real Madrid win the 2025–26 La Liga? | Will the Ottawa Senators win the Eastern Conference? | -0.0073 |

## Correlation and Risk Structure
- category count: `16`
- avg abs category correlation: `0.0009`
- max abs category correlation: `0.0254`
- top eigenvalue share: `0.8018`
- variance ratio constrained vs baseline: `0.6742`

## Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0067)
- [x] Constrained holdout drawdown better than baseline (+3.0059%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0254)
- [ ] No single domain dominates returns (top domain share: 102.5%)
