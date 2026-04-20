# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week9_H`
- constrained artifact stem: `week9_H`
- min history days used after backoff: `24.0`
- market count: `15`
- tuning steps: `4616`
- holdout steps: `1154`
- objective: `1.7`-var / `2.1`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1515 | 0.1435 | -0.0079 |
| Max drawdown | -26.9194% | -26.3512% | +0.5683% |
| Mean return | 0.00105010 | 0.00095891 | -0.00009119 |
| Volatility | 0.03134382 | 0.03090035 | -0.00044347 |

## Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `12.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0520 | -0.0618 |
| Open | Mean return | 0.00032605 | -0.00037813 |
| Open | Volatility | 0.01751349 | 0.01048731 |
| Open | Max drawdown (subset) | -14.4095% | -14.7920% |
| Closed | Sortino | 0.1637 | 0.1691 |
| Closed | Mean return | 0.00114926 | 0.00114202 |
| Closed | Volatility | 0.03278551 | 0.03271470 |
| Closed | Max drawdown (subset) | -26.9194% | -26.3512% |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0808`
- baseline max drawdown (full): `-26.9194%`

## Attribution — What Drove Returns

**Biggest single market:** Will Hubert Hurkacz win the 2026 Men's US Open? (`tennis`) — 85.5% of total return
**Biggest domain:** `tennis` — 85.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Hubert Hurkacz win the 2026 Men's US Open? | `tennis` | 0.945584 | 85.5% | 0.0690 |
| 2 | Will Wesley Hunt win the 2026 Texas Republican Primary? | `republican-primary` | 0.188536 | 17.0% | 0.0717 |
| 3 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | 0.142645 | 12.9% | 0.0706 |
| 4 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.101636 | -9.2% | 0.0600 |
| 5 | Will Kamala Harris announce a Presidential run before 2027? | `celebrities` | 0.097756 | 8.8% | 0.0716 |
| 6 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067838 | 6.1% | 0.0532 |
| 7 | Will the Phoenix Suns win the NBA Western Conference Finals? | `nba` | -0.060525 | -5.5% | 0.0682 |
| 8 | Will Israel strike 4 countries in 2026? | `middle-east` | -0.054220 | -4.9% | 0.0708 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.052491 | -4.7% | 0.0476 |
| 10 | Will the Democrats win the South Dakota Senate race in 2026? | `south-dakota-midterm` | -0.048156 | -4.4% | 0.0698 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `tennis` | 0.945584 | 85.5% |
| 2 | `republican-primary` | 0.188536 | 17.0% |
| 3 | `primaries` | 0.142645 | 12.9% |
| 4 | `claude-5` | -0.101636 | -9.2% |
| 5 | `celebrities` | 0.097756 | 8.8% |
| 6 | `nba` | -0.060525 | -5.5% |
| 7 | `middle-east` | -0.054220 | -4.9% |
| 8 | `south-dakota-midterm` | -0.048156 | -4.4% |
| 9 | `hockey` | 0.039337 | 3.6% |
| 10 | `nba-champion` | -0.019396 | -1.8% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Israel strike 4 countries in 2026? | Will the Los Angeles Kings win the Western Conference? | -0.2557 |
| Will the Phoenix Suns win the NBA Western Conference Finals? | Will the Los Angeles Kings win the Western Conference? | 0.2146 |
| Will Kamala Harris announce a Presidential run before 2027? | Will Israel strike 4 countries in 2026? | 0.0602 |
| Will Rennes win the 2025–26 French Ligue 1? | Will Israel strike 4 countries in 2026? | 0.0160 |
| Will Claude 5 be released by April 30, 2026? | Will the Democrats win the South Dakota Senate race in 2026? | 0.0147 |

## Correlation and Risk Structure
- category count: `14`
- avg abs category correlation: `0.0027`
- max abs category correlation: `0.0807`
- top eigenvalue share: `0.7807`
- variance ratio constrained vs baseline: `0.9863`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0079)
- [x] Constrained holdout drawdown better than baseline (+0.5683%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0807)
- [ ] No single domain dominates returns (top domain share: 85.5%)
