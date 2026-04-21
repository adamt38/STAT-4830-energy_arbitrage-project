# Week 9 Diagnostics Report

## Run Context
- artifact prefix: `week16_rr_beatbaseline`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `48474`
- holdout steps: `12119`
- objective: `0.5`-var / `1.0`-downside mean-downside surrogate

## Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1798 | 0.1538 | -0.0260 |
| Max drawdown | -5.4547% | -9.3999% | -3.9452% |
| Mean return | 0.00018792 | 0.00028244 | +0.00009452 |
| Volatility | 0.00522424 | 0.00905327 | +0.00382903 |

## Full-Series Baseline Reference
- baseline sortino (full): `0.0520`
- baseline max drawdown (full): `-8.2541%`

## Attribution — What Drove Returns

**Biggest single market:** Will Lucas rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 98.2% of total return
**Biggest domain:** `best-of-2025` — 98.2% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Lucas rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 3.362122 | 98.2% | 0.0329 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.085205 | 2.5% | 0.0285 |
| 3 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.059443 | 1.7% | 0.0223 |
| 4 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.040266 | -1.2% | 0.0077 |
| 5 | Felix Protocol FDV above $300M one day after launch? | `felix` | -0.030915 | -0.9% | 0.0229 |
| 6 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.013928 | -0.4% | 0.0199 |
| 7 | Epstein client list released by June 30? | `epstein` | 0.011220 | 0.3% | 0.0759 |
| 8 | Extended FDV above $300M one day after launch? | `extended` | 0.007832 | 0.2% | 0.0308 |
| 9 | Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | `bernie-sanders` | 0.006218 | 0.2% | 0.0112 |
| 10 | Will Arsenal win the 2025–26 Champions League? | `champions-league` | -0.006113 | -0.2% | 0.0136 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 3.362122 | 98.2% |
| 2 | `colombia-election` | 0.085205 | 2.5% |
| 3 | `colorado-midterm` | 0.059443 | 1.7% |
| 4 | `claude-5` | -0.040266 | -1.2% |
| 5 | `felix` | -0.030915 | -0.9% |
| 6 | `2026-fifa-world-cup` | -0.013928 | -0.4% |
| 7 | `epstein` | 0.011220 | 0.3% |
| 8 | `extended` | 0.007832 | 0.2% |
| 9 | `bernie-sanders` | 0.006218 | 0.2% |
| 10 | `champions-league` | -0.006113 | -0.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | Will Erin Stewart win the 2026 Connecticut Governor Republican primary election? | -0.0723 |
| Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | Will Europe win the 2026 FIFA World Cup? | -0.0270 |
| Will Claude 5 be released by April 30, 2026? | Will Luiz Inácio Lula da Silva qualify for Brazil's presidential runoff? | 0.0196 |
| Extended FDV above $300M one day after launch? | Will Arsenal win the 2025–26 Champions League? | -0.0162 |
| Will 2026 be the second-hottest year on record? | Base FDV above $2B one day after launch? | -0.0114 |

## Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0009`
- max abs category correlation: `0.0543`
- top eigenvalue share: `0.5408`
- variance ratio constrained vs baseline: `1.1640`

## Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0260)
- [ ] Constrained holdout drawdown better than baseline (-3.9452%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0543)
- [ ] No single domain dominates returns (top domain share: 98.2%)
