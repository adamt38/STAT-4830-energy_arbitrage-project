# Week 9 Diagnostics Report

This file aggregates **multiple cloud pod runs** that were merged into `main` on different branches. Each section is self-contained for one artifact prefix.

---

## Run: `week11_I` (source branch: `cloud-runs-I4`)

### Run Context
- artifact prefix: `week11_I`
- constrained artifact stem: `week11_I_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6197`
- holdout steps: `1550`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0963 | 0.1040 | +0.0077 |
| Max drawdown | -29.6971% | -28.5396% | +1.1575% |
| Mean return | 0.00033985 | 0.00036985 | +0.00003000 |
| Volatility | 0.00921816 | 0.00940606 | +0.00018790 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.2606 | 0.3216 |
| Open | Mean return | 0.00054150 | 0.00067592 |
| Open | Volatility | 0.00609932 | 0.00788247 |
| Open | Max drawdown (subset) | -3.4018% | -3.2368% |
| Closed | Sortino | 0.0794 | 0.0811 |
| Closed | Mean return | 0.00029845 | 0.00030702 |
| Closed | Volatility | 0.00973507 | 0.00968804 |
| Closed | Max drawdown (subset) | -29.4053% | -28.2444% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0369`
- baseline max drawdown (full): `-29.6971%`

### Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 34.7% of total return
**Biggest domain:** `best-of-2025` — 34.7% of total return

#### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.198840 | 34.7% | 0.0498 |
| 2 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.138570 | 24.2% | 0.0500 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.092983 | 16.2% | 0.0501 |
| 4 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.086616 | 15.1% | 0.0499 |
| 5 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.075581 | 13.2% | 0.0500 |
| 6 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066231 | 11.6% | 0.0499 |
| 7 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.062327 | 10.9% | 0.0500 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.048438 | -8.4% | 0.0500 |
| 9 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.047060 | -8.2% | 0.0500 |
| 10 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044579 | -7.8% | 0.0500 |

#### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.198840 | 34.7% |
| 2 | `colombia-election` | 0.152847 | 26.7% |
| 3 | `league-of-legends` | 0.138570 | 24.2% |
| 4 | `serie-a` | 0.092983 | 16.2% |
| 5 | `nba-champion` | 0.075581 | 13.2% |
| 6 | `taylor-swift` | 0.062327 | 10.9% |
| 7 | `new-mexico-primary` | -0.048438 | -8.4% |
| 8 | `primaries` | -0.047060 | -8.2% |
| 9 | `ufc` | -0.044579 | -7.8% |
| 10 | `ligue-1` | -0.041397 | -7.2% |

#### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0974 |
| Will Taylor Fritz win the 2026 Men's French Open? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0338 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0105 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Starmer out by June 30, 2026? | -0.0098 |
| Will Connie Chan receive the most votes in the CA-11 primary? | Will Petr Yan fight Pedro Munhoz next? | 0.0062 |

### Correlation and Risk Structure
- category count: `19`
- avg abs category correlation: `0.0011`
- max abs category correlation: `0.0474`
- top eigenvalue share: `0.3273`
- variance ratio constrained vs baseline: `0.9421`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0077)
- [x] Constrained holdout drawdown better than baseline (+1.1575%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0474)
- [x] No single domain dominates returns (top domain share: 34.7%)

---

## Run: `week13_S5` (source branch: `cloud-runs-S5`)

### Run Context
- artifact prefix: `week13_S5`
- constrained artifact stem: `week13_S5_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5634`
- holdout steps: `1409`
- objective: `1.1`-var / `2.8`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0755 | 0.0843 | +0.0088 |
| Max drawdown | -17.3682% | -18.8047% | -1.4365% |
| Mean return | 0.00028744 | 0.00032649 | +0.00003905 |
| Volatility | 0.00731503 | 0.00830711 | +0.00099209 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.2%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0261 | 0.0909 |
| Open | Mean return | 0.00011755 | 0.00037560 |
| Open | Volatility | 0.00822882 | 0.01046951 |
| Open | Max drawdown (subset) | -13.0319% | -12.8598% |
| Closed | Sortino | 0.0895 | 0.0827 |
| Closed | Mean return | 0.00032517 | 0.00031559 |
| Closed | Volatility | 0.00709564 | 0.00774549 |
| Closed | Max drawdown (subset) | -16.7868% | -16.4467% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0486`
- baseline max drawdown (full): `-17.9351%`

### Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 23.6% of total return
**Biggest domain:** `colombia-election` — 32.9% of total return

#### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.108447 | 23.6% | 0.0477 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.086227 | 18.7% | 0.0497 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.072141 | 15.7% | 0.0509 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067507 | 14.7% | 0.0497 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.065172 | 14.2% | 0.0497 |
| 6 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.046446 | 10.1% | 0.0511 |
| 7 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.045027 | 9.8% | 0.0509 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.044667 | -9.7% | 0.0493 |
| 9 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.040633 | 8.8% | 0.0495 |
| 10 | Will Allen Waters be the Republican nominee for Senate in Rhode Island? | `rhode-island-primary` | 0.036683 | 8.0% | 0.0506 |

#### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.151399 | 32.9% |
| 2 | `nba-champion` | 0.149080 | 32.4% |
| 3 | `serie-a` | 0.072141 | 15.7% |
| 4 | `uefa-europa-league` | 0.046446 | 10.1% |
| 5 | `airdrops` | 0.045027 | 9.8% |
| 6 | `rhode-island-primary` | 0.036683 | 8.0% |
| 7 | `roland-garros` | -0.036179 | -7.9% |
| 8 | `ligue-1` | 0.035965 | 7.8% |
| 9 | `hockey` | -0.031296 | -6.8% |
| 10 | `stanley-cup` | -0.028683 | -6.2% |

#### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3929 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1685 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |
| Will the Portland Trail Blazers win the 2026 NBA Finals? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0919 |
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0408 |

### Correlation and Risk Structure
- category count: `15`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.0613`
- top eigenvalue share: `0.8332`
- variance ratio constrained vs baseline: `0.6902`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0088)
- [ ] Constrained holdout drawdown better than baseline (-1.4365%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 32.9%)

---

## Run: `week11_K` (source branch: `cloud-runs-K4`)

### Run Context
- artifact prefix: `week11_K`
- constrained artifact stem: `week11_K_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6424`
- holdout steps: `1606`
- objective: `2.0`-var / `2.7`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0618 | 0.0615 | -0.0003 |
| Max drawdown | -29.1169% | -28.0285% | +1.0884% |
| Mean return | 0.00024762 | 0.00023892 | -0.00000871 |
| Volatility | 0.00821342 | 0.00817748 | -0.00003594 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.2%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1301 | 0.1817 |
| Open | Mean return | 0.00037162 | 0.00048149 |
| Open | Volatility | 0.00655537 | 0.00790220 |
| Open | Max drawdown (subset) | -5.6269% | -5.7454% |
| Closed | Sortino | 0.0528 | 0.0461 |
| Closed | Mean return | 0.00022189 | 0.00018858 |
| Closed | Volatility | 0.00851692 | 0.00823256 |
| Closed | Max drawdown (subset) | -27.8410% | -26.8757% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0579`
- baseline max drawdown (full): `-29.1169%`

### Attribution — What Drove Returns

**Biggest single market:** Will Ninjas in Pyjamas win the LPL 2026 season? (`league-of-legends`) — 36.2% of total return
**Biggest domain:** `colombia-election` — 40.5% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.139033 | 36.2% | 0.0500 |
| 2 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.088711 | 23.1% | 0.0501 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.088668 | 23.1% | 0.0498 |
| 4 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.076466 | 19.9% | 0.0500 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066740 | 17.4% | 0.0499 |
| 6 | Will Jack Draper win the 2026 Men's French Open? | `roland-garros` | -0.064986 | -16.9% | 0.0499 |
| 7 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.064338 | 16.8% | 0.0500 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.048544 | -12.7% | 0.0500 |
| 9 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044878 | -11.7% | 0.0500 |
| 10 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.044845 | -11.7% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.155408 | 40.5% |
| 2 | `league-of-legends` | 0.139033 | 36.2% |
| 3 | `roland-garros` | -0.103212 | -26.9% |
| 4 | `serie-a` | 0.088711 | 23.1% |
| 5 | `nba-champion` | 0.076466 | 19.9% |
| 6 | `taylor-swift` | 0.064338 | 16.8% |
| 7 | `new-mexico-primary` | -0.048544 | -12.7% |
| 8 | `ufc` | -0.044878 | -11.7% |
| 9 | `primaries` | -0.044845 | -11.7% |
| 10 | `uefa-europa-league` | 0.043142 | 11.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will Jack Draper win the 2026 Men's French Open? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0926 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Starmer out by June 30, 2026? | -0.0098 |
| Will Petr Yan fight Pedro Munhoz next? | Will Connie Chan receive the most votes in the CA-11 primary? | 0.0059 |
| Will Connie Chan receive the most votes in the CA-11 primary? | Starmer out by June 30, 2026? | 0.0020 |
| Will AC Milan win the 2025–26 Serie A league? | Will Jack Draper win the 2026 Men's French Open? | 0.0018 |

### Correlation and Risk Structure
- category count: `18`
- avg abs category correlation: `0.0006`
- max abs category correlation: `0.0342`
- top eigenvalue share: `0.8864`
- variance ratio constrained vs baseline: `0.8180`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0003)
- [x] Constrained holdout drawdown better than baseline (+1.0884%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0342)
- [x] No single domain dominates returns (top domain share: 40.5%)

---

## Run: `week13_S1` (source branch: `cloud-runs-S1`)

### Run Context
- artifact prefix: `week13_S1`
- constrained artifact stem: `week13_S1_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5829`
- holdout steps: `1458`
- objective: `1.5`-var / `2.3`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0607 | 0.0756 | +0.0149 |
| Max drawdown | -22.6707% | -21.4885% | +1.1822% |
| Mean return | 0.00022821 | 0.00028901 | +0.00006080 |
| Volatility | 0.00712312 | 0.00815020 | +0.00102708 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0201 | 0.0813 |
| Open | Mean return | 0.00008766 | 0.00032676 |
| Open | Volatility | 0.00797581 | 0.01011669 |
| Open | Max drawdown (subset) | -14.7134% | -14.2337% |
| Closed | Sortino | 0.0719 | 0.0743 |
| Closed | Mean return | 0.00025972 | 0.00028055 |
| Closed | Volatility | 0.00691716 | 0.00764018 |
| Closed | Max drawdown (subset) | -20.8604% | -19.5117% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0507`
- baseline max drawdown (full): `-22.6707%`

### Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 33.0% of total return
**Biggest domain:** `nba-champion` — 42.2% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.138857 | 33.0% | 0.0498 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.088946 | 21.1% | 0.0500 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.069612 | 16.5% | 0.0505 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.066373 | 15.8% | 0.0500 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066094 | 15.7% | 0.0500 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.044396 | 10.5% | 0.0503 |
| 7 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.043158 | 10.2% | 0.0503 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.042090 | -10.0% | 0.0497 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041213 | -9.8% | 0.0497 |
| 10 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.039106 | 9.3% | 0.0500 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `nba-champion` | 0.177963 | 42.2% |
| 2 | `colombia-election` | 0.155040 | 36.8% |
| 3 | `serie-a` | 0.069612 | 16.5% |
| 4 | `airdrops` | 0.044396 | 10.5% |
| 5 | `uefa-europa-league` | 0.043158 | 10.2% |
| 6 | `roland-garros` | -0.038179 | -9.1% |
| 7 | `primaries` | -0.035069 | -8.3% |
| 8 | `awards` | 0.033077 | 7.8% |
| 9 | `hockey` | -0.030481 | -7.2% |
| 10 | `ligue-1` | 0.025160 | 6.0% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3929 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1686 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |
| Will the Portland Trail Blazers win the 2026 NBA Finals? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0936 |
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0416 |

### Correlation and Risk Structure
- category count: `15`
- avg abs category correlation: `0.0024`
- max abs category correlation: `0.0613`
- top eigenvalue share: `0.8198`
- variance ratio constrained vs baseline: `0.6922`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0149)
- [x] Constrained holdout drawdown better than baseline (+1.1822%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 42.2%)

---

## Run: `week12_Q` (source branch: `cloud-runs-Q5`)

### Run Context
- artifact prefix: `week12_Q`
- constrained artifact stem: `week12_Q_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6264`
- holdout steps: `1566`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0751 | 0.0808 | +0.0057 |
| Max drawdown | -35.0992% | -32.3445% | +2.7548% |
| Mean return | 0.00027988 | 0.00029933 | +0.00001945 |
| Volatility | 0.00921683 | 0.00922723 | +0.00001040 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `16.9%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0295 | 0.0351 |
| Open | Mean return | 0.00008447 | 0.00011493 |
| Open | Volatility | 0.00594989 | 0.00731658 |
| Open | Max drawdown (subset) | -4.6859% | -5.0436% |
| Closed | Sortino | 0.0824 | 0.0890 |
| Closed | Mean return | 0.00031969 | 0.00033689 |
| Closed | Volatility | 0.00974850 | 0.00956933 |
| Closed | Max drawdown (subset) | -32.8653% | -30.1273% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0356`
- baseline max drawdown (full): `-35.0992%`

### Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 42.7% of total return
**Biggest domain:** `best-of-2025` — 42.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.200155 | 42.7% | 0.0500 |
| 2 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.090422 | 19.3% | 0.0506 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.085220 | 18.2% | 0.0485 |
| 4 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.075996 | 16.2% | 0.0503 |
| 5 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.065505 | 14.0% | 0.0486 |
| 6 | Taylor Swift pregnant before marriage? | `taylor-swift` | 0.061985 | 13.2% | 0.0503 |
| 7 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.056711 | 12.1% | 0.0486 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.051520 | -11.0% | 0.0503 |
| 9 | Will Petr Yan fight Pedro Munhoz next? | `ufc` | -0.044805 | -9.6% | 0.0502 |
| 10 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.043362 | 9.3% | 0.0506 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.200155 | 42.7% |
| 2 | `colombia-election` | 0.141931 | 30.3% |
| 3 | `serie-a` | 0.090422 | 19.3% |
| 4 | `nba-champion` | 0.075996 | 16.2% |
| 5 | `taylor-swift` | 0.061985 | 13.2% |
| 6 | `new-mexico-primary` | -0.051520 | -11.0% |
| 7 | `ufc` | -0.044805 | -9.6% |
| 8 | `airdrops` | 0.043362 | 9.3% |
| 9 | `primaries` | -0.042036 | -9.0% |
| 10 | `uefa-europa-league` | 0.040708 | 8.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0971 |
| Will Kash Patel leave the Trump administration before 2027? | SCOTUS accepts sports event contract case by December 31, 2026? | 0.0353 |
| Will Taylor Fritz win the 2026 Men's French Open? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0338 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | SCOTUS accepts sports event contract case by December 31, 2026? | -0.0105 |
| Will James rank #1 among boy names on the SSA’s official list for 2025? | Will Kash Patel leave the Trump administration before 2027? | -0.0087 |

### Correlation and Risk Structure
- category count: `18`
- avg abs category correlation: `0.0012`
- max abs category correlation: `0.0473`
- top eigenvalue share: `0.3440`
- variance ratio constrained vs baseline: `0.9269`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0057)
- [x] Constrained holdout drawdown better than baseline (+2.7548%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0473)
- [x] No single domain dominates returns (top domain share: 42.7%)

---

## Run: `week13_S4_seed3` (source branch: `cloud-runs-S4`)

### Run Context
- artifact prefix: `week13_S4_seed3`
- constrained artifact stem: `week13_S4_seed3_macro_both`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5829`
- holdout steps: `1458`
- objective: `1.8`-var / `2.2`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0607 | 0.0763 | +0.0156 |
| Max drawdown | -22.6707% | -21.5841% | +1.0866% |
| Mean return | 0.00022821 | 0.00028994 | +0.00006173 |
| Volatility | 0.00712312 | 0.00814067 | +0.00101755 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `18.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0201 | 0.0859 |
| Open | Mean return | 0.00008766 | 0.00034134 |
| Open | Volatility | 0.00797581 | 0.01018735 |
| Open | Max drawdown (subset) | -14.7134% | -14.2422% |
| Closed | Sortino | 0.0719 | 0.0740 |
| Closed | Mean return | 0.00025972 | 0.00027842 |
| Closed | Volatility | 0.00691716 | 0.00760660 |
| Closed | Max drawdown (subset) | -20.8604% | -19.5489% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0507`
- baseline max drawdown (full): `-22.6707%`

### Attribution — What Drove Returns

**Biggest single market:** Will the Phoenix Suns win the 2026 NBA Finals? (`nba-champion`) — 33.5% of total return
**Biggest domain:** `nba-champion` — 42.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will the Phoenix Suns win the 2026 NBA Finals? | `nba-champion` | 0.141591 | 33.5% | 0.0498 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.087563 | 20.7% | 0.0499 |
| 3 | Will AC Milan win the 2025–26 Serie A league? | `serie-a` | 0.069342 | 16.4% | 0.0501 |
| 4 | Will Rennes win the 2025–26 French Ligue 1? | `ligue-1` | 0.067352 | 15.9% | 0.0499 |
| 5 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.066702 | 15.8% | 0.0499 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.043646 | 10.3% | 0.0501 |
| 7 | Will Nott'm Forest win the 2025-26 UEFA Europa League? | `uefa-europa-league` | 0.042857 | 10.1% | 0.0501 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.042109 | -10.0% | 0.0499 |
| 9 | Will Marseille win the 2025–26 French Ligue 1? | `ligue-1` | -0.041298 | -9.8% | 0.0499 |
| 10 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.039846 | 9.4% | 0.0499 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `nba-champion` | 0.181437 | 42.9% |
| 2 | `colombia-election` | 0.154265 | 36.5% |
| 3 | `serie-a` | 0.069342 | 16.4% |
| 4 | `airdrops` | 0.043646 | 10.3% |
| 5 | `uefa-europa-league` | 0.042857 | 10.1% |
| 6 | `roland-garros` | -0.038201 | -9.0% |
| 7 | `primaries` | -0.035531 | -8.4% |
| 8 | `awards` | 0.033172 | 7.8% |
| 9 | `hockey` | -0.030213 | -7.1% |
| 10 | `ligue-1` | 0.026054 | 6.2% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will AC Milan win the 2025–26 Serie A league? | Will the Portland Trail Blazers win the 2026 NBA Finals? | 0.3929 |
| Will Real Madrid win the 2025–26 La Liga? | Will Israel strike 11 countries in 2026? | 0.1686 |
| MegaETH market cap (FDV) >$2B one day after launch? | Will the Phoenix Suns win the NBA Western Conference Finals? | 0.0998 |
| Will the Portland Trail Blazers win the 2026 NBA Finals? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0936 |
| Will AC Milan win the 2025–26 Serie A league? | Will Nikola Jokic win the 2025–2026 NBA MVP? | -0.0416 |

### Correlation and Risk Structure
- category count: `15`
- avg abs category correlation: `0.0024`
- max abs category correlation: `0.0613`
- top eigenvalue share: `0.8198`
- variance ratio constrained vs baseline: `0.6931`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0156)
- [x] Constrained holdout drawdown better than baseline (+1.0866%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0613)
- [x] No single domain dominates returns (top domain share: 42.9%)

---

## Run: `week11_L` (source branch: `cloud-runs-L4`)

### Run Context
- artifact prefix: `week11_L`
- constrained artifact stem: `week11_L_macro_both`
- min history days used after backoff: `24.0`
- market count: `25`
- tuning steps: `6724`
- holdout steps: `1681`
- objective: `1.7`-var / `2.9`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1123 | 0.0600 | -0.0523 |
| Max drawdown | -22.3789% | -24.8857% | -2.5068% |
| Mean return | 0.00038644 | 0.00019040 | -0.00019604 |
| Volatility | 0.01189919 | 0.00984945 | -0.00204974 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `17.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.3781 | 0.3633 |
| Open | Mean return | 0.00113242 | 0.00109855 |
| Open | Volatility | 0.01982562 | 0.01974648 |
| Open | Max drawdown (subset) | -10.6618% | -9.9417% |
| Closed | Sortino | 0.0640 | -0.0016 |
| Closed | Mean return | 0.00022570 | -0.00000528 |
| Closed | Volatility | 0.00934135 | 0.00580353 |
| Closed | Max drawdown (subset) | -21.4939% | -24.2960% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0847`
- baseline max drawdown (full): `-22.3789%`

### Attribution — What Drove Returns

**Biggest single market:** Starmer out by April 30, 2026? (`grooming-gangs`) — 121.6% of total return
**Biggest domain:** `grooming-gangs` — 121.6% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Starmer out by April 30, 2026? | `grooming-gangs` | 0.389308 | 121.6% | 0.0354 |
| 2 | Will Ninjas in Pyjamas win the LPL 2026 season? | `league-of-legends` | 0.108790 | 34.0% | 0.0418 |
| 3 | Will Zachary Shrewsbury be the Democratic nominee for Senate in West Virginia? | `west-virginia-primary` | 0.083050 | 25.9% | 0.0417 |
| 4 | Will Jack Draper win the 2026 Men's French Open? | `roland-garros` | -0.056022 | -17.5% | 0.0370 |
| 5 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | `basketball` | -0.049808 | -15.6% | 0.0338 |
| 6 | Will there be fewer than 5 earthquakes of magnitude 7.0 or higher worldwide in 2026? | `natural-disasters` | -0.047372 | -14.8% | 0.0391 |
| 7 | Will the Portland Trail Blazers win the 2026 NBA Finals? | `nba-champion` | 0.041385 | 12.9% | 0.0403 |
| 8 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | `new-mexico-primary` | -0.039208 | -12.3% | 0.0415 |
| 9 | Will Connie Chan receive the most votes in the CA-11 primary? | `primaries` | -0.037557 | -11.7% | 0.0404 |
| 10 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.037468 | -11.7% | 0.0365 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `grooming-gangs` | 0.389308 | 121.6% |
| 2 | `league-of-legends` | 0.108790 | 34.0% |
| 3 | `roland-garros` | -0.083575 | -26.1% |
| 4 | `west-virginia-primary` | 0.083050 | 25.9% |
| 5 | `economic-policy` | -0.063605 | -19.9% |
| 6 | `basketball` | -0.049808 | -15.6% |
| 7 | `natural-disasters` | -0.047372 | -14.8% |
| 8 | `nba-champion` | 0.041385 | 12.9% |
| 9 | `new-mexico-primary` | -0.039208 | -12.3% |
| 10 | `primaries` | -0.037557 | -11.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | -0.2116 |
| Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | Taylor Swift pregnant before marriage? | -0.1236 |
| Will Connie Chan receive the most votes in the CA-11 primary? | USD.AI FDV above $300M one day after launch? | 0.0758 |
| Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? | Taylor Swift pregnant before marriage? | -0.0548 |
| Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election? | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | 0.0230 |

### Correlation and Risk Structure
- category count: `23`
- avg abs category correlation: `0.0010`
- max abs category correlation: `0.0804`
- top eigenvalue share: `0.7010`
- variance ratio constrained vs baseline: `0.7771`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0523)
- [ ] Constrained holdout drawdown better than baseline (-2.5068%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0804)
- [ ] No single domain dominates returns (top domain share: 121.6%)

---

## Run: `week9_B` (source branch: `cloud-runs-B2`)

### Run Context
- artifact prefix: `week9_B`
- constrained artifact stem: `week9_B`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `7691`
- holdout steps: `1923`
- objective: `1.0`-var / `2.0`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0997 | 0.0997 | +0.0000 |
| Max drawdown | -6.0264% | -6.0264% | +0.0000% |
| Mean return | 0.00014660 | 0.00014660 | -0.00000000 |
| Volatility | 0.00465710 | 0.00465710 | -0.00000000 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `14.5%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.2066 | 0.2066 |
| Open | Mean return | 0.00032272 | 0.00032272 |
| Open | Volatility | 0.00554201 | 0.00554201 |
| Open | Max drawdown (subset) | -3.7338% | -3.7338% |
| Closed | Sortino | 0.0803 | 0.0803 |
| Closed | Mean return | 0.00011683 | 0.00011683 |
| Closed | Volatility | 0.00448968 | 0.00448968 |
| Closed | Max drawdown (subset) | -6.2968% | -6.2968% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0430`
- baseline max drawdown (full): `-9.5967%`

### Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 47.1% of total return
**Biggest domain:** `best-of-2025` — 47.1% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.132736 | 47.1% | 0.0250 |
| 2 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.084806 | 30.1% | 0.0250 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033333 | 11.8% | 0.0250 |
| 4 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.023806 | 8.4% | 0.0250 |
| 5 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.018366 | -6.5% | 0.0250 |
| 6 | Will Claude 5 be released by April 30, 2026? | `claude-5` | 0.017920 | 6.4% | 0.0250 |
| 7 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.015015 | 5.3% | 0.0250 |
| 8 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.010173 | 3.6% | 0.0250 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.010000 | -3.5% | 0.0250 |
| 10 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.010000 | -3.5% | 0.0250 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.132736 | 47.1% |
| 2 | `felix` | 0.084806 | 30.1% |
| 3 | `colombia-election` | 0.033333 | 11.8% |
| 4 | `airdrops` | 0.023806 | 8.4% |
| 5 | `economic-policy` | -0.018366 | -6.5% |
| 6 | `claude-5` | 0.017920 | 6.4% |
| 7 | `colorado-midterm` | 0.015015 | 5.3% |
| 8 | `comex-silver-futures` | 0.010173 | 3.6% |
| 9 | `2026-fifa-world-cup` | -0.010000 | -3.5% |
| 10 | `california-midterm` | -0.010000 | -3.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8457 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7529 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7341 |
| Felix Protocol FDV above $300M one day after launch? | Will 2026 be the third-hottest year on record? | 0.1477 |
| Will the Republicans win the Colorado Senate race in 2026? | Will 2026 be the third-hottest year on record? | 0.1322 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0034`
- max abs category correlation: `0.2653`
- top eigenvalue share: `0.4675`
- variance ratio constrained vs baseline: `1.0000`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (+0.0000)
- [ ] Constrained holdout drawdown better than baseline (+0.0000%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2653)
- [x] No single domain dominates returns (top domain share: 47.1%)

---

## Run: `week8_C` (source branch: `cloud-runs-C`)

### Run Context
- artifact prefix: `week8_C`
- constrained artifact stem: `week8_C_macro_explicit`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `7868`
- holdout steps: `1968`
- objective: `1.0`-var / `2.3`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0812 | 0.0215 | -0.0597 |
| Max drawdown | -9.1099% | -9.5640% | -0.4541% |
| Mean return | 0.00015022 | 0.00002890 | -0.00012131 |
| Volatility | 0.00479357 | 0.00344995 | -0.00134362 |

### Holdout — US equity session vs closed (exogenous mask)

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

### Full-Series Baseline Reference
- baseline sortino (full): `0.0699`
- baseline max drawdown (full): `-12.8811%`

### Attribution — What Drove Returns

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

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.2653`
- top eigenvalue share: `0.8258`
- variance ratio constrained vs baseline: `0.9551`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0597)
- [ ] Constrained holdout drawdown better than baseline (-0.4541%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2653)
- [ ] No single domain dominates returns (top domain share: 171.2%)

---

## Run: `week8_A` (source branch: `cloud-runs-A`)

### Run Context
- artifact prefix: `week8_A`
- constrained artifact stem: `week8_A`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8339`
- holdout steps: `2085`
- objective: `0.7`-var / `2.2`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0642 | 0.0539 | -0.0103 |
| Max drawdown | -8.9604% | -9.1526% | -0.1922% |
| Mean return | 0.00010922 | 0.00008162 | -0.00002761 |
| Volatility | 0.00436554 | 0.00357510 | -0.00079044 |

### Holdout — US equity session vs closed (exogenous mask)

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

### Full-Series Baseline Reference
- baseline sortino (full): `0.0679`
- baseline max drawdown (full): `-10.2520%`

### Attribution — What Drove Returns

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

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0017`
- max abs category correlation: `0.1132`
- top eigenvalue share: `0.8341`
- variance ratio constrained vs baseline: `0.6718`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0103)
- [ ] Constrained holdout drawdown better than baseline (-0.1922%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.1132)
- [ ] No single domain dominates returns (top domain share: 100.8%)

---

## Run: `week9_C` (source branch: `cloud-runs-C2`)

### Run Context
- artifact prefix: `week9_C`
- constrained artifact stem: `week9_C_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8619`
- holdout steps: `2155`
- objective: `1.9`-var / `2.0`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1240 | 0.1280 | +0.0040 |
| Max drawdown | -8.1956% | -8.3810% | -0.1854% |
| Mean return | 0.00019803 | 0.00016291 | -0.00003511 |
| Volatility | 0.00507946 | 0.00528384 | +0.00020438 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `14.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0667 | 0.1759 |
| Open | Mean return | 0.00014238 | 0.00015724 |
| Open | Volatility | 0.00548862 | 0.00392487 |
| Open | Max drawdown (subset) | -6.8498% | -2.6505% |
| Closed | Sortino | 0.1398 | 0.1235 |
| Closed | Mean return | 0.00020759 | 0.00016389 |
| Closed | Volatility | 0.00500573 | 0.00548356 |
| Closed | Max drawdown (subset) | -7.2057% | -6.7245% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0427`
- baseline max drawdown (full): `-8.1956%`

### Attribution — What Drove Returns

**Biggest single market:** Will James rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — 48.3% of total return
**Biggest domain:** `best-of-2025` — 48.3% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.169407 | 48.3% | 0.0214 |
| 2 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.125484 | 35.7% | 0.0170 |
| 3 | Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? | `basketball` | 0.078185 | 22.3% | 0.0176 |
| 4 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.049883 | 14.2% | 0.0308 |
| 5 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.047372 | -13.5% | 0.0125 |
| 6 | MegaETH market cap (FDV) >$2B one day after launch? | `airdrops` | 0.027973 | 8.0% | 0.0265 |
| 7 | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | `economic-policy` | -0.018186 | -5.2% | 0.0205 |
| 8 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.012663 | -3.6% | 0.0260 |
| 9 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.009700 | 2.8% | 0.0160 |
| 10 | Will Bernie endorse Antonio Delgado for NY-Gov by Nov 2 2026 ET? | `bernie-sanders` | -0.009571 | -2.7% | 0.0231 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `best-of-2025` | 0.169407 | 48.3% |
| 2 | `felix` | 0.125484 | 35.7% |
| 3 | `basketball` | 0.078185 | 22.3% |
| 4 | `colombia-election` | 0.049883 | 14.2% |
| 5 | `claude-5` | -0.047372 | -13.5% |
| 6 | `airdrops` | 0.027973 | 8.0% |
| 7 | `economic-policy` | -0.018186 | -5.2% |
| 8 | `california-midterm` | -0.012663 | -3.6% |
| 9 | `comex-silver-futures` | 0.009700 | 2.8% |
| 10 | `bernie-sanders` | -0.009571 | -2.7% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8669 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.0981 |
| Will Mitch Johnson win the 2025–2026 NBA Coach of the Year? | Will Bitcoin outperform Gold in 2026? | 0.0320 |
| Will Silver (SI) hit (HIGH) $250 by end of June? | Will Bernie endorse Antonio Delgado for NY-Gov by Nov 2 2026 ET? | 0.0132 |
| Will Claude 5 be released by April 30, 2026? | Will Gold (GC) hit (HIGH) $6,500 by end of June? | 0.0103 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0029`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.4246`
- variance ratio constrained vs baseline: `0.6031`

### Interpretation Checklist
- [x] Constrained holdout Sortino beats baseline (+0.0040)
- [ ] Constrained holdout drawdown better than baseline (-0.1854%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [x] No single domain dominates returns (top domain share: 48.3%)

---

## Run: `week9_A` (source branch: `cloud-runs-A2`)

### Run Context
- artifact prefix: `week9_A`
- constrained artifact stem: `week9_A`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8382`
- holdout steps: `2096`
- objective: `1.1`-var / `2.3`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0387 | -0.0209 | -0.0597 |
| Max drawdown | -8.0707% | -13.6616% | -5.5910% |
| Mean return | 0.00005719 | -0.00002817 | -0.00008536 |
| Volatility | 0.00380155 | 0.00268330 | -0.00111825 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.1%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.1183 | 0.0134 |
| Open | Mean return | 0.00018203 | 0.00001500 |
| Open | Volatility | 0.00516096 | 0.00383377 |
| Open | Max drawdown (subset) | -3.9132% | -4.8751% |
| Closed | Sortino | 0.0238 | -0.0260 |
| Closed | Mean return | 0.00003495 | -0.00003586 |
| Closed | Volatility | 0.00350392 | 0.00242150 |
| Closed | Max drawdown (subset) | -6.5584% | -12.8567% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0565`
- baseline max drawdown (full): `-9.2357%`

### Attribution — What Drove Returns

**Biggest single market:** Will Daniel Quintero win the 2026 Colombian presidential election? (`colombia-election`) — -84.1% of total return
**Biggest domain:** `colombia-election` — -84.1% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.049656 | -84.1% | 0.0282 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.042481 | 72.0% | 0.0137 |
| 3 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.039861 | -67.5% | 0.0180 |
| 4 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.026140 | -44.3% | 0.0233 |
| 5 | Will the number of Republican Senate members who retire in 2026 be exactly 5? | `congress` | -0.021611 | 36.6% | 0.0170 |
| 6 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | -0.019790 | 33.5% | 0.0168 |
| 7 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.016929 | 28.7% | 0.0164 |
| 8 | Epstein client list released by June 30? | `epstein` | -0.016512 | 28.0% | 0.0173 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.016127 | 27.3% | 0.0290 |
| 10 | Will Hyperliquid dip to $8 by December 31, 2026? | `crypto-prices` | -0.011973 | 20.3% | 0.0229 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `colombia-election` | 0.049656 | -84.1% |
| 2 | `claude-5` | -0.042481 | 72.0% |
| 3 | `felix` | 0.039861 | -67.5% |
| 4 | `fifa-world-cup` | 0.026140 | -44.3% |
| 5 | `congress` | -0.021611 | 36.6% |
| 6 | `best-of-2025` | -0.019790 | 33.5% |
| 7 | `economic-policy` | -0.016929 | 28.7% |
| 8 | `epstein` | -0.016512 | 28.0% |
| 9 | `2026-fifa-world-cup` | -0.016127 | 27.3% |
| 10 | `crypto-prices` | -0.011973 | 20.3% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7555 |
| Felix Protocol FDV above $300M one day after launch? | Will 2026 be the third-hottest year on record? | 0.1911 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will the number of Republican Senate members who retire in 2026 be exactly 5? | -0.1032 |
| Will Daniel Quintero win the 2026 Colombian presidential election? | Will Claude 5 be released by April 30, 2026? | 0.0981 |
| Will 2026 be the third-hottest year on record? | Opensea FDV above $2B one day after launch? | 0.0911 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0024`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8596`
- variance ratio constrained vs baseline: `0.4760`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0597)
- [ ] Constrained holdout drawdown better than baseline (-5.5910%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [x] No single domain dominates returns (top domain share: -84.1%)

---

## Run: `week8_B` (source branch: `cloud-runs-B`)

### Run Context
- artifact prefix: `week8_B`
- constrained artifact stem: `week8_B`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8340`
- holdout steps: `2085`
- objective: `1.0`-var / `2.3`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0405 | 0.0063 | -0.0342 |
| Max drawdown | -9.0698% | -13.7333% | -4.6635% |
| Mean return | 0.00006136 | 0.00001079 | -0.00005057 |
| Volatility | 0.00378400 | 0.00371290 | -0.00007110 |

### Holdout — US equity session vs closed (exogenous mask)

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

### Full-Series Baseline Reference
- baseline sortino (full): `0.0662`
- baseline max drawdown (full): `-10.8395%`

### Attribution — What Drove Returns

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

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0023`
- max abs category correlation: `0.2661`
- top eigenvalue share: `0.8483`
- variance ratio constrained vs baseline: `0.9590`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0342)
- [ ] Constrained holdout drawdown better than baseline (-4.6635%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2661)
- [x] No single domain dominates returns (top domain share: -290.3%)

---

## Run: `week9_F` (source branch: `cloud-runs-F2`)

### Run Context
- artifact prefix: `week9_F`
- constrained artifact stem: `week9_F`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8447`
- holdout steps: `2112`
- objective: `1.0`-var / `2.0`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0571 | 0.0513 | -0.0058 |
| Max drawdown | -9.4627% | -9.2857% | +0.1770% |
| Mean return | 0.00007693 | 0.00006739 | -0.00000954 |
| Volatility | 0.00357437 | 0.00343809 | -0.00013628 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.7%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0242 | 0.0057 |
| Open | Mean return | 0.00003608 | 0.00000824 |
| Open | Volatility | 0.00403716 | 0.00366953 |
| Open | Max drawdown (subset) | -6.2110% | -6.2359% |
| Closed | Sortino | 0.0641 | 0.0608 |
| Closed | Mean return | 0.00008452 | 0.00007838 |
| Closed | Volatility | 0.00348153 | 0.00339323 |
| Closed | Max drawdown (subset) | -6.6919% | -6.4899% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0637`
- baseline max drawdown (full): `-10.7215%`

### Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 54.9% of total return
**Biggest domain:** `felix` — 54.9% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.078141 | 54.9% | 0.0240 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.049766 | 35.0% | 0.0262 |
| 3 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.017799 | -12.5% | 0.0238 |
| 4 | Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.014324 | 10.1% | 0.0239 |
| 5 | Will Silver (SI) hit (HIGH) $250 by end of June? | `comex-silver-futures` | 0.012098 | 8.5% | 0.0243 |
| 6 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.011834 | -8.3% | 0.0246 |
| 7 | Jeffrey Epstein confirmed to be alive before 2027? | `epstein` | -0.010236 | -7.2% | 0.0245 |
| 8 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.009526 | -6.7% | 0.0263 |
| 9 | Will Rick Caruso win the California Governor Election in 2026? | `california-midterm` | -0.009510 | -6.7% | 0.0255 |
| 10 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.008468 | 5.9% | 0.0241 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.078141 | 54.9% |
| 2 | `colombia-election` | 0.049766 | 35.0% |
| 3 | `economic-policy` | -0.017799 | -12.5% |
| 4 | `best-of-2025` | 0.014324 | 10.1% |
| 5 | `comex-silver-futures` | 0.012098 | 8.5% |
| 6 | `congress` | -0.011834 | -8.3% |
| 7 | `epstein` | -0.010236 | -7.2% |
| 8 | `2026-fifa-world-cup` | -0.009526 | -6.7% |
| 9 | `california-midterm` | -0.009510 | -6.7% |
| 10 | `colorado-midterm` | 0.008468 | 5.9% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8670 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7796 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7737 |
| Will Theodore rank #1 among boy names on the SSA’s official list for 2025? | Will Trump acquire Greenland before 2027? | 0.1760 |
| Will Trump acquire Greenland before 2027? | Extended FDV above $300M one day after launch? | 0.0941 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0028`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8384`
- variance ratio constrained vs baseline: `0.9241`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0058)
- [x] Constrained holdout drawdown better than baseline (+0.1770%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 54.9%)

---

## Run: `week8_D` (source branch: `cloud-runs-D`)

### Run Context
- artifact prefix: `week8_D`
- constrained artifact stem: `week8_D_macro_both`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8678`
- holdout steps: `2170`
- objective: `1.6`-var / `1.8`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0464 | 0.0258 | -0.0206 |
| Max drawdown | -7.9225% | -8.1648% | -0.2423% |
| Mean return | 0.00006175 | 0.00004021 | -0.00002154 |
| Volatility | 0.00349382 | 0.00337768 | -0.00011613 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.3%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0394 | -0.0974 |
| Open | Mean return | 0.00005626 | -0.00014508 |
| Open | Volatility | 0.00399671 | 0.00160941 |
| Open | Max drawdown (subset) | -4.9693% | -5.5903% |
| Closed | Sortino | 0.0478 | 0.0469 |
| Closed | Mean return | 0.00006275 | 0.00007368 |
| Closed | Volatility | 0.00339504 | 0.00360477 |
| Closed | Max drawdown (subset) | -7.8388% | -5.8697% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0629`
- baseline max drawdown (full): `-9.4604%`

### Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 101.0% of total return
**Biggest domain:** `felix` — 101.0% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.088156 | 101.0% | 0.0252 |
| 2 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.046818 | -53.7% | 0.0250 |
| 3 | Will Daniel Quintero win the 2026 Colombian presidential election? | `colombia-election` | 0.033129 | 38.0% | 0.0248 |
| 4 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.027925 | -32.0% | 0.0247 |
| 5 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.020669 | -23.7% | 0.0248 |
| 6 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.019961 | 22.9% | 0.0250 |
| 7 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.017521 | 20.1% | 0.0245 |
| 8 | Will Saudi Aramco be the largest company in the world by market cap on June 30? | `big-tech` | 0.010750 | 12.3% | 0.0252 |
| 9 | MegaETH market cap (FDV) >$3B one day after launch? | `airdrops` | 0.010714 | 12.3% | 0.0254 |
| 10 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.009995 | -11.5% | 0.0246 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.088156 | 101.0% |
| 2 | `claude-5` | -0.046818 | -53.7% |
| 3 | `colombia-election` | 0.033129 | 38.0% |
| 4 | `economic-policy` | -0.027925 | -32.0% |
| 5 | `congress` | -0.020669 | -23.7% |
| 6 | `colorado-midterm` | 0.019961 | 22.9% |
| 7 | `fifa-world-cup` | 0.017521 | 20.1% |
| 8 | `big-tech` | 0.010750 | 12.3% |
| 9 | `airdrops` | 0.010714 | 12.3% |
| 10 | `2026-fifa-world-cup` | -0.009995 | -11.5% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Will the Republicans win the Colorado Senate race in 2026? | 0.8594 |
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7505 |
| Will the Republicans win the Colorado Senate race in 2026? | Opensea FDV above $2B one day after launch? | 0.7484 |
| Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | Opensea FDV above $2B one day after launch? | -0.3013 |
| Will the Republicans win the Colorado Senate race in 2026? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.2767 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0029`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.8418`
- variance ratio constrained vs baseline: `1.0176`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0206)
- [ ] Constrained holdout drawdown better than baseline (-0.2423%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 101.0%)

---

## Run: `week9_D` (source branch: `cloud-runs-D2`)

### Run Context
- artifact prefix: `week9_D`
- constrained artifact stem: `week9_D_macro_explicit`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `8628`
- holdout steps: `2157`
- objective: `1.9`-var / `2.4`-downside mean-downside surrogate

### Holdout Performance Comparison

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.0962 | 0.0555 | -0.0407 |
| Max drawdown | -8.7653% | -13.5571% | -4.7919% |
| Mean return | 0.00013075 | 0.00007085 | -0.00005990 |
| Volatility | 0.00429574 | 0.00384035 | -0.00045538 |

### Holdout — US equity session vs closed (exogenous mask)

Subset metrics use chronological holdout steps where `is_equity_open` is 1 (NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps (gapped timeline, not calendar-interpolated).

- Holdout steps with equity open: `15.0%`
- Holdout steps marked exog-stale: `0.0%`

| Subset | Metric | Baseline | Constrained |
|--------|--------|----------|-------------|
| Open | Sortino | 0.0589 | -0.1094 |
| Open | Mean return | 0.00008886 | -0.00013405 |
| Open | Volatility | 0.00410650 | 0.00143417 |
| Open | Max drawdown (subset) | -4.7309% | -4.8177% |
| Closed | Sortino | 0.1037 | 0.0831 |
| Closed | Mean return | 0.00013813 | 0.00010694 |
| Closed | Volatility | 0.00432816 | 0.00412005 |
| Closed | Max drawdown (subset) | -7.1228% | -9.9985% |

### Full-Series Baseline Reference
- baseline sortino (full): `0.0422`
- baseline max drawdown (full): `-8.8591%`

### Attribution — What Drove Returns

**Biggest single market:** Felix Protocol FDV above $300M one day after launch? (`felix`) — 74.7% of total return
**Biggest domain:** `felix` — 74.7% of total return

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|-------------|-------|--------|
| 1 | Felix Protocol FDV above $300M one day after launch? | `felix` | 0.114095 | 74.7% | 0.0166 |
| 2 | Will James rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 0.075298 | 49.3% | 0.0226 |
| 3 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.068018 | 44.5% | 0.0235 |
| 4 | Will Claude 5 be released by April 30, 2026? | `claude-5` | -0.047632 | -31.2% | 0.0139 |
| 5 | Epstein client list released by June 30? | `epstein` | -0.029359 | -19.2% | 0.0191 |
| 6 | Will Oceania win the 2026 FIFA World Cup? | `fifa-world-cup` | 0.025013 | 16.4% | 0.0230 |
| 7 | Will the number of Republican Senate members who retire in 2026 be less than 5? | `congress` | -0.020254 | -13.3% | 0.0167 |
| 8 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | `economic-policy` | -0.017066 | -11.2% | 0.0143 |
| 9 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | -0.014146 | -9.3% | 0.0247 |
| 10 | Will Toni Atkins win the California Governor Election in 2026? | `california-midterm` | -0.012418 | -8.1% | 0.0236 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|-------------|-------|
| 1 | `felix` | 0.114095 | 74.7% |
| 2 | `best-of-2025` | 0.075298 | 49.3% |
| 3 | `colombia-election` | 0.068018 | 44.5% |
| 4 | `claude-5` | -0.047632 | -31.2% |
| 5 | `epstein` | -0.029359 | -19.2% |
| 6 | `fifa-world-cup` | 0.025013 | 16.4% |
| 7 | `congress` | -0.020254 | -13.3% |
| 8 | `economic-policy` | -0.017066 | -11.2% |
| 9 | `2026-fifa-world-cup` | -0.014146 | -9.3% |
| 10 | `california-midterm` | -0.012418 | -8.1% |

### Top 5 Correlated Contributor Pairs

| Market A | Market B | Correlation |
|----------|----------|-------------|
| Felix Protocol FDV above $300M one day after launch? | Opensea FDV above $2B one day after launch? | 0.7796 |
| Opensea FDV above $2B one day after launch? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.3181 |
| Felix Protocol FDV above $300M one day after launch? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | -0.2884 |
| Will Claude 5 be released by April 30, 2026? | Will Tarcisio de Frietas qualify for Brazil's presidential runoff? | 0.0301 |
| Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? | MegaETH market cap (FDV) >$3B one day after launch? | 0.0186 |

### Correlation and Risk Structure
- category count: `40`
- avg abs category correlation: `0.0025`
- max abs category correlation: `0.2654`
- top eigenvalue share: `0.4510`
- variance ratio constrained vs baseline: `0.6589`

### Interpretation Checklist
- [ ] Constrained holdout Sortino beats baseline (-0.0407)
- [ ] Constrained holdout drawdown better than baseline (-4.7919%)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.2654)
- [ ] No single domain dominates returns (top domain share: 74.7%)
