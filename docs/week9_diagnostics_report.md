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
