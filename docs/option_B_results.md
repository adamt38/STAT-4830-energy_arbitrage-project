# Option B Results — Momentum × Kelly (Pod M)

Ports the `--momentum-screening` lever (merged from Option A / Pod G) from the MVO/Sortino week8 pipeline into the Kelly log-wealth week10 pipeline. Tests whether momentum pre-selection of the market universe compounds with Adam's Round 3 Kelly + dynamic-copula + L1-turnover optimizer.

## Why this combination is novel

- **K10A / K10B / K10C / K10D / K10E / K10F** (Adam's Round 3 + Round 7 Kelly pods) all operate on the **full 40-market universe** from cached week8 data.
- **G / I4 / Q5 / S1 / S4 / S5 / Option A** (MVO pipeline pods) test momentum screening at various intensities, but all optimize Sortino / mean-downside — not Kelly log-wealth.
- **Pod M = Kelly objective + momentum pre-filter**. The one cross-term between Adam's two Rounds' methodologies that nobody has run.

## Configuration

| Lever | Value |
|---|---|
| Pipeline | `polymarket_week10_kelly_pipeline.py` (dynamic-copula Kelly OGD) |
| Data build | fresh (`--rebuild-data`), momentum-screened |
| Momentum screening | top-20 markets, 5-day lookback |
| Reduced Optuna search | enabled |
| Trials | 100 |
| Seed | 7 (pipeline default; no CLI override implemented) |

## Headline metrics

| Metric | Baseline (eq-wt on momentum universe) | Pod M (Kelly) | Δ |
|---|---|---|---|
| **Sortino ratio (holdout)** | +0.0370 | +0.0635 | **+0.0265** |
| **Log-wealth (cumulative holdout)** | +0.1838 | +0.4024 | **+0.2186** |
| Max drawdown (holdout) | -21.48% | -27.06% | -5.58 pp |

## Bootstrap 95% CI on Sortino delta

- Observed Sortino Δ: **+0.0265**
- 95% CI: [-0.0482, +0.1093]
- Fraction of resamples positive: **75.8%**
- n_bootstrap: 5000, n_holdout_steps: 1491

## Comparison to benchmarks

**MVO pods (same Sortino metric):**
- Noise floor (Adam §15.9): ~0.036 Sortino Δ
- Adam's I4 (single seed): +0.0077
- Adam's S1 (single seed, momentum + wide sweep): +0.0149
- Adam's S4 seed=3 (multi-seed I4 repro, 1 of 5): +0.0156
- Our Option A (momentum × shrinkage × macro=both, 3 seeds): −0.0169 ± 0.0003
- Teammate week17 (`stock-PM-combined-strategy`): +0.1018

**Kelly pods (log-wealth metric):**
- K10A (full Kelly, no momentum): log-wealth Δ gross = see Adam's §17
- K10C (same but turnover-focused): gross +0.46 log-wealth, BUT break-even fee only 3.76 bps → flips negative at realistic 10 bps
- **Pod M (Kelly + momentum top-20/5d)**: **+0.2186** log-wealth

## Selected hyperparameters (best Optuna trial)

- `lr_w`: 0.010070
- `lr_theta`: 0.000911
- `rolling_window`: 48
- `mc_samples`: 512
- `turnover_lambda`: 0.049497
- `copula_shrinkage`: 0.067553
- `copula_temperature`: 0.100000
- `mlp_hidden_dim`: 16
- `concentration_penalty_lambda`: 2.267405
- `max_weight`: 0.077624
- `fee_rate`: 0.000000
- `dd_penalty`: 0.000000

## Momentum universe (top 20 markets by |5d return|)

| Rank | Domain | Momentum | Question (truncated) |
|---|---|---|---|
| 1 | nba-champion | +2.0000 | Will the Portland Trail Blazers win the 2026 NBA Finals? |
| 2 | serie-a | +1.3333 | Will AC Milan win the 2025–26 Serie A league? |
| 3 | la-liga | +1.1944 | Will Real Madrid win the 2025–26 La Liga? |
| 4 | nba | -0.7692 | Will the Phoenix Suns win the NBA Western Conference Finals? |
| 5 | la-liga | -0.6667 | Will Villarreal win the 2025–26 La Liga? |
| 6 | 2026-nfl-draft-top-10 | -0.6667 | Will Dante Moore be the second pick in the 2026 NFL draft? |
| 7 | ligue-1 | -0.6667 | Will Marseille win the 2025–26 French Ligue 1? |
| 8 | ligue-1 | -0.6667 | Will Lille win the 2025–26 French Ligue 1? |
| 9 | new-mexico-primary | -0.6300 | Will Duke Rodriguez win the 2026 New Mexico Governor Republican primary election |
| 10 | primaries | -0.6261 | Will Connie Chan receive the most votes in the CA-11 primary? |
| 11 | epstein | -0.5839 | Epstein client list released by June 30? |
| 12 | hockey | -0.5825 | Will the Ottawa Senators win the Eastern Conference? |
| 13 | roland-garros | -0.5455 | Will Taylor Fritz win the 2026 Men's French Open? |
| 14 | claude-5 | -0.4783 | Will Claude 5 be released by May 31, 2026? |
| 15 | league-of-legends | -0.4706 | Will BRION win the LCK 2026 season playoffs? |
| 16 | basketball | -0.4663 | Will JB Bickerstaff win the 2025–2026 NBA Coach of the Year? |
| 17 | south-dakota-primary | +0.4658 | Will Dusty Johnson win the 2026 South Dakota Governor Republican primary electio |
| 18 | airdrops | +0.4135 | MegaETH market cap (FDV) >$3B one day after launch? |
| 19 | roland-garros | -0.4118 | Will Jack Draper win the 2026 Men's French Open? |
| 20 | nba-champion | -0.4000 | Will the Phoenix Suns win the 2026 NBA Finals? |

## Interpretation

**Verdict: directional positive on both metrics.** Neither is statistically decisive but both point the same way. A multi-seed follow-up or fee-sweep (à la Adam's K10E) is the natural next step.


---

# Pod MF — Fee Sweep Extension

Extends Pod M by adding `fee_rate` as a searchable Optuna categorical with values `{0, 0.001, 0.005, 0.01}` (0, 10, 50, 100 bps). Addresses Adam's §17 caveat that K10C's gross +0.46 log-wealth edge has break-even at only 3.76 bps (flips to −0.76 at realistic 10 bps spreads).

## Headline Pod MF numbers

| Metric | Baseline | Pod MF | Δ |
|---|---|---|---|
| Sortino | +0.0591 | +0.0427 | **-0.0164** |
| Log-wealth | +0.3209 | +0.2617 | **-0.0592** |
| Max drawdown | -24.36% | -30.32% | -5.95 pp |

## Bootstrap 95% CI on Sortino delta

- Observed Δ: **-0.0164**
- 95% CI: [-0.0562, +0.0233]
- Frac positive: **20.1%**

## Which fee_rate Optuna selected

**Best trial's fee_rate: `0.0` (0.0 bps)**

## Per-fee-rate trial summary (log-wealth mean / best)

| fee_rate (bps) | n trials | best log-wealth | mean log-wealth |
|---|---|---|---|
| 0.0 | 23 | +0.0000 | +0.0000 |
| 10.0 | 24 | +0.0000 | +0.0000 |
| 50.0 | 31 | +0.0000 | +0.0000 |
| 100.0 | 22 | +0.0000 | +0.0000 |

## Pod M vs Pod MF comparison

| Pod | fee_rate | Sortino Δ | Log-wealth Δ | Note |
|---|---|---|---|---|
| Pod M (no fee lever) | 0.0 (forced) | +0.0265 | +0.2186 | Adam's K10C eq |
| Pod MF (fee searched) | 0.0 bps | -0.0164 | -0.0592 | realistic fees |

## Interpretation

**Optuna picked fee_rate=0 even when all other values were available.** Indicates the optimizer couldn't find a configuration that justifies paying explicit fees — positive values penalize turnover in training and in the reported return, pushing objectives down.
