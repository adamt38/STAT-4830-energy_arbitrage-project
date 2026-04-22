# Week 11 — Round 7 Diagnostics Report (Final Kelly Squeeze)

Round 7 closes the three caveats on the K10C headline (`kelly + dynamic copula`, gross +0.46 log-wealth and +58 pp CAGR vs the equal-weight baseline):

1. **Fees were not modeled** inside the optimizer or the reported metrics.
2. **Max drawdown blew up** to −11.7% vs the baseline's −4.5% (2.6×).
3. **Single-seed / single-config** — no statistical CI on the log-wealth delta.

The round has two arms: **three laptop post-hocs** (ranked net-of-fees; fractional-Kelly α-blend on K10C; circular-block bootstrap CI), and **two new GPU pods** on `cloud-runs-R7` that push fees and drawdown *into the optimizer* (K10E fee-aware Kelly; K10F drawdown-controlled Kelly). K10D (pre-existing, `K10C + turnover_lambda` sweep on pod `ba619f11`) continues in parallel as the no-fee control. An optional M5 run tests whether the week8 MVO Sortino whisper survives a 40-market universe.

## Run inventory

| Short tag | Pipeline | Key change vs prior runs | Branch | Status (as of this report) |
|---|---|---|---|---|
| K10A | `script/polymarket_week10_kelly_pipeline.py` | Kelly + log-wealth, **no copula** | `cloud-runs-K10A` | complete (Round 3) |
| K10B | `script/polymarket_week10_kelly_pipeline.py` | Kelly + log-wealth + dynamic copula, `λ_turn=0` | `cloud-runs-K10B` | complete (Round 3) |
| K10C | `script/polymarket_week10_kelly_pipeline.py` | Kelly + log-wealth + dynamic copula, full stack | `cloud-runs-K10C` | complete (Round 3) — **headline** |
| K10D | `script/polymarket_week10_kelly_pipeline.py` | K10C + `turnover_lambda` sweep | `cloud-runs-K10D` | running on pod `ba619f11` (Round 6) |
| **K10E** | `script/polymarket_week10_kelly_pipeline.py` (**patched R7**) | K10C + **`fee_rate`** sweep {0, 10, 50, 200 bps} × `turnover_lambda` | `cloud-runs-R7` → `cloud-runs-K10E` | pod launch pending |
| **K10F** | `script/polymarket_week10_kelly_pipeline.py` (**patched R7**) | K10C + **downside-semivariance** penalty {0, 0.5, 2, 5, 10} | `cloud-runs-R7` → `cloud-runs-K10F` | pod launch pending |
| M5 (optional) | `script/polymarket_week8_pipeline.py` | S1 best config at `--market-count-override 40` | `cloud-runs-R7` → `cloud-runs-M5` | pod launch optional |

## 1. Laptop post-hoc #1 — Net-of-fees re-ranking (Kelly runs)

**Input:** `data/round7_cache/{K10A,K10B,K10C}_{kelly_best,baseline}_timeseries.csv` (11 429 holdout steps each). **Script:** `script/posthoc_fee_ranking.py`. **Full tables:** `data/processed/round7_fee_ranking.{csv,md}`.

We subtract `fee_rate · turnover_l1` from each step's gross portfolio return before re-accumulating log-wealth, Sortino, and max drawdown. Equal-weight baseline is zero-turnover so its metrics are fee-invariant.

### Headline Δ total-log-wealth (Kelly − equal-weight baseline)

| Run | fee = 0 bps | fee = 10 bps | fee = 50 bps | fee = 200 bps |
|---|---:|---:|---:|---:|
| K10A (no copula) | **+0.1762** | −0.1383 | −1.3967 | −6.1169 |
| K10B (copula, λ_turn=0) | **+0.2903** | **+0.0246** | −1.0381 | −5.0243 |
| K10C (full stack) | **+0.4598** | −0.7645 | −5.6631 | −24.0525 |

### Per-run break-even fee (linear-extrapolation)

`break_even_bps ≈ 10 000 · (Δ log-w at fee = 0) / (n_steps · avg_turnover_per_step)`.

| Run | Δ log-w @ 0 | avg L1 turnover | **break-even fee** |
|---|---:|---:|---:|
| K10A | +0.1762 | 0.0275 | **5.60 bps** |
| K10B | +0.2903 | 0.0232 | **10.93 bps** |
| K10C | +0.4598 | 0.1071 | **3.76 bps** |

**Takeaway.** K10C's gross +0.46 log-wealth edge has a break-even of only ~3.76 bps — below any realistic Polymarket spread (conservatively 10+ bps, realistically 50–200 bps on thin markets). K10B is **~3× more fee-robust** than K10C because its turnover is ~4.6× lower while its gross edge is ~63% as large. This is the single most important finding of Round 7 and directly motivates K10E (fee-aware Kelly) — we need to let the optimizer *know* about fees so it can trade off gross edge against friction at training time, rather than cashing a gross edge that the realized fees immediately destroy.

## 2. Laptop post-hoc #2 — Fractional-Kelly α-blend on K10C

**Input:** `data/round7_cache/K10C_{kelly_best,baseline}_timeseries.csv` aligned via the script's last-N-rows convention. **Script:** `script/posthoc_alpha_blend.py`. **Full table:** `data/processed/week10_kelly_C_alpha_blend_summary.md` (+ `*_sortino.png`).

Blend definition: `r_blend(t; α) = α · r_constrained(t) + (1 − α) · r_baseline(t)`, evaluated on the 11 429 aligned holdout steps. This is a *post-hoc* operation — it cannot invent information the optimizer didn't already have, but it does trace the DD / log-wealth frontier induced by scaling down the Kelly position.

| α | Sortino | Max DD | Total log-wealth | % of α = 1 log-wealth kept |
|---:|---:|---:|---:|---:|
| 0.00 (pure baseline) | +0.0525 | −4.5185% | +0.4081 | 47% |
| 0.25 | +0.0573 | −5.8409% | +0.5295 | 61% |
| **0.50** | **+0.0586** | **−7.6082%** | **+0.6465** | **75%** |
| 0.60 (**Sortino argmax**) | **+0.0586** | −8.3910% | +0.6921 | 80% |
| 0.75 | +0.0585 | −9.6259% | +0.7593 | 87% |
| 1.00 (pure K10C) | +0.0579 | −11.7410% | +0.8678 | 100% |

**Takeaway.** The interior Sortino-argmax at **α ≈ 0.60** means K10C is over-levered on a risk-adjusted basis — a half-to-60% blend *dominates* the pure Kelly position in Sortino while cutting max DD from −11.7% to −7.6% (α = 0.5) or −8.4% (α = 0.6). This is a real, actionable result **and** a diagnosis: if the gross edge is this Sortino-efficient only at fractional leverage, the K10F downside-semivariance penalty (which achieves the same DD reduction *inside* the optimizer, so the full Kelly objective can re-allocate under the constraint) should dominate post-hoc α-blending *if it works at all*. That is the explicit success criterion for K10F.

## 3. Laptop post-hoc #3 — Bootstrap CI on log-wealth delta

**Input:** `data/round7_cache/{K10A,K10B,K10C}_{kelly_best,baseline}_timeseries.csv`. **Script:** `script/posthoc_bootstrap_ci.py` (circular-block, paired — identical block offsets for Kelly and baseline so step alignment is preserved). **Setup:** 1 000 replicates; block length 50 (~2.1 days on 60-min bars); seed 7. **Full table:** `data/processed/round7_bootstrap_ci.{csv,md}`.

| Run | n_steps | Observed Δ | Bootstrap mean | Std | 95% CI | Pr(Δ > 0) | z |
|---|---:|---:|---:|---:|---|---:|---:|
| K10A | 11 429 | +0.1762 | +0.1886 | 0.2194 | [−0.232, +0.599] | 0.807 | +0.80 |
| K10B | 11 429 | +0.2903 | +0.2990 | 0.1998 | [−0.097, +0.696] | 0.934 | +1.45 |
| K10C | 11 429 | +0.4598 | +0.4610 | 0.2410 | **[+0.005, +0.973]** | **0.975** | **+1.91** |

**Takeaway.** K10C's 95% CI **barely** excludes zero (lower bound +0.005). On a *z*-score basis (+1.91) this is marginal — between one- and two-sigma. K10A and K10B CIs straddle zero. All three CIs are **gross-of-fees**; combined with the break-even table (§1) this means even the statistically-marginal gross K10C win is underwater net of realistic fees. This is the quantitative justification for not treating K10C as a "done deal" and for doing Round 7 at all: a net-of-fees, DD-controlled, bootstrap-significant Kelly run would be a genuinely new strategy; re-reporting K10C is not.

## 4. Pod arm — K10E fee-aware Kelly (launched via `docs/cloud_runbook.md` §17.2)

**Code paths patched on `cloud-runs-R7`:**
- `src/kelly_copula_optimizer.py::_run_kelly_online_pass` — training loss now adds `fee_rate · turnover` (jointly with `turnover_lambda · turnover`), and realized return is `r_gross − fee_rate · step_turnover_l1`. `kelly_log_wealth` was split so the semivariance penalty (used by K10F) has access to the MC-sample tensor; `kelly_log_wealth` preserves the original `scalar → scalar` contract for all existing callers.
- `script/polymarket_week10_kelly_pipeline.py` — new `--fee-rate-values` / `--fee-rate-override` CLI flags, parsed via `_parse_float_csv` and propagated into `KellyExperimentConfig.fee_rates`, `run_kelly_optuna_search`, the per-trial records CSV (`kelly_experiment_grid.csv`), and the run manifest's `extra` block.

**Sweep:** `fee_rate ∈ {0, 0.0010, 0.0050, 0.0200}` (0 / 10 / 50 / 200 bps per unit L1 turnover), crossed with the default `turnover_lambdas = (0.0, 0.001, 0.01, 0.05, 0.1)`, 200 Optuna trials total.

**Status.** Pending pod launch. Results will populate the table below at fan-in time (§6 below).

| Regime | Best `λ_turn` | Best `fee_rate` | Holdout log-wealth Δ (net) | Max DD | avg L1 turnover |
|---|---:|---:|---:|---:|---:|
| fee = 0 bps | TBD | 0 | TBD | TBD | TBD |
| fee = 10 bps | TBD | 0.0010 | TBD | TBD | TBD |
| fee = 50 bps | TBD | 0.0050 | TBD | TBD | TBD |
| fee = 200 bps | TBD | 0.0200 | TBD | TBD | TBD |

**Success criterion:** ≥1 trial with `Δ log-wealth ≥ +0.10` *net* at 10 bps. Equivalent: the optimizer, when told fees exist, re-allocates enough toward low-turnover policies that the Kelly edge survives friction.

## 5. Pod arm — K10F drawdown-controlled Kelly (launched via §17.3)

**Code paths patched on `cloud-runs-R7`:**
- `src/kelly_copula_optimizer.py::_run_kelly_online_pass` — training loss adds `dd_penalty · mean(relu(−ρ_mc)²)` where `ρ_mc` is the MC-sampled portfolio return tensor from the copula log-wealth integrator. This is a downside-semivariance penalty; convex in weights, non-zero gradient everywhere `r_mc < 0`, and uses the *same* MC samples the log-wealth expectation uses (no extra cost per step).
- `script/polymarket_week10_kelly_pipeline.py` — new `--dd-penalty-values` / `--dd-penalty-override` CLI flags, parsed into `KellyExperimentConfig.dd_penalty_lambdas`, threaded to `run_kelly_optuna_search`, and added to the run manifest + diagnostics report's "Selected Hyperparameters" table.

**Sweep:** `dd_penalty ∈ {0, 0.5, 2, 5, 10}`, other levers pinned via `--lambda-turnover-override 0.0` and the K10C best config, 150 Optuna trials.

**Status.** Pending pod launch. Frontier table below will populate at fan-in.

| `dd_penalty` | Holdout log-wealth Δ | Max DD | Annualized log-growth | Sortino | Note |
|---:|---:|---:|---:|---:|---|
| 0 | TBD | TBD | TBD | TBD | reproduction of K10C |
| 0.5 | TBD | TBD | TBD | TBD |  |
| 2 | TBD | TBD | TBD | TBD |  |
| 5 | TBD | TBD | TBD | TBD |  |
| 10 | TBD | TBD | TBD | TBD |  |

**Success criterion:** ≥1 frontier point with `max_drawdown ≥ −7%` AND `Δ log-wealth ≥ +0.30`. If K10F delivers that point *and* K10E delivers a fee-aware strategy, we intersect the two (either as a post-hoc α-blend of the K10F frontier or as a combined sweep in a follow-up round).

## 6. Fan-in (after all pods finish)

See `docs/cloud_runbook.md` §17.6 for the explicit merge + cache-refresh + post-hoc-rerun commands. The headline tables in §4 and §5 above will be populated from the pod outputs by re-running `script/posthoc_fee_ranking.py --runs K10A,K10B,K10C,K10D,K10E,K10F` and `script/posthoc_bootstrap_ci.py --runs K10A,K10B,K10C,K10D,K10E,K10F` at that time. This document will then be updated in-place with a new §7 close-out verdict.

## 7. Close-out verdict (to be written at fan-in)

*Placeholder.* Fill in after K10D/E/F (and optional M5) complete. Expected structure:

- **Net-of-fees headline table** — one row per run at fees ∈ {0, 10, 50, 200} bps, Kelly Δ total-log-wealth and Δ Sortino vs baseline.
- **DD frontier** — K10F's (`dd_penalty`, max_DD, log-wealth Δ) points plus the K10C α-blend trace for comparison, in a single chart.
- **Best net-of-fees bootstrap CI** — run the circular-block bootstrap on the best K10E net-of-fees config and report the 95% CI + Pr(Δ > 0). A confidence interval that excludes zero at 10 bps is the Round 7 win condition.
- **Recommendation** — single sentence: "the policy to deploy" or "no strategy survived fees; recommend shelving the Polymarket Kelly thread and pivoting."

## What Round 7 does NOT do

- Does not redo K10D's turnover sweep (already in flight on `ba619f11`).
- Does not try more MVO recipes on the 20-market universe (every Round 4–6 MVO variant tied baseline within the ±0.035 seed-noise band).
- Does not expand the market universe beyond 40 (data availability + Kelly copula MLP is O(K²) per step).
- Does not add a CVaR hard constraint inside the Kelly loss (considered but rejected: batch-level CVaR is gradient-noisy on the N_mc scales we use; semivariance is a more stable surrogate for the DD goal).
