# Project Report: Cross-Domain Portfolio Optimization on Polymarket

## Problem Statement

### What are you optimizing?
We optimize a portfolio of Polymarket prediction contracts to maximize risk-adjusted return while penalizing over-allocation to a single event domain (for example politics or crypto).

### Why does this matter?
Prediction markets are exposed to event clustering risk. A portfolio that appears diversified by the number of positions may still be concentrated in one real-world driver. Our project tests whether explicit domain-level constraints reduce tail risk versus a naive equal-weight allocation.

### How do we measure success?
Success is measured against an equal-weight baseline using:
- Sortino ratio (risk-adjusted return),
- maximum drawdown,
- domain exposure concentration.

### Data and constraints
- **Data source:** `gamma-api.polymarket.com/events` (market discovery and tags) + `clob.polymarket.com/prices-history` (token price history).
- **Optimization constraint:** penalize any domain exposure above a limit `L_k` via a differentiable quadratic penalty.
- **Project objective:** maximize `Sortino_t(R_p) - lambda * sum_k max(0, S_k - L_k)^2`.

### Risks
- API coverage may be uneven by market and domain.
- Domain mapping from tags is noisy and needs normalization.
- First-pass online optimization may be unstable and overfit.

---

## Technical Approach

### Data pipeline
We implemented a reproducible pipeline in `src/polymarket_data.py` that:
1. pulls paginated active events,
2. flattens event/market records,
3. maps event tags to coarse domains,
4. selects binary market tokens,
5. fetches token-level historical prices,
6. caches outputs in `data/raw` and `data/processed`.

### Baseline
`src/baseline.py` computes:
- equal-weight market allocation,
- portfolio return series,
- cumulative return,
- max drawdown,
- Sortino ratio,
- domain exposure shares.

### Constrained optimizer
`src/constrained_optimizer.py` implements a first OGD/SGD-style online routine in PyTorch:
- rolling-window updates,
- softmax weights over markets,
- Sortino-based objective with domain-penalty term,
- small grid search over learning rate, penalty lambda, and window length.

---

## Week 4 Prototype Results (Completed Work So Far)

Artifacts were generated through `script/polymarket_week8_pipeline.py`.

### Dataset quality snapshot
- markets retained: **19**
- history points: **12,509**
- missing-history markets: **0**
- non-monotonic token series: **0**
- duplicate timestamp-token points: **0**

### Baseline (equal-weight)
- Sortino: **0.0706**
- Max drawdown: **-14.35%**
- Mean return per step: **0.000309**
- Volatility: **0.01451**
- Category exposure now spans many categories, including: `us-presidential-election`, `world-elections`, `global-elections`, `sports`, `soccer`, `serie-a`, `la-liga`, `movies`, `jerome-powell`, `airdrops`, `politics`, `world`, and `elections`.

### Constrained first iteration (best grid point)
- learning rate: **0.05**
- lambda penalty: **10.0**
- rolling window: **48**
- Sortino: **0.0148**
- Max drawdown: **-84.18%**
- Mean return per step: **0.000729**
- Volatility: **0.03591**

### Interpretation
The constrained model currently does **not** beat baseline on risk-adjusted quality or drawdown. This is our Week 4 prototype baseline and establishes a working benchmark for the next iteration.

---

## Week 8 Iteration (Current Work)

### Week 8 achieved updates
1. Expanded from coarse domains to high-liquidity tag categories.
2. Rebuilt dataset with category-balanced selection targeting 50-100 categories.
3. Implemented category-equal baseline weighting (equal total allocation per category).
4. Regenerated artifacts with renamed Week 8 files (`week8_*` and `week8_iteration_*`).

### Week 8 latest metrics
- markets retained: **80**
- categories retained: **80**
- baseline category exposure: **equal at 1.25% per category**
- baseline Sortino: **0.0730**
- baseline max drawdown: **-4.73%**

---

## Week 11 — Round 7 (Final Kelly Squeeze) Status

Full details live in [`docs/week11_round7_diagnostics_report.md`](docs/week11_round7_diagnostics_report.md). This section is a one-page status for the rolling report.

### Headline from Round 3 that Round 7 is testing

K10C (Kelly + dynamic copula, `script/polymarket_week10_kelly_pipeline.py`) is the only pipeline result that clearly clears baseline on the Kelly objective: **+0.46 total log-wealth, +58 pp CAGR**, max DD −11.7%. Every MVO variant (Round 4–6: I4, Q5, S1, S4, S5) ties baseline within a ±0.035 seed-noise band.

### Round 7 laptop post-hocs (completed on 2026-04-19)

1. **Net-of-fees re-ranking** (`script/posthoc_fee_ranking.py`, output `data/processed/round7_fee_ranking.md`). K10C's break-even fee is only **~3.76 bps** per unit L1 turnover — its gross +0.46 log-wealth edge flips to **−0.76** at 10 bps. K10B is **3× more fee-robust** (break-even 10.93 bps) because its turnover is ~4.6× lower.
2. **Fractional-Kelly α-blend on K10C** (`script/posthoc_alpha_blend.py`, output `data/processed/week10_kelly_C_alpha_blend_summary.md`). Sortino argmax at **α ≈ 0.60**; α = 0.5 keeps ~75% of K10C's gross log-wealth gain (+0.647) while cutting max DD from −11.7% to −7.6%. This tells us K10C is over-levered on a risk-adjusted basis.
3. **Circular-block bootstrap** (`script/posthoc_bootstrap_ci.py`, output `data/processed/round7_bootstrap_ci.md`; 1 000 replicates, block = 50). K10C 95% CI `[+0.005, +0.973]`, `Pr(Δ > 0) = 0.975`, `z = +1.91` — just barely excludes zero gross. K10A and K10B CIs both straddle zero.

### Round 7 pod arms (branch `cloud-runs-R7`, launch blocks in `docs/cloud_runbook.md` §17)

- **K10E fee-aware Kelly:** `src/kelly_copula_optimizer.py::_run_kelly_online_pass` now adds `fee_rate · turnover` to the training loss and subtracts `fee_rate · step_turnover_l1` from each step's realized return. New CLI flag `--fee-rate-values`. Sweep: `fee_rate ∈ {0, 10, 50, 200 bps}` × default `turnover_lambdas`. Target: net Δ log-wealth ≥ +0.10 at 10 bps.
- **K10F drawdown-controlled Kelly:** same file adds `dd_penalty · mean(relu(−ρ_mc)²)` (downside semivariance on the MC-sample tensor) to the loss. New CLI flag `--dd-penalty-values`. Sweep: `dd_penalty ∈ {0, 0.5, 2, 5, 10}`. Target: frontier point with `max_DD ≥ −7%` AND `Δ log-wealth ≥ +0.30`.
- **M5 (optional):** `script/polymarket_week8_pipeline.py` with S1 best config at `--market-count-override 40` — tests whether the S1 Sortino whisper survives a 2× universe.

### Close-out verdict

*Pending pod fan-in.* The diagnostics report §7 will be finalized once K10D (already running on pod `ba619f11`), K10E, K10F (and optional M5) complete and are merged into `cloud-runs-R7-fanin`. The win condition for Round 7 is a net-of-fees bootstrap CI on the best K10E config that excludes zero at 10 bps with `Pr(Δ > 0) ≥ 0.95`. If that condition fails across all of K10D/E/F, Round 7 closes the Kelly thread and the project pivots.
