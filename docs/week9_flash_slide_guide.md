# Week 9 Flash Slide Guide

This guide supersedes `docs/week8_flash_slide_guide.md` for **Slides Draft 3** and later check-ins. It matches the **current** pipeline in `script/polymarket_week8_pipeline.py` and optimizer in `src/constrained_optimizer.py` (artifact prefix remains `week8_*` on disk).

---

## 1) Why this problem is interesting (unchanged core + Week 9 angle)

- Prediction markets are a real-money, real-time setting where beliefs update quickly.
- Categories (sports, elections, geopolitics, macro, etc.) create a natural cross-domain diversification problem.
- Core proposal idea: cap domain exposure while pursuing risk-adjusted returns with an **online** allocator.
- **Week 9 emphasis:** Week 8 showed that a naive setup (ratio objective, heavy uniform mixing, huge universes) can look “optimized” while still behaving like equal weight. The project story is now **methodological iteration**: smoother training objective, scaled data frequency, smaller tradable set, Bayesian tuning, and honest holdout + attribution.

---

## 2) Process overview (one slide, repo-specific)

1. **Ingest events** (`src/polymarket_data.py::fetch_active_events`): Gamma API with pagination; raw cache e.g. `data/raw/week8_events_raw.json`.
2. **Flatten + domain tagging** (`flatten_event_markets`): event→market rows, tag→domain, YES token ids for binary markets.
3. **Balanced selection** (`_select_balanced_market_rows`): liquidity-aware category sampling, per-category caps, excluded slugs (see `BuildConfig` in `script/polymarket_week8_pipeline.py`).
4. **Price history** (`fetch_price_history`): CLOB `prices-history` with configurable `**history_fidelity`** (pipeline uses **10-minute** bars for more observations per calendar day).
5. **Cached dataset** (`build_dataset`): `data/processed/week8_markets_filtered.csv`, `week8_price_history.csv`, `week8_data_quality.json`, etc.
6. **Baseline** (`src/baseline.py::run_equal_weight_baseline`): equal weight **by category/domain construction**; metrics + time series → `week8_baseline_*`.
7. **Constrained tuning** (`src/constrained_optimizer.py::run_optuna_search`): **Optuna** trials over constraints, penalties, windows, mixing, and (optionally) mean-downside weights; **walk-forward** blocks with **online** evaluation on the holdout (weights keep updating after the tuning split).
8. **Diagnostics + figures + report**: `src/covariance_diagnostics.py`; comparison figures still emitted as `figures/week8_iteration_*`; written summary `docs/week9_diagnostics_report.md` (generated at end of pipeline via `_make_week9_diagnostics_report`).

---

## 3) Core math formulas (one slide)

**Definitions** (same as Week 8 for portfolio algebra):

- $r_t \in \mathbb{R}^N$: token returns at step $t$
- $\tilde{w}_t \in \Delta^N$: portfolio weights on the simplex (after softmax + optional uniform mix)
- $\mathcal{D}$, $S_d(\tilde{w})$, $L_d$: domains, domain weight sums, domain caps

**Step return, value, drawdown, Sortino (evaluation metrics):**

$$
R_t = \tilde{w}*t^\top r_t, \qquad
V_t = \prod*{s=1}^{t}(1+R_s), \qquad
\mathrm{DD}*t = \frac{V_t}{\max*{s\le t} V_s}-1
$$

$$
\mathrm{Sortino}(R)=\frac{\mathbb{E}[R]}{\sqrt{\mathbb{E}[(\min(R,0))^2]}+\varepsilon}
$$

**Penalties and bonuses** (training includes these; same functional forms as Week 8):

$$
P_{\mathrm{domain}}(\tilde{w})=\sum_{d\in\mathcal{D}}\max(0,S_d(\tilde{w})-L_d)^2, \qquad
P_{\mathrm{conc}}(\tilde{w})=\sum_i \max(0,w_i-w_{\max})^2
$$

$$
P_{\mathrm{cov}}(\tilde{w})=\tilde{w}^\top \Sigma \tilde{w}
\quad\text{(sample cov on the rolling window, with diagonal shrinkage)}
$$

$$
B_{\mathrm{ent}}(\tilde{w})=-\sum_i w_i\log(w_i+\varepsilon)
$$

**Uniform mixing** (diversification floor; $\alpha$ is `uniform_mix`):

$$
\tilde{w}_t=(1-\alpha)\mathrm{softmax}(z_t)+\alphau
$$

**Training objective (default): mean–downside surrogate** — implemented as `_mean_downside_objective` on **window portfolio returns** $R_{\mathrm{window}}$:

$$
J_{\mathrm{md}}(R)=\mathbb{E}[R]-\alpha_v\mathrm{Var}(R)-\beta_d\mathbb{E}\big[\max(0,-R)^2\big]
$$

**Full per-step training objective** (maximize; code minimizes $-\texttt{obj}$):

$$
\max_{\tilde{w}} \Big[
J_{\mathrm{md}}(R_{\mathrm{window}}(\tilde{w}))
-\lambdaP_{\mathrm{domain}}(\tilde{w})
-\lambda_cP_{\mathrm{conc}}(\tilde{w})
-\lambda_{\Sigma}P_{\mathrm{cov}}(\tilde{w})
+\lambda_eB_{\mathrm{ent}}(\tilde{w})
\Big]
$$

**Legacy mode:** `ExperimentConfig.objective == "sortino"` replaces $J_{\mathrm{md}}$ with $\mathrm{Sortino}(R_{\mathrm{window}})$ (ratio form); default is `**mean_downside`**.

---

## 4) Objective term-by-term: code map

---

## 5) What changed since Week 8 (dedicate one slide)


| Topic              | Week 8 flash guide                     | Week 9 (current)                                                                                                             |
| ------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Training target    | Sortino ratio in-window                | **Mean–downside** surrogate; Sortino for **reporting**                                                                       |
| Uniform mix        | Often high (e.g. 0.88 in old snapshot) | Search over **low** floors, including 0 (`uniform_mixes` in pipeline)                                                        |
| Inner optimization | Single-step narrative                  | `**steps_per_window=5`**, **Adam**                                                                                           |
| Evaluation         | Frozen vs online was a Week 8 issue    | Default **online** holdout                                                                                                   |
| Universe           | 41-market story in old guide           | Pipeline `**max_markets=40`** (deliberate signal-to-parameter balance)                                                       |
| Data clock         | Hourly mental model                    | `**history_fidelity=10`** (10-minute bars); walk-forward lengths **scaled** (~10d train / ~2d test per fold in 10-min steps) |
| Tuning             | Grid emphasis                          | **Optuna** + pruner fix                                                                                                      |
| Reporting          | Figures only                           | **Automated Week 9 diagnostics** (attribution tables, correlation summary)                                                   |


---

## 6) Where we are so far (Week 9 — use `docs/week9_diagnostics_report.md`)

The pipeline regenerates `docs/week9_diagnostics_report.md` after each full run. The following snapshot matches the **committed** report (update your slides if you re-ran the pipeline):

- **Run context:** `week8` artifact prefix, `min_history_days_used = 24.0`, **40** markets, mean-downside with **variance / downside penalties = 1.2 / 1.2** (as recorded in that report).
- **Holdout:**


| Metric       | Baseline | Constrained | Delta    |
| ------------ | -------- | ----------- | -------- |
| Sortino      | 0.0791   | 0.0700      | −0.0091  |
| Max drawdown | −7.16%   | −7.34%      | −0.18%   |
| Volatility   | 0.00336  | 0.00297     | −0.00040 |


- **Attribution (same report):** PnL attribution is **concentrated**: top market/domain (`databricks`) ~**42%** of attributed return; use this to discuss **idiosyncratic event risk**, not “diversified factor exposure.”
- **Risk structure:** variance ratio constrained vs baseline ≈ **0.93**; max abs category correlation small (~0.035 in that run).

**Interpretation for slides:**

- Constrained portfolio is **not** ahead on holdout Sortino in this snapshot; volatility is slightly lower.
- Weights in many regimes still sit **near** equal-weight; the project’s honest punchline is **iterative diagnosis** (dilution, penalties, objective conditioning) and **ongoing** Optuna search under richer data.

---

## 7) Recommended slide order (Week 9 / Draft 3)

1. **Problem + motivation:** cross-domain Polymarket allocation; tail risk from domain concentration.
2. **“Since Week 8” delta slide:** table in section 5 above (audience sees you responded to negative results).
3. **Data pipeline:** APIs → tagging → **10-min** histories → `**max_markets=40`** → cached `week8_`* CSVs.
4. **Method:** online rolling-window updates; **mean–downside** training objective; domain + concentration + **covariance** + optional entropy; **uniform mix** as floor.
5. **Evaluation:** walk-forward folds (scaled for 10-min data) + **online** holdout; Optuna for hyperparameters.
6. **Results table:** holdout metrics from `**docs/week9_diagnostics_report.md`** (refresh numbers if you rerun).
7. **Attribution / risk structure:** top contributors + domain shares; correlation / variance-ratio bullets from the same report.
8. **Figures (if refreshed by pipeline):**
  `figures/week8_iteration_equity_curve_comparison.png`,  
   `figures/week8_iteration_drawdown_comparison.png`,  
   `figures/week8_iteration_risk_return_snapshot.png`,  
   rolling mean + return distribution,  
   category exposure + top exposure deltas.
9. **Next steps:** e.g. bootstrap CIs on Sortino, stress tests, narrower universe prescreen, further Optuna trials (`OPTUNA_N_TRIALS` in pipeline), tuning `variance_penalties` / `downside_penalties` (already in search tuples).

**Repro command (full run, slow):** `python script/polymarket_week8_pipeline.py`  
(Uses `QUICK_SANITY_CHECK` / `OPTUNA_N_TRIALS` at top of `main()` — adjust for demos vs production search.)

---

## 8) Suggested one-line takeaway

“We replaced a fragile Sortino-as-loss setup with a smooth mean–downside objective, 10-minute data, a 40-market universe, and Optuna-driven walk-forward search; holdout Sortino still favors the baseline in our latest snapshot, but diagnostics and attribution make the failure mode explicit and guide the next experiments.”