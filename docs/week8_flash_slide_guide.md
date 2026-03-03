# Week 8 Flash Slide Guide

## 1) Why this problem is interesting

- Prediction markets are a real-money, real-time setting where beliefs update quickly.
- Categories (sports, elections, geopolitics, macro, etc.) create a natural cross-domain diversification problem.
- Core proposal idea: explicitly penalize over-exposure to a single event domain to reduce tail risk.
- A naive equal-weight portfolio is simple but may miss structure in return/risk dynamics.
- A constrained online optimizer can adapt over time while respecting diversification limits.

---

## 2) Process overview (one slide, repo-specific)

1. **Ingest active events** (`src/polymarket_data.py::fetch_active_events`): call `gamma-api.polymarket.com/events` with pagination (`limit`, `offset`), save raw events to `data/raw/week8_events_raw.json`.
2. **Flatten + domain tagging** (`flatten_event_markets`): convert event->market nested JSON to market rows, map tags to category/domain, and keep YES token IDs for binary interpretation.
3. **Category-balanced market selection** (`_select_balanced_market_rows`): prioritize high-liquidity categories, apply per-category caps, and exclude noisy slugs.
4. **Fetch token histories** (`fetch_price_history`): call `clob.polymarket.com/prices-history`, then filter markets by minimum history depth (`min_history_points`) and time span (`min_history_days`).
5. **Build cached Week 8 dataset** (`build_dataset`): write `week8_markets_filtered.csv`, `week8_price_history.csv`, `week8_data_quality.json`, `week8_category_liquidity.csv`, `week8_considered_domains.json`.
6. **Baseline run** (`src/baseline.py::run_equal_weight_baseline`): build price matrix, compute returns, allocate equal weight by domain, and save baseline metrics/time series.
7. **Constrained run + tuning** (`src/constrained_optimizer.py::run_experiment_grid`): optimize online with Sortino objective + domain/concentration penalties + entropy bonus, tune via walk-forward blocks, then select params.
8. **Holdout evaluation + diagnostics**: evaluate selected params on holdout split, run covariance diagnostics (`src/covariance_diagnostics.py`), and generate comparison figures in `figures/week8_iteration_*` (`script/polymarket_week8_pipeline.py`).

---

## 3) Core math formulas (one slide)

Use LaTeX math blocks for rendering:

Definitions:

- $r_t \in \mathbb{R}^N$: vector of token returns at step $t$
- $w_t \in \Delta^N$: portfolio weights on simplex, $\sum_i w_{t,i}=1$ and $w_{t,i}\ge 0$
- $\mathcal{D}$: set of event domains/categories
- $S_d(w)=\sum_{i\in d} w_i$: total capital allocated to domain $d$
- $L_d$: domain limit for domain $d$

Step return:

$$
R_t = w_t^\top r_t
$$

Cumulative value:

$$
V_t = \prod_{s=1}^{t} (1 + R_s)
$$

Drawdown:

$$
\mathrm{DD}_t = \frac{V_t}{\max_{s \le t} V_s} - 1
$$

Sortino:

$$
\mathrm{Sortino}(R)=\frac{\mathbb{E}[R]}{\sqrt{\mathbb{E}\!\left[(\min(R,0))^2\right]}+\varepsilon}
$$

Domain overexposure penalty:

$$
P_{\mathrm{domain}}(w)=\sum_{d\in\mathcal{D}} \max\!\left(0,S_d(w)-L_d\right)^2
$$

Concentration penalty:

$$
P_{\mathrm{conc}}(w)=\sum_i \max(0, w_i - w_{\max})^2
$$

Entropy bonus:

$$
B_{\mathrm{ent}}(w)=-\sum_i w_i\log(w_i+\varepsilon)
$$

Weight mixing with uniform allocation $u$:

$$
\tilde{w}_t=(1-\alpha)\,\mathrm{softmax}(z_t)+\alpha\,u
$$

Objective per update window:

$$
\max \left[
\mathrm{Sortino}(R_{\mathrm{window}})
-\lambda\,P_{\mathrm{domain}}(\tilde{w}_t)
-\lambda_c\,P_{\mathrm{conc}}(\tilde{w}_t)
+\lambda_e\,B_{\mathrm{ent}}(\tilde{w}_t)
\right]
$$

Proposal baseline objective (special case framing):

$$
\max\ \mathrm{Sortino}_t(R_p)-\lambda\sum_{d\in\mathcal{D}}\left[\max(0,S_d-L_d)\right]^2
$$

How the math is applied in the current pipeline:

- **Data to returns ($r_t$):** `src/baseline.py::_build_price_matrix` aligns token prices by timestamp; `_compute_returns` converts prices to step returns.
- **Portfolio return ($R_t=w_t^\top r_t$):** realized in baseline via `portfolio_returns = returns_matrix @ weights`, and in constrained online updates via `float(returns_matrix[t] @ current_weights)`.
- **Sortino objective:** implemented in constrained training as `_sortino_torch(portfolio)` in `src/constrained_optimizer.py`.
- **Domain penalty ($P_{\mathrm{domain}}$):** implemented in `_domain_penalty`, aggregating weights by domain and penalizing excess above `domain_limit`.
- **Concentration + entropy terms:** implemented in `_run_online_pass` using `max_weight` cap penalty and entropy bonus to discourage collapse.
- **Simplex weights ($w_t\in\Delta^N$):** enforced by `softmax(logits)` and mixed with uniform allocation (`uniform_mix`) for a diversification floor.
- **Walk-forward and holdout:** hyperparameters are selected on rolling walk-forward validation, then evaluated on holdout (`holdout_fraction`) to produce final reported metrics.

---

## 4) Where we are so far (Week 8)

- **Markets retained:** 41
- **Category count:** 41
- **Current constrained setup:** `uniform_mix=0.88`, `max_weight=0.05`, `domain_limit=0.12`
- **Tuning (walk-forward) constrained Sortino:** 0.0517
- **Holdout constrained mean return:** -1.38e-06 (near flat)
- **Holdout constrained max drawdown:** -8.29%
- **Baseline Sortino:** 0.0466
- **Baseline max drawdown:** -7.84%

Interpretation:
- The constrained strategy is now close to baseline risk/return and no longer collapses into one category.
- Holdout performance is near flat, so this is a stable prototype but not yet a clear outperformance result.
- The current implementation is end-to-end reproducible from one script: `python script/polymarket_week8_pipeline.py`.

---

## 5) Recommended flash slide order (proposal-aligned)

1. **Problem statement + objective:** maximize risk-adjusted return while capping domain exposure.
2. **Data pipeline slide (concrete):** events API -> market flattening/tagging -> category-balanced selection -> price history filters -> cached `week8_*` artifacts.
3. **Method slide:** online optimization (OGD/SGD-style sequential updates) with differentiable domain penalties.
4. **Math/definitions slide:** use section 3 formulas and include where each term appears in code.
5. **Evaluation design slide:** walk-forward tuning + untouched holdout split.
6. **Equity comparison:** `figures/week8_iteration_equity_curve_comparison.png`
7. **Drawdown comparison:** `figures/week8_iteration_drawdown_comparison.png`
8. **Risk-return snapshot:** `figures/week8_iteration_risk_return_snapshot.png`
9. **Rolling mean return + distribution:** `figures/week8_iteration_rolling_mean_return_comparison.png` and `figures/week8_iteration_return_distribution_comparison.png`
10. **Exposure results:** `figures/week8_iteration_category_exposure_comparison.png` and `figures/week8_iteration_top_exposure_deltas.png`
11. **Success metric close:** baseline outperformance + drawdown context relative to prior PRISM benchmark (60.6% historical drawdown), plus next-step tuning targets.

---

## 6) Suggested one-line takeaway

"We built an end-to-end cross-domain Polymarket portfolio pipeline with walk-forward tuning and holdout evaluation; constrained optimization now remains diversified and stable, but improving out-of-sample edge is the main next step."
