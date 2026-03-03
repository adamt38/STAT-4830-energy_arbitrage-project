# Week 8 Flash Slide Guide

## 1) Why this problem is interesting

- Prediction markets are a real-money, real-time setting where beliefs update quickly.
- Categories (sports, elections, geopolitics, macro, etc.) create a natural cross-domain diversification problem.
- A naive equal-weight portfolio is simple but may miss structure in return/risk dynamics.
- A constrained online optimizer can adapt over time while respecting diversification limits.

---

## 2) Process overview (one slide)

1. Pull active Polymarket events and markets.
2. Keep liquid categories and markets with enough history.
3. Build YES-token price history matrix and compute step returns.
4. Run baseline: equal category-weight portfolio.
5. Run constrained online optimizer with diversification penalties.
6. Tune hyperparameters with walk-forward validation.
7. Evaluate selected setup on untouched holdout.
8. Compare baseline vs constrained in equity, drawdown, exposure, and distribution figures.

---

## 3) Core math formulas (one slide)

Let \(r_t \in \mathbb{R}^N\) be the vector of token returns at step \(t\), and \(w_t \in \Delta^N\) be portfolio weights on the simplex.

Step return:
\[
R_t = w_t^\top r_t
\]

Cumulative value:
\[
V_t = \prod_{s=1}^{t} (1 + R_s)
\]

Drawdown:
\[
\text{DD}_t = \frac{V_t}{\max_{s \le t} V_s} - 1
\]

Sortino (used in optimization objective):
\[
\text{Sortino}(R) = \frac{\mathbb{E}[R]}{\sqrt{\mathbb{E}[(\min(R,0))^2]} + \epsilon}
\]

Domain overexposure penalty (for domains \(d\)):
\[
\mathcal{P}_{\text{domain}}(w) = \sum_d \max\!\left(0, \sum_{i \in d} w_i - L_d\right)^2
\]

Concentration penalty:
\[
\mathcal{P}_{\text{conc}}(w) = \sum_i \max(0, w_i - w_{\max})^2
\]

Entropy bonus:
\[
\mathcal{B}_{\text{ent}}(w) = -\sum_i w_i \log(w_i + \epsilon)
\]

Weight mixing with uniform allocation \(u\):
\[
\tilde{w}_t = (1-\alpha)\,\text{softmax}(z_t) + \alpha u
\]

Optimization objective per update window:
\[
\max \; \text{Sortino}(R_{\text{window}})
- \lambda \mathcal{P}_{\text{domain}}(\tilde{w})
- \lambda_c \mathcal{P}_{\text{conc}}(\tilde{w})
+ \lambda_e \mathcal{B}_{\text{ent}}(\tilde{w})
\]

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

---

## 5) Recommended slide figure order

1. `figures/week8_iteration_equity_curve_comparison.png`
2. `figures/week8_iteration_drawdown_comparison.png`
3. `figures/week8_iteration_risk_return_snapshot.png`
4. `figures/week8_iteration_rolling_mean_return_comparison.png`
5. `figures/week8_iteration_return_distribution_comparison.png`
6. `figures/week8_iteration_category_exposure_comparison.png`
7. `figures/week8_iteration_top_exposure_deltas.png`

---

## 6) Suggested one-line takeaway

"We built an end-to-end cross-domain Polymarket portfolio pipeline with walk-forward tuning and holdout evaluation; constrained optimization now remains diversified and stable, but improving out-of-sample edge is the main next step."
