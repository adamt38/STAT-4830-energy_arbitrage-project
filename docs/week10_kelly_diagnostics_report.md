# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD

This document aggregates **multiple merged branches**; each section is one pod’s frozen metrics snapshot.

---

## Run: `week14_MF_seed7` (prior `main` / `option-B-kelly-momentum`)

### Run Context
- artifact prefix: `week14_MF_seed7`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `6137`
- holdout steps: `1535`
- step duration assumed: `60 min` (steps/year = `8760`)
- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**
- Optuna trials: `100` completed
- selected MLP hidden dim: `8`
- MLP parameter count: `1814`

### Selected Hyperparameters

| Parameter | Value |
|-----------|-------|
| Weight learning rate (`lr_w`) | `0.01007` |
| MLP learning rate (`lr_theta`) | `0.00091` |
| Rolling window | `24` |
| Monte-Carlo samples per inner step | `512` |
| L1 turnover penalty (`lambda_turn`) | `0.0495` |
| Copula shrinkage to identity | `0.068` |
| Straight-through Bernoulli temperature | `0.100` |
| Concentration penalty `lambda_conc` | `2.267` |
| Per-asset cap (`max_weight`) | `0.078` |
| Transaction fee rate (`fee_rate`) | `0.00000` (0.0 bps per unit L1 turnover) |
| Downside penalty (`dd_penalty`) | `0.000` |

### Holdout Log-Wealth Growth (Kelly objective, primary metric)

> Kelly columns are gross of fees (``fee_rate = 0``).

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.320947 | +0.261717 | -0.059230 |
| Per-step log-wealth | +0.00020909 | +0.00017050 | -0.00003859 |
| Annualized log-growth | +1.8316 | +1.4936 | -0.3380 |
| Equivalent CAGR | +524.38% | +345.30% | -179.08% |
| Max drawdown (realized) | -24.3649% | -30.3175% | -5.9526% |
| Mean realized step return | +0.00024125 | +0.00022392 | — |
| Realized return volatility | 0.00813496 | 0.01054194 | — |

### Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `18.0267` |
| Average per-step L1 turnover | `0.011744` |
| Maximum per-step L1 turnover | `0.582576` |
| Equal-weight baseline turnover (reference) | `0.0000` |

### Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `-0.6528` |
| Average off-diagonal correlation (mean abs) | `0.6684` |
| Maximum off-diagonal correlation (abs) | `0.8445` |
| Average correlation matrix condition number | `38.69` |
| Holdout steps requiring PD shrinkage fallback | `22543` (1468.6% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `False` |

### Interpretation Checklist
- [ ] Kelly OGD beats equal-weight on cumulative log-wealth (delta `-0.059230`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.0117`)
- [ ] PD shrinkage fallback fires on at most 5% of steps (`1468.6%`)

---

## Run: `week10_kelly_A` (source branch: `cloud-runs-K10A`)

### Run Context
- artifact prefix: `week10_kelly_A`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `45712`
- holdout steps: `11429`
- step duration assumed: `60 min` (steps/year = `8760`)
- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**
- Optuna trials: `100` completed
- selected MLP hidden dim: `8`
- MLP parameter count: `7124`

### Selected Hyperparameters

| Parameter | Value |
|-----------|-------|
| Weight learning rate (`lr_w`) | `0.01007` |
| MLP learning rate (`lr_theta`) | `0.00091` |
| Rolling window | `96` |
| Monte-Carlo samples per inner step | `256` |
| L1 turnover penalty (`lambda_turn`) | `0.0495` |
| Copula shrinkage to identity | `0.068` |
| Straight-through Bernoulli temperature | `0.100` |
| Concentration penalty `lambda_conc` | `2.267` |
| Per-asset cap (`max_weight`) | `0.078` |

### Holdout Log-Wealth Growth (Kelly objective, primary metric)

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.408057 | +0.584289 | +0.176232 |
| Per-step log-wealth | +0.00003570 | +0.00005112 | +0.00001542 |
| Annualized log-growth | +0.3128 | +0.4478 | +0.1351 |
| Equivalent CAGR | +36.72% | +56.49% | +19.77% |
| Max drawdown (realized) | -4.5185% | -12.4060% | -7.8875% |
| Mean realized step return | +0.00003726 | +0.00005502 | — |
| Realized return volatility | 0.00179298 | 0.00283497 | — |

### Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `314.5714` |
| Average per-step L1 turnover | `0.027524` |
| Maximum per-step L1 turnover | `0.976817` |
| Equal-weight baseline turnover (reference) | `0.0000` |

### Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `-0.3662` |
| Average off-diagonal correlation (mean abs) | `0.5440` |
| Maximum off-diagonal correlation (abs) | `0.9093` |
| Average correlation matrix condition number | `75.47` |
| Holdout steps requiring PD shrinkage fallback | `170816` (1494.6% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `False` |

### Interpretation Checklist
- [x] Kelly OGD beats equal-weight on cumulative log-wealth (delta `+0.176232`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.0275`)
- [ ] PD shrinkage fallback fires on at most 5% of steps (`1494.6%`)
