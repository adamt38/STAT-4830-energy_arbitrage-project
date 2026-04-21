# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD

## Run Context
- artifact prefix: `week10_kelly_B`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `45712`
- holdout steps: `11429`
- step duration assumed: `60 min` (steps/year = `8760`)
- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**
- Optuna trials: `100` completed
- selected MLP hidden dim: `16`
- MLP parameter count: `13596`

## Selected Hyperparameters

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

## Holdout Log-Wealth Growth (Kelly objective, primary metric)

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.408057 | +0.698340 | +0.290283 |
| Per-step log-wealth | +0.00003570 | +0.00006110 | +0.00002540 |
| Annualized log-growth | +0.3128 | +0.5353 | +0.2225 |
| Equivalent CAGR | +36.72% | +70.79% | +34.07% |
| Max drawdown (realized) | -4.5185% | -11.9854% | -7.4668% |
| Mean realized step return | +0.00003726 | +0.00006563 | — |
| Realized return volatility | 0.00179298 | 0.00305676 | — |

## Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `265.6738` |
| Average per-step L1 turnover | `0.023246` |
| Maximum per-step L1 turnover | `0.866654` |
| Equal-weight baseline turnover (reference) | `0.0000` |

## Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `+0.0000` |
| Average off-diagonal correlation (mean abs) | `0.0000` |
| Maximum off-diagonal correlation (abs) | `0.0000` |
| Average correlation matrix condition number | `1.00` |
| Holdout steps requiring PD shrinkage fallback | `0` (0.0% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `True` |

## Interpretation Checklist
- [x] Kelly OGD beats equal-weight on cumulative log-wealth (delta `+0.290283`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.0232`)
- [x] PD shrinkage fallback fires on at most 5% of steps (`0.0%`)
