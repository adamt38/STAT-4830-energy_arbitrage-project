# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD

## Run Context
- artifact prefix: `week10_kelly_C`
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
| Weight learning rate (`lr_w`) | `0.01037` |
| MLP learning rate (`lr_theta`) | `0.00060` |
| Rolling window | `48` |
| Monte-Carlo samples per inner step | `512` |
| L1 turnover penalty (`lambda_turn`) | `0.0000` |
| Copula shrinkage to identity | `0.081` |
| Straight-through Bernoulli temperature | `0.100` |
| Concentration penalty `lambda_conc` | `5.832` |
| Per-asset cap (`max_weight`) | `0.082` |

## Holdout Log-Wealth Growth (Kelly objective, primary metric)

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.408057 | +0.867827 | +0.459770 |
| Per-step log-wealth | +0.00003570 | +0.00007593 | +0.00004023 |
| Annualized log-growth | +0.3128 | +0.6652 | +0.3524 |
| Equivalent CAGR | +36.72% | +94.48% | +57.76% |
| Max drawdown (realized) | -4.5185% | -11.7410% | -7.2225% |
| Mean realized step return | +0.00003726 | +0.00008299 | — |
| Realized return volatility | 0.00179298 | 0.00392336 | — |

## Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `1224.3187` |
| Average per-step L1 turnover | `0.107124` |
| Maximum per-step L1 turnover | `1.116585` |
| Equal-weight baseline turnover (reference) | `0.0000` |

## Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `-0.4425` |
| Average off-diagonal correlation (mean abs) | `0.5501` |
| Maximum off-diagonal correlation (abs) | `0.8957` |
| Average correlation matrix condition number | `250.06` |
| Holdout steps requiring PD shrinkage fallback | `170599` (1492.7% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `False` |

## Interpretation Checklist
- [x] Kelly OGD beats equal-weight on cumulative log-wealth (delta `+0.459770`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.1071`)
- [ ] PD shrinkage fallback fires on at most 5% of steps (`1492.7%`)
