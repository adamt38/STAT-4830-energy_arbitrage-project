# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD

## Run Context
- artifact prefix: `week10_kelly_D`
- min history days used after backoff: `24.0`
- market count: `40`
- tuning steps: `45712`
- holdout steps: `11429`
- step duration assumed: `60 min` (steps/year = `8760`)
- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**
- Optuna trials: `100` completed
- selected MLP hidden dim: `8`
- MLP parameter count: `7124`

## Selected Hyperparameters

| Parameter | Value |
|-----------|-------|
| Weight learning rate (`lr_w`) | `0.01007` |
| MLP learning rate (`lr_theta`) | `0.00091` |
| Rolling window | `48` |
| Monte-Carlo samples per inner step | `1024` |
| L1 turnover penalty (`lambda_turn`) | `0.0495` |
| Copula shrinkage to identity | `0.068` |
| Straight-through Bernoulli temperature | `0.100` |
| Concentration penalty `lambda_conc` | `2.267` |
| Per-asset cap (`max_weight`) | `0.078` |

## Holdout Log-Wealth Growth (Kelly objective, primary metric)

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.408057 | +0.677669 | +0.269612 |
| Per-step log-wealth | +0.00003570 | +0.00005929 | +0.00002359 |
| Annualized log-growth | +0.3128 | +0.5194 | +0.2066 |
| Equivalent CAGR | +36.72% | +68.10% | +31.38% |
| Max drawdown (realized) | -4.5185% | -10.8277% | -6.3091% |
| Mean realized step return | +0.00003726 | +0.00006377 | — |
| Realized return volatility | 0.00179298 | 0.00305383 | — |

## Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `231.8823` |
| Average per-step L1 turnover | `0.020289` |
| Maximum per-step L1 turnover | `1.088553` |
| Equal-weight baseline turnover (reference) | `0.0000` |

## Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `-0.4429` |
| Average off-diagonal correlation (mean abs) | `0.5140` |
| Maximum off-diagonal correlation (abs) | `0.9073` |
| Average correlation matrix condition number | `146.72` |
| Holdout steps requiring PD shrinkage fallback | `171170` (1497.7% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `False` |

## Interpretation Checklist
- [x] Kelly OGD beats equal-weight on cumulative log-wealth (delta `+0.269612`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.0203`)
- [ ] PD shrinkage fallback fires on at most 5% of steps (`1497.7%`)
