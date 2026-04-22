# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD

## Run Context
- artifact prefix: `week14_M_seed7`
- min history days used after backoff: `24.0`
- market count: `20`
- tuning steps: `5963`
- holdout steps: `1491`
- step duration assumed: `60 min` (steps/year = `8760`)
- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**
- Optuna trials: `100` completed
- selected MLP hidden dim: `16`
- MLP parameter count: `3566`

## Selected Hyperparameters

| Parameter | Value |
|-----------|-------|
| Weight learning rate (`lr_w`) | `0.01007` |
| MLP learning rate (`lr_theta`) | `0.00091` |
| Rolling window | `48` |
| Monte-Carlo samples per inner step | `512` |
| L1 turnover penalty (`lambda_turn`) | `0.0495` |
| Copula shrinkage to identity | `0.068` |
| Straight-through Bernoulli temperature | `0.100` |
| Concentration penalty `lambda_conc` | `2.267` |
| Per-asset cap (`max_weight`) | `0.078` |
| Transaction fee rate (`fee_rate`) | `0.00000` (0.0 bps per unit L1 turnover) |
| Downside penalty (`dd_penalty`) | `0.000` |

## Holdout Log-Wealth Growth (Kelly objective, primary metric)

> Kelly columns are gross of fees (``fee_rate = 0``).

| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |
|--------|-----------------------|--------------------------|-------|
| Total log-wealth ($\sum_t \log(1+r_t)$) | +0.183848 | +0.402425 | +0.218577 |
| Per-step log-wealth | +0.00012330 | +0.00026990 | +0.00014660 |
| Annualized log-growth | +1.0802 | +2.3643 | +1.2842 |
| Equivalent CAGR | +194.51% | +963.71% | +769.19% |
| Max drawdown (realized) | -21.4785% | -27.0609% | -5.5824% |
| Mean realized step return | +0.00015458 | +0.00033539 | — |
| Realized return volatility | 0.00798895 | 0.01175843 | — |

## Turnover Metrics (L1 weight changes, market-friction proxy)

| Metric | Value |
|--------|-------|
| Total holdout L1 turnover | `17.1223` |
| Average per-step L1 turnover | `0.011484` |
| Maximum per-step L1 turnover | `0.569884` |
| Equal-weight baseline turnover (reference) | `0.0000` |

## Dynamic Copula Diagnostics

| Metric | Value |
|--------|-------|
| Average off-diagonal correlation (mean) | `-0.6800` |
| Average off-diagonal correlation (mean abs) | `0.7101` |
| Maximum off-diagonal correlation (abs) | `0.8743` |
| Average correlation matrix condition number | `77.56` |
| Holdout steps requiring PD shrinkage fallback | `22061` (1479.6% of steps) |
| Exogenous features used | `spy_ret_1, qqq_ret_1, btc_usd_ret_1` |
| Copula disabled (R = I baseline) | `False` |

## Interpretation Checklist
- [x] Kelly OGD beats equal-weight on cumulative log-wealth (delta `+0.218577`)
- [ ] Kelly OGD has smaller (less negative) max drawdown than baseline
- [x] Average per-step L1 turnover stays below 0.5 (`0.0115`)
- [ ] PD shrinkage fallback fires on at most 5% of steps (`1479.6%`)
