# Post-hoc alpha-blend evaluation: `week10_kelly_C`

- **baseline series** `week10_kelly_C_baseline_timeseries.csv` (full rows: 57141)
- **optimized series** `week10_kelly_C_kelly_best_timeseries.csv` (holdout rows: 11429)
- **aligned holdout rows** used: 11429

Blend definition: `r_blend(t; alpha) = alpha * r_constrained(t) + (1 - alpha) * r_baseline(t)`. `alpha = 1.0` is the pure constrained portfolio (what the pipeline reports); `alpha = 0.0` is pure equal-weight baseline.

| alpha | Sortino | mean_ret | volatility | max_dd | total_ret | cum_log_wealth |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | +0.0525 | +0.000037 | 0.001793 | -4.5185% | +50.3893% | +0.408057 |
| 0.10 | +0.0550 | +0.000042 | 0.001947 | -4.7795% | +57.9625% | +0.457188 |
| 0.20 | +0.0567 | +0.000046 | 0.002124 | -5.4882% | +65.7966% | +0.505591 |
| 0.25 | +0.0573 | +0.000049 | 0.002219 | -5.8409% | +69.8124% | +0.529524 |
| 0.30 | +0.0578 | +0.000051 | 0.002318 | -6.1925% | +73.8946% | +0.553279 |
| 0.40 | +0.0583 | +0.000056 | 0.002525 | -6.8927% | +82.2594% | +0.600261 |
| 0.50 | +0.0586 | +0.000060 | 0.002743 | -7.6082% | +90.8936% | +0.646546 |
| 0.60 | +0.0586 | +0.000065 | 0.002968 | -8.3910% | +99.7994% | +0.692144 |
| 0.70 | +0.0586 | +0.000069 | 0.003200 | -9.2158% | +108.9789% | +0.737063 |
| 0.75 | +0.0585 | +0.000072 | 0.003318 | -9.6259% | +113.6717% | +0.759270 |
| 0.80 | +0.0584 | +0.000074 | 0.003438 | -10.0465% | +118.4335% | +0.781311 |
| 0.90 | +0.0582 | +0.000078 | 0.003679 | -10.8971% | +128.1645% | +0.824897 |
| 1.00 | +0.0579 | +0.000083 | 0.003923 | -11.7410% | +138.1730% | +0.867827 |

## Argmax summary

- **best Sortino** at alpha = 0.60 (Sortino = +0.0586)
- **best total return** at alpha = 1.00 (total = +138.1730%)
- **least-negative max drawdown** at alpha = 0.00 (max_dd = -4.5185%)

If the Sortino argmax is strictly interior (0 < alpha < 1) and beats the alpha=1 row, the optimized constrained portfolio is *over*-exposed on a risk-adjusted basis — a blended allocation would have dominated on this holdout. If the argmax is at alpha = 1.0, the constrained portfolio is already efficient relative to the baseline and there is no free lunch in post-hoc dilution.
