# Round 7 circular-block bootstrap CI on Kelly log-wealth delta

- **replicates:** 1000; **block length:** 50 steps (~2.1 days on 60-min bars); **seed:** 7.
- Delta = `total_log_wealth(kelly) - total_log_wealth(baseline)` on paired circular-block resamples (same block offsets applied to both series so step-by-step alignment is preserved).

| Run | n_steps | Observed Δ | Bootstrap mean | Std | 95% CI | 99% CI | Pr(Δ>0) | z |
|---|---:|---:|---:|---:|---|---|---:|---:|
| K10A | 11429 | +0.1762 | +0.1886 | 0.2194 | [-0.2323, +0.5988] | [-0.3510, +0.7494] | 0.807 | +0.80 |
| K10B | 11429 | +0.2903 | +0.2990 | 0.1998 | [-0.0971, +0.6964] | [-0.1999, +0.8388] | 0.934 | +1.45 |
| K10C | 11429 | +0.4598 | +0.4610 | 0.2410 | [+0.0047, +0.9730] | [-0.0980, +1.1336] | 0.975 | +1.91 |

## Interpretation

- **Pr(Δ>0)** = fraction of bootstrap replicates where Kelly beat baseline on resampled blocks. A value near 1.0 with a 95% CI that excludes zero is strong evidence of a real (as opposed to noise-driven) gross edge. A CI that straddles zero means the apparent +0.46 log-wealth is within the range of luck given the autocorrelation structure.
- **z-score** = observed_delta / bootstrap_std. Rule of thumb: |z|>2 is marginal evidence, |z|>3 is strong, |z|>5 is overwhelming.
- **Caveat:** these CIs are gross-of-fees. The fee-ranking post-hoc shows break-even fees of 3.8 / 10.9 / 5.6 bps for K10C / K10B / K10A; the CI here only tells you whether the **gross** edge is statistically real.
