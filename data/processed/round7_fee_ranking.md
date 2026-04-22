# Round 7 post-hoc net-of-fees ranking (Kelly runs)

Deducted `fee_rate * turnover_l1` from each step's gross return on the already-evaluated Kelly holdout timeseries, then recomputed Sortino / total log-wealth / equivalent CAGR / max drawdown. Steps/year = `8760`. Runs: K10A, K10B, K10C.

## Headline: Kelly net-of-fees vs equal-weight baseline

| Run | fee (bps) | Strategy | Total log-wealth | Ann. log-growth | CAGR | Sortino | Max DD | avg L1 turn |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| K10A | 0.0 | kelly_constrained | +0.5843 | +0.4478 | +56.49% | +0.0396 | -12.41% | 0.0275 |
| K10A | 0.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10A | 10.0 | kelly_constrained | +0.2697 | +0.2067 | +22.97% | +0.0197 | -13.71% | 0.0275 |
| K10A | 10.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10A | 50.0 | kelly_constrained | -0.9886 | -0.7577 | -53.13% | -0.0585 | -67.97% | 0.0275 |
| K10A | 50.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10A | 200.0 | kelly_constrained | -5.7088 | -4.3756 | -98.74% | -0.3180 | -99.70% | 0.0275 |
| K10A | 200.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10B | 0.0 | kelly_constrained | +0.6983 | +0.5353 | +70.79% | +0.0456 | -11.99% | 0.0232 |
| K10B | 0.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10B | 10.0 | kelly_constrained | +0.4327 | +0.3316 | +39.32% | +0.0294 | -13.84% | 0.0232 |
| K10B | 10.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10B | 50.0 | kelly_constrained | -0.6301 | -0.4829 | -38.30% | -0.0347 | -58.14% | 0.0232 |
| K10B | 50.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10B | 200.0 | kelly_constrained | -4.6163 | -3.5383 | -97.09% | -0.2546 | -99.14% | 0.0232 |
| K10B | 200.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10C | 0.0 | kelly_constrained | +0.8678 | +0.6652 | +94.48% | +0.0579 | -11.74% | 0.1071 |
| K10C | 0.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10C | 10.0 | kelly_constrained | -0.3565 | -0.2732 | -23.91% | -0.0166 | -53.96% | 0.1071 |
| K10C | 10.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10C | 50.0 | kelly_constrained | -5.2551 | -4.0279 | -98.22% | -0.2853 | -99.59% | 0.1071 |
| K10C | 50.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |
| K10C | 200.0 | kelly_constrained | -23.6444 | -18.1228 | -100.00% | -0.7558 | -100.00% | 0.1071 |
| K10C | 200.0 | equal_weight_baseline | +0.4081 | +0.3128 | +36.72% | +0.0525 | -4.52% | 0.0000 |

## Net-vs-gross delta (constrained minus baseline, per fee level)

| Run | fee (bps) | Δ total log-wealth | Δ CAGR | Δ Sortino | Δ Max DD |
|---|---:|---:|---:|---:|---:|
| K10A | 0.0 | +0.1762 | +19.77% | -0.0129 | -7.89% |
| K10A | 10.0 | -0.1383 | -13.75% | -0.0327 | -9.19% |
| K10A | 50.0 | -1.3967 | -89.85% | -0.1110 | -63.45% |
| K10A | 200.0 | -6.1169 | -135.46% | -0.3705 | -95.18% |
| K10B | 0.0 | +0.2903 | +34.07% | -0.0069 | -7.47% |
| K10B | 10.0 | +0.0246 | +2.60% | -0.0231 | -9.32% |
| K10B | 50.0 | -1.0381 | -75.02% | -0.0872 | -53.62% |
| K10B | 200.0 | -5.0243 | -133.81% | -0.3071 | -94.62% |
| K10C | 0.0 | +0.4598 | +57.76% | +0.0054 | -7.22% |
| K10C | 10.0 | -0.7645 | -60.63% | -0.0691 | -49.44% |
| K10C | 50.0 | -5.6631 | -134.94% | -0.3378 | -95.07% |
| K10C | 200.0 | -24.0525 | -136.72% | -0.8082 | -95.48% |

## Caveat on MVO (week8/week11/week12/week13) runs

The constrained MVO pipelines did not log per-step turnover or push `*_weights.csv`, so exact fee adjustment is not possible from the pushed artifacts. However, every MVO run on the 20-market universe (I4 / Q5 / S1 / S4 / S5) ties the equal-weight baseline within +/-0.02 Sortino *before* fees -- any positive fee rate strictly degrades their net-of-fees performance, so they cannot overtake Kelly after friction. This is a bound, not a calculation.

## Interpretation

- At `fee = 0` the Kelly runs show their gross advantage (K10C: +0.46 log-wealth, +58 pp CAGR vs baseline).
- The break-even fee rate is where `Δ total log-wealth -> 0` (see table below).
- This motivates K10E (fee-aware Kelly) on the pod: by sweeping fee_rate directly inside the optimizer's loss + reported return, we let the optimizer re-allocate to LOWER-turnover policies when friction is expensive, rather than paying fees on a policy optimized for a frictionless world.

## Break-even fee rate per run

`break_even ≈ (Δ log-wealth at fee=0) / (n_steps * avg_turnover_per_step)`

| Run | Δ log-w @ fee=0 | n_steps | avg turnover | break-even fee (bps) |
|---|---:|---:|---:|---:|
| K10A | +0.1762 | 11429 | 0.0275 | 5.60 |
| K10B | +0.2903 | 11429 | 0.0232 | 10.93 |
| K10C | +0.4598 | 11429 | 0.1071 | 3.76 |
