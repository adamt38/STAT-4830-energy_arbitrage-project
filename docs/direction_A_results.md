# Direction A Results — Learnable Market Inclusion

Summary of three runs: Pod G (momentum screening, seed=7), Pod G replication (same config, seed=42), and Pod L (learnable inclusion on full universe).

## Head-to-head Sortino (holdout)

| Run | N markets | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD |
|---|---|---|---|---|---|---|---|
| G (seed=7) | 20 | +0.2237 | +0.2576 | **+0.0339** | [-0.0311, +0.1151] (frac+ 82.1%) | -22.50% | -21.26% |
| G (seed=42) | 20 | +0.0169 | +0.0236 | **+0.0067** | [-0.0200, +0.0391] (frac+ 66.0%) | -29.07% | -26.06% |
| L (learnable-inclusion) | 40 | +0.0394 | +0.0223 | **-0.0172** | [-0.0693, +0.0229] (frac+ 23.3%) | -7.31% | -9.91% |

## Replication check (Pod G)

- seed=7 observed delta: **+0.0339**  (95% CI [-0.0311, +0.1151] (frac+ 82.1%))
- seed=42 observed delta: **+0.0067**  (95% CI [-0.0200, +0.0391] (frac+ 66.0%))
- same sign across seeds: **True**
- absolute seed-to-seed spread: **0.0272**

**Interpretation:** same-sign replication but meaningful spread; result is directionally stable but absolute magnitude is seed-sensitive.

## Learnable inclusion (Pod L)

- Universe: 40 markets (full, no pre-filter)
- Holdout Sortino: baseline +0.0394, constrained +0.0223, delta **-0.0172**
- 95% CI on delta: [-0.0693, +0.0229] (frac+ 23.3%)
- Weight std (domain): 0.0195  (equal-weight = 0; higher = more concentrated)
- Max domain exposure: 0.0514  (equal-weight on 40 markets ≈ 0.0250)

**Selected hyperparameters (best trial):**
  - `learning_rate`: 0.0153
  - `rolling_window`: 96
  - `domain_limit`: 0.1006
  - `max_weight`: 0.0541
  - `concentration_penalty_lambda`: 42.9973
  - `covariance_penalty_lambda`: 9.5427
  - `uniform_mix`: 0.0109

## Pod L (learnable) vs Pod G (hand-picked top-K)

- Pod G delta:     **+0.0339**  (universe: top-20 momentum, 5d lookback)
- Pod L delta:     **-0.0172**  (universe: full 40 markets, learned inclusion gates)
- Pod L − Pod G:  -0.0510

**Interpretation:** learnable inclusion does NOT beat baseline in this run, while hand-picked top-K does. The added flexibility (4 new hyperparameters, joint optimization) likely overfits the tuning set. Worth trying: stronger commitment penalty, smaller target_k, or combining learnable inclusion with hand-picked pre-filter.

