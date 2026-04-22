# Option A Results — Momentum × Baseline-Shrinkage × macro=both

A novel three-lever combination that Adam's Round-2/4/5 pods never tested jointly. Round 2's B2 tested baseline-shrinkage alone with `macro=rescale` (Δ = 0.000 — Optuna collapsed α to 1.0). Round 2's G/Round 4's I4 tested momentum alone with `macro=both` (Δ ∈ {+0.034, +0.008}). Nothing combines all three: shrinkage + momentum + macro=both.

**Hypothesis**: shrinkage reduces selection overfit (α < 1 blends toward equal-weight at eval time, attenuating any one Optuna best-trial's idiosyncrasies). Momentum removes dead-weight markets (reduces parameter count). macro=both is the Round 4 winner. All three are orthogonal risk-reduction mechanisms; combining them should reduce the tuning→holdout gap that killed Round 5 G seeds.

## Configuration

| Lever | Value |
|---|---|
| Macro mode | `both` (rescale path + additive `J_macro` term) |
| Momentum screening | top-20 markets, 5-day lookback |
| Baseline shrinkage | enabled (Optuna tunes α ∈ [0, 1]) |
| Reduced search | enabled (denser Sobol on the meaningful subspace) |
| Top-K bagging | 5 |
| Optuna trials | 100 per seed |
| Seeds | {7, 42, 123} |

## Results by seed

| Seed | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD | Max dom exp |
|---|---|---|---|---|---|---|---|
| 7 | +0.1914 | +0.1749 | **-0.0166** | [-0.0737, +0.0220] (frac+ 23.6%) | -20.33% | -20.89% | 0.1000 |
| 42 | +0.1914 | +0.1742 | **-0.0173** | [-0.0756, +0.0217] (frac+ 23.2%) | -20.33% | -20.87% | 0.0999 |
| 123 | +0.1914 | +0.1747 | **-0.0168** | [-0.0745, +0.0220] (frac+ 23.8%) | -20.33% | -20.91% | 0.0994 |

**Ensemble stats (3 seeds):**
- Mean Δ: **-0.0169 ± 0.0003** (std)
- Δ range: [-0.0173, -0.0166]
- Fraction of seeds with positive Δ: **0%**

**Benchmarks for comparison:**
- Seed noise floor (Adam §15.9): **~0.036** Sortino (from G2 ↔ G-seed42 |ΔΔ|)
- Adam's I4 single-seed Δ: **+0.0077**
- Teammate's week17 Δ: **+0.1018** (on `origin/stock-PM-combined-strategy`)

**Verdict**: novel combo **does not beat baseline**. Most seeds show negative delta; the added shrinkage lever likely over-regularizes the momentum-screened universe.

## Selected hyperparameters per seed

**seed=7**:
  - `learning_rate`: 0.0056
  - `rolling_window`: 96
  - `domain_limit`: 0.0936
  - `max_weight`: 0.0450
  - `uniform_mix`: 0.0065
  - `concentration_penalty_lambda`: 5.5791
  - `covariance_penalty_lambda`: 6.6255
  - `variance_penalty`: 1.8394
  - `downside_penalty`: 2.2820

**seed=42**:
  - `learning_rate`: 0.0474
  - `rolling_window`: 24
  - `domain_limit`: 0.1037
  - `max_weight`: 0.0414
  - `uniform_mix`: 0.0376
  - `concentration_penalty_lambda`: 9.1069
  - `covariance_penalty_lambda`: 4.5456
  - `variance_penalty`: 1.2104
  - `downside_penalty`: 2.8808

**seed=123**:
  - `learning_rate`: 0.0114
  - `rolling_window`: 24
  - `domain_limit`: 0.1156
  - `max_weight`: 0.0466
  - `uniform_mix`: 0.0734
  - `concentration_penalty_lambda`: 28.7541
  - `covariance_penalty_lambda`: 3.1030
  - `variance_penalty`: 1.9219
  - `downside_penalty`: 2.9844

