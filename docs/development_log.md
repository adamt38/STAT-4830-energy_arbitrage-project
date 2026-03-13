# Development Log

## Week 4 Prototype (Completed)

### Objective of this cycle
- Pivot from battery arbitrage to the proposal topic: cross-domain portfolio optimization on Polymarket.
- Follow execution-first order: code and experiments first, then notebook figures, then documentation.

### Work completed

1. **Data engineering pipeline (completed)**
   - Added `src/polymarket_data.py` with:
     - paginated event retrieval from Gamma API,
     - event-market flattening,
     - tag-to-domain mapping,
     - token-level price history pulls from CLOB API,
     - cached outputs in `data/raw` and `data/processed`,
     - quality checks (missingness, duplicates, monotonic timestamps, domain coverage).
   - Added package marker `src/__init__.py`.

2. **Baseline metrics (completed)**
   - Added `src/baseline.py` with equal-weight benchmark and metrics:
     - Sortino ratio,
     - max drawdown,
     - return volatility,
     - domain exposure shares.
   - Outputs persisted to:
     - `data/processed/baseline_metrics.json`
     - `data/processed/baseline_timeseries.csv`

3. **First constrained experiments (completed)**
   - Added `src/constrained_optimizer.py`:
     - rolling-window OGD/SGD-style updates in PyTorch,
     - differentiable domain-overexposure penalty,
     - grid search over learning rate, penalty lambda, and window length.
   - Outputs persisted to:
     - `data/processed/constrained_experiment_grid.csv`
     - `data/processed/constrained_best_metrics.json`
     - `data/processed/constrained_best_timeseries.csv`

4. **Reproducible orchestration + figures (completed)**
   - Added `script/polymarket_week8_pipeline.py`:
     - runs data build -> baseline -> constrained experiments,
     - generates Week 8 figure artifacts:
       - `figures/week8_equity_curve_comparison.png`
       - `figures/week8_drawdown_comparison.png`
       - `figures/week8_domain_exposure_comparison.png`

5. **Notebook packaging (completed)**
   - Added `notebooks/week8_flash_results.ipynb` to display metrics and figures from cached artifacts.

### Current empirical snapshot
- Markets retained: 7
- Price history points: 4,547
- Baseline: Sortino 0.0274, max drawdown -26.30%
- Best constrained run: Sortino 0.0115, max drawdown -61.92%

### What failed / limitations observed
- First constrained setup is currently less stable than baseline on risk metrics.
- Domain coverage is skewed toward crypto in this first data slice.
- Small number of retained markets limits diversification quality.

## Week 8 Iteration (Now In Progress)

### Planned Week 8 upgrades
1. Expand from coarse domains to many specific category tags.
2. Increase event universe and use category-balanced sampling.
3. Re-run baseline and constrained experiments with expanded categories.
4. Refresh notebook and figure outputs for flash presentation.

### Week 8 update (applied)
- Category mapping now uses specific tag slugs (not only coarse buckets).
- Latest run retained 80 markets and 80 categories selected from high-liquidity groups.
- Baseline now uses equal category weights by construction (each category gets the same total allocation).
- Artifacts are now consistently prefixed and renamed for Week 8 iteration:
  - `data/processed/week8_*`
  - `figures/week8_iteration_*`
  - `notebooks/week8_iteration_flash_results.ipynb`

### Week 8 findings (latest full run)
- Run completed end-to-end with fallback history threshold:
  - `min_history_days_used = 24.0` (30.0 produced zero surviving markets)
  - final universe retained: 177 markets
- Baseline (full series):
  - Sortino: 0.2549
  - max drawdown: -3.67%
  - mean return: 0.0002163
  - volatility: 0.0040589
- Best constrained model (frozen-mode tuning winner):
  - tuning Sortino: 0.3397
  - holdout Sortino: 0.2018
  - holdout max drawdown: -5.01%
  - holdout mean return: 0.0002584
  - holdout volatility: 0.0041359
- Baseline on same holdout horizon (for fair comparison):
  - Sortino: 0.3508
  - max drawdown: -2.73%
  - mean return: 0.0002630
  - volatility: 0.0037610
- Interpretation:
  - Constrained model currently does not beat baseline out-of-sample on risk-adjusted return or drawdown.
  - Covariance diagnostics indicate constrained portfolio is still too concentrated in correlated risk:
    - `variance_ratio_constrained_vs_baseline = 14.06`

### Code-quality and methodology findings applied this cycle
- Removed lookahead leakage in price alignment (no pre-listing backfill in `src/baseline.py`).
- Added strict binary `Yes/No` market filtering and required valid `Yes` token (`src/polymarket_data.py`).
- Added robust parsing for API numeric fields and explicit typed error for no surviving markets.
- Fixed constrained loader to respect `artifact_prefix` instead of hardcoded `week8` paths.
- Added frozen-vs-online evaluation toggle and switched pipeline to frozen-only default for reporting defensibility.
- Hardened covariance diagnostics for degenerate category-count cases.

### Recommended next iteration settings
1. Tighten exposure/concentration controls:
   - `domain_limits=(0.06, 0.08)`
   - `max_weights=(0.03, 0.04)`
   - `max_domain_exposure_threshold=0.04`
2. Increase covariance risk penalty:
   - `covariance_penalty_lambdas=(10.0, 20.0)`
   - keep `concentration_penalty_lambdas` high (e.g., 120.0)
3. Slow adaptation and extend memory:
   - `learning_rates=(0.01, 0.03)`
   - `rolling_windows=(48, 96)`
   - `penalties_lambda=(1.0, 2.0)`

## Week 9 Optimizer Overhaul

### Motivation
Analysis of the Week 8 results revealed four structural issues preventing the constrained optimizer from differentiating itself from the equal-weight baseline:

1. **Sortino ratio as a gradient objective is poorly conditioned.** The ratio form `mean / sqrt(semivariance)` has degenerate gradients when the denominator is small (few or no negative returns in the rolling window). With only 48–96 samples per window, the Sortino estimate is noisy and the gradient landscape is rough — the optimizer chases noise rather than signal.

2. **`uniform_mix = 0.9` neutered the optimizer.** Final portfolio weights were 90% equal-weight and only 10% optimizer-chosen, leaving almost no room for the constrained strategy to express any view.

3. **`steps_per_window = 1` was insufficient.** A single gradient step per time period on a non-convex, ratio-based objective with heavily diluted weights could not accumulate meaningful signal.

4. **Frozen evaluation mode hid the algorithm's adaptive capability.** The holdout froze weights from the end of tuning, testing "how good were the final weights" rather than "how well does the online algorithm adapt going forward."

### Changes applied

#### 1. Mean-downside surrogate objective (`_mean_downside_objective`)
Replaced the Sortino ratio as the optimization target with an additive decomposition:

```
J(w) = mean(r_p) - α·Var(r_p) - β·mean(max(0, -r_p)²)
       - λ_dom · Σ max(0, S_k - L_k)²
       - λ_conc · Σ max(0, w_j - w̄)²
       - λ_cov · w^T Σ w
       + λ_ent · H(w)
```

This captures the same economic intent as Sortino (reward mean return, penalize downside risk) but with well-behaved, bounded gradients. The `α` (variance_penalty) and `β` (downside_penalty) parameters are configurable. The actual Sortino ratio is still computed as an evaluation metric — we just don't optimize it directly.

The `objective` field on `ExperimentConfig` selects between `"mean_downside"` (new default) and `"sortino"` (legacy).

#### 2. Reduced `uniform_mix` from 0.9 to 0.4
The optimizer now controls 60% of the portfolio weight allocation, giving it meaningful room to deviate from equal-weight while still maintaining a 40% diversification floor.

#### 3. Increased `steps_per_window` from 1 to 3
Three gradient steps per time period allows the optimizer to make more progress on the (now smoother) objective before committing to weights for the next realized return.

#### 4. Switched from frozen to online evaluation mode
The holdout now uses `evaluation_modes=("online",)`, meaning the optimizer continues to adapt its weights during the holdout period. This tests the algorithm's actual value proposition: can it improve on equal-weight by learning from incoming data in real time?

### Expected impact
- The smoother objective should produce more stable weight trajectories and reduce oscillation.
- Lower uniform_mix gives the optimizer enough freedom to express meaningful tilts.
- More gradient steps per window should improve convergence quality.
- Online evaluation should show whether the algorithm genuinely adapts or merely overfits tuning data.

### Week 9 full-grid results (post-overhaul)
- Grid search completed in ~73,600 seconds (~20.4 hours) on M2 Air.
- Best config selected: lr=0.01, penalty=1.0, window=96, covariance_penalty=20.0, uniform_mix=0.4.
- Holdout results:

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino | 0.3316 | 0.3234 | -0.0082 |
| Max drawdown | -1.45% | -1.60% | -0.15% |
| Volatility | 0.00226 | 0.00213 | -0.00013 |

- **Diagnosis: optimizer produced near-identical weights to equal-weight baseline.**
  Every market had mean_weight ≈ 0.0038 (= 1/260). Domain exposures were
  indistinguishable from the equal-category-weight baseline.

### Root cause: constraints too loose relative to equal-weight
The combination of settings created a regime where equal-weight already satisfied
every constraint, and the optimizer had no incentive to deviate:
1. `uniform_mix=0.4` locked 40% of weight to equal-weight.
2. `entropy_lambda=0.02` rewarded staying uniform (entropy is maximized at equal weight).
3. `domain_limit=0.06` was never triggered — equal-weight domains peak at ~0.0115.
4. `max_weight=0.03` was never triggered — equal weight per market is 0.0038.
5. `concentration_penalty_lambda=120.0` further penalized any deviation from uniformity.

The return/risk signal from 24 days of noisy prediction market data was too weak
to overcome all these forces pulling toward equal-weight.

### Changes applied
1. **`uniform_mix`: 0.4 → 0.1** — optimizer now controls 90% of allocation, with
   only a 10% diversification floor. The domain penalty, concentration penalty,
   and covariance penalty remain in place to prevent collapse.
2. **`entropy_lambda`: 0.02 → 0.0** — removed the entropy bonus entirely. It was
   actively rewarding the optimizer for staying at equal-weight, counteracting
   the return signal. Diversification is still enforced by the domain penalty,
   concentration penalty, and the remaining 10% uniform floor.
