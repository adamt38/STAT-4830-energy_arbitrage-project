# Week 4 Self-Critique (Prototype Baseline)

## OBSERVE

The repo is now aligned to the Polymarket project at the implementation level. We have a working end-to-end script (`script/polymarket_week8_pipeline.py`) that builds a cached dataset, computes a baseline, runs a constrained optimizer grid, and generates figures. This is our Week 4 prototype baseline, and the constrained model currently underperforms on risk metrics.

---

## ORIENT

### Strengths
- **Execution-first pivot succeeded.** We now have code artifacts directly matching the proposal objective and a reproducible pipeline.
- **Baseline and constrained comparison exists.** This is enough to present a credible flash-progress story with evidence.
- **Data quality checks are in place.** We can quantify missingness, monotonicity, and domain coverage for debugging.

### Areas for Improvement
1. **Constrained model underperforms baseline.** Current penalty and update settings are not yet giving risk improvements.
2. **Domain imbalance in selected markets.** The first data slice remains crypto-heavy, reducing the value of cross-domain constraints.
3. **Evaluation protocol is still light.** We need cleaner train/validation walk-forward splits to avoid tuning on the full series.

### Critical Risks / Assumptions
We assume tag-based domain mapping is reliable enough for constraints, but tag granularity and ambiguity can distort exposure accounting. We also assume CLOB time-series resolution is consistent across selected tokens; sparse or uneven histories may bias results.

---

## DECIDE

### Concrete Next Actions
1. Rebuild market universe with explicit per-domain quotas before optimization.
2. Expand hyperparameter grid and add learning-rate scheduling for stability.
3. Add walk-forward evaluation and report out-of-sample baseline vs constrained metrics.

---

## ACT

### Resource Needs
- Confirm preferred Polymarket endpoint contracts and field conventions from your team notes (especially historical price pull settings).
- Optional: lightweight helper doc for manual domain overrides on ambiguous tags.
- Time allocation: one focused iteration cycle on model stabilization before Week 9 slides.

---

## Week 8 Addendum (Current State)

### Updated Observe
- The Week 8 pipeline now runs end-to-end with strict binary `Yes/No` market filtering and frozen holdout mode.
- Latest full run retained 177 markets after fallback from 30-day to 24-day minimum history.
- Baseline currently outperforms constrained model on matched holdout horizon:
  - baseline holdout Sortino: 0.3508 vs constrained holdout Sortino: 0.2018
  - baseline holdout max drawdown: -2.73% vs constrained holdout max drawdown: -5.01%
- Covariance diagnostics show constrained portfolio risk is still too concentrated (`variance_ratio_constrained_vs_baseline = 14.06`).

### Updated Orient
#### What improved
- Removed a lookahead leakage bug in price matrix construction.
- Enforced consistent binary-market semantics by requiring explicit `Yes` token.
- Strengthened robustness with typed no-survivor exceptions and safer API numeric parsing.
- Clarified evaluation by making frozen holdout available and setting it as default reporting mode.

#### Remaining weaknesses
1. Constrained strategy still fails to deliver out-of-sample risk-adjusted gains over baseline.
2. Exposure controls are not yet tight enough; realized max-domain weight can still spike above target.
3. Current data-selection rules often require history fallback (24-day), reducing long-history coverage.

### Updated Decide (next experiment cycle)
1. Tighten risk controls:
   - `domain_limits=(0.06, 0.08)`
   - `max_weights=(0.03, 0.04)`
   - `max_domain_exposure_threshold=0.04`
2. Penalize covariance risk more strongly:
   - `covariance_penalty_lambdas=(10.0, 20.0)`
   - keep high concentration penalty (e.g., 120.0)
3. Reduce overreaction and improve stability:
   - `learning_rates=(0.01, 0.03)`
   - `rolling_windows=(48, 96)`
   - `penalties_lambda=(1.0, 2.0)`

### Updated Act
- Next run should use frozen-only holdout and a narrower, higher-quality grid (fewer but more defensible candidates).
- Success criterion for next cycle: constrained holdout should beat baseline on at least one of:
  - Sortino ratio (primary),
  - max drawdown (secondary),
  - while not worsening volatility materially.

---

## Week 9 Addendum (Optimizer Overhaul)

### Updated Observe
- The Week 8 constrained optimizer consistently failed to beat the equal-weight baseline on holdout (Sortino 0.20 vs 0.35, drawdown -5.0% vs -2.7%).
- Root-cause analysis identified four structural issues in the optimization setup, not just hyperparameter tuning problems:
  1. The Sortino ratio used as the gradient objective has degenerate gradients when downside deviation is small and is noisy over short rolling windows (48–96 samples).
  2. `uniform_mix = 0.9` left the optimizer controlling only 10% of portfolio weight — not enough to express any meaningful view.
  3. `steps_per_window = 1` gave a single gradient step per period on a non-convex ratio objective, insufficient for convergence.
  4. Frozen evaluation mode tested static end-of-tuning weights rather than the algorithm's adaptive capability.

### Updated Orient

#### What improved this cycle
- **Replaced Sortino with mean-downside surrogate objective.** The new objective decomposes the Sortino's intent into additive terms: `mean(r) - α·Var(r) - β·semivariance(r)`. This eliminates the ratio form whose gradient degrades when the denominator is small. Gradients are now bounded and well-behaved regardless of window content. The Sortino ratio is still used as an evaluation metric.
- **Reduced `uniform_mix` from 0.9 to 0.4.** The optimizer now controls 60% of portfolio allocation, giving it real room to express tilts while maintaining a 40% diversification floor.
- **Increased `steps_per_window` from 1 to 3.** More gradient steps per period on the smoother objective should improve convergence quality.
- **Switched to online evaluation mode.** The holdout now tests the algorithm's actual adaptive behavior rather than frozen end-of-tuning weights.

#### Remaining weaknesses
1. **No empirical validation yet.** These changes are theoretically motivated but have not been run — the next pipeline execution will reveal whether they actually improve holdout metrics.
2. **Variance and downside penalty weights (α=1.0, β=2.0) are initial guesses.** These may need tuning; if β is too high the optimizer may become overly conservative.
3. **`uniform_mix = 0.4` may be too aggressive.** If the optimizer's learned weights are poor, a 60% allocation to them could hurt performance. May need to search over `uniform_mixes=(0.3, 0.4, 0.5)`.
4. **Online mode introduces a new risk: overfitting to recent holdout data.** The optimizer adapts during holdout, which is more realistic but also means poor hyperparameters could cause weight drift.
5. **The domain penalty still operates on tag labels, not correlation structure.** Two markets in different domains can be highly correlated. The covariance penalty partially addresses this, but a correlation-aware domain grouping would be more principled.

### Updated Decide (next experiment cycle)
1. Run the pipeline with the new settings and compare holdout metrics against the Week 8 baseline.
2. If the optimizer still underperforms, investigate:
   - Whether `uniform_mix` should be searched as a grid parameter (0.3, 0.4, 0.5).
   - Whether `variance_penalty` and `downside_penalty` need adjustment.
   - Whether `steps_per_window=5` would help further.
3. If the optimizer improves on at least one metric (Sortino or drawdown) without worsening the other, declare this iteration a success and focus on robustness.

### Updated Act
- Run `python script/polymarket_week8_pipeline.py` with the new config and capture results.
- Success criterion: constrained holdout should beat matched-horizon baseline on Sortino ratio or max drawdown.
- Secondary criterion: weight trajectories should be smoother (less oscillation) than the Week 8 run.
- Document results in `docs/week9_diagnostics_report.md` after the run completes.
