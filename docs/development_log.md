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

### Week 9 run 3 results (uniform_mix=0.1, entropy=0.0)
- Grid search completed in ~1,233 minutes (~20.6 hours) on M2 Air, 64 stage-1 candidates.
- Best config selected: lr=0.03, penalty=1.0, window=96, covariance_penalty=20.0, uniform_mix=0.1.
- Holdout results:

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino | 0.3277 | 0.3137 | -0.0140 |
| Max drawdown | -1.56% | -1.74% | -0.18% |
| Mean return | 0.000176 | 0.000169 | -0.000006 |
| Volatility | 0.00225 | 0.00211 | -0.00013 |

- **Diagnosis: weights still near-uniform despite uniform_mix=0.1.**
  Every market had mean_weight ≈ 0.0077 (1-market domains) or ≈ 0.0115 (2-market
  domains). The exposure delta chart showed maximum deviations of ±0.003 — the
  optimizer was making only tiny adjustments around equal-weight.

### Root cause: too many markets for the signal strength
With 260 markets and only 24 days of hourly data, three structural problems
prevented the optimizer from learning meaningful tilts:

1. **Extreme dilution.** Each market carries weight ≈ 1/260 = 0.0038. Even a
   large relative tilt (e.g. doubling one weight) produces a portfolio-level
   effect of only 0.4%, invisible against noise.
2. **Massively underdetermined problem.** The optimizer must learn 260 weight
   parameters from ~24 days of noisy returns — far too few observations per
   parameter for the gradient to distinguish signal from noise.
3. **Penalties still dominate.** `concentration_penalty_lambda=120.0` penalizes
   any weight deviation quadratically. With the return gradient being noisy and
   tiny per-market, the penalty gradient always wins, pulling weights back to
   uniform. `covariance_penalty_lambda=10.0/20.0` similarly favors the
   minimum-variance portfolio, which for 260 near-uncorrelated assets is
   approximately equal-weight.
4. **Constraint thresholds irrelevant.** `max_weight=0.03` and `domain_limit=0.06`
   were never binding — equal-weight per market is 0.0038, and domain exposure
   peaks at 0.0115. The optimizer had no constraint pressure to respond to.

### Changes applied (run 4 configuration)

#### 1. Reduced `max_markets` from 260 to 40
The single most impactful change. With 40 markets:
- Equal weight per market = 1/40 = 2.5%, making tilts visible.
- The optimizer learns 40 parameters instead of 260 — much better determined.
- Per-candidate runtime drops ~6.5× (linear in market count for the inner loop).
- Constraint thresholds become meaningful (see below).

#### 2. Rescaled constraint thresholds for a 40-market portfolio
- `max_weight`: (0.03, 0.04) → (0.06, 0.08, 0.10). Equal weight is now 2.5%,
  so these caps at 6-10% allow meaningful concentration without collapse.
- `domain_limit`: (0.06, 0.08) → (0.10, 0.15, 0.20). With fewer markets per
  domain, these thresholds now actually constrain the optimizer.
- `max_domain_exposure_threshold`: 0.04 → 0.12. Feasibility filter adjusted to
  match the new domain limits.

#### 3. Reduced penalty strengths
- `concentration_penalty_lambda`: 120.0 → search over (5.0, 10.0, 20.0).
  At 120 the penalty gradient overwhelmed the return signal; at 5-20 it still
  discourages extreme concentration but lets the optimizer express views.
- `covariance_penalty_lambda`: (10.0, 20.0) → (1.0, 5.0, 10.0).
  Lower values let the optimizer take on some correlated risk when the return
  signal justifies it.

#### 4. Widened the hyperparameter grid
- `learning_rates`: (0.01, 0.03) → (0.01, 0.05, 0.1). Higher LRs may help
  the optimizer move faster given the short data window.
- `penalties_lambda`: (1.0, 2.0) → (0.5, 1.0, 2.0). Lower domain penalty
  weight gives more freedom.
- `rolling_windows`: (48, 96) → (24, 48, 96). Shorter windows capture more
  recent signal, which may matter for fast-moving prediction markets.
- `uniform_mixes`: (0.1,) → (0.0, 0.1). Includes a fully optimizer-controlled
  variant alongside the 10% floor.

#### 5. Two-stage search retained
Stage 1 coarse grid is now 3×3×3×3×3×3×3×1×1×2 = 4,374 combinations (before
coarse subsetting). The two-stage approach prunes this to a manageable set.

## Optuna Run 1 and Pruner Fix

### What happened on first Optuna run (100 trials)
- **Result:** 16 completed, 84 pruned. Total time ~1.2h. Best trial had holdout Sortino 0.043.
- **Weights unchanged:** The "best" config produced a portfolio with domain exposures all ~2.5% (1/40). Every market had mean_weight ≈ 0.025. So the optimizer was effectively outputting equal-weight despite different hyperparameters.

### Why weights looked equal (two causes)
1. **Structural (data regime):** With 40 markets and ~28 days of hourly data, the return signal is weak and the penalties (domain, concentration, covariance) pull weights toward uniform. This is the same finding as earlier grid runs — not a bug in Optuna.
2. **Pruning too aggressive:** We used `MedianPruner(n_warmup_steps=2)`. Pruning started after **2 walk-forward folds** out of ~210 (~1% of the run). Early-fold Sortino is very noisy, so the pruner treated most trials as "below median" and pruned them. The 16 "completed" trials were mostly the ones that got lucky in the first few folds, not necessarily better configs. So the search did not meaningfully explore the space.

### Changes made
- **Pruner:** `n_warmup_steps` is now set from the number of folds: warmup ≈ 10% of folds, minimum 5, cap 25 (e.g. ~21–25 folds). We only prune after enough folds that the reported Sortino has some signal. Expect many more completed trials on the next run.
- **Covariance diagnostics:** Guard added so we only compute category returns when the category has at least one asset (avoids "Mean of empty slice" warning).

### Data granularity and grid (later run)
- **10-minute price history:** Pipeline and data build now use `history_fidelity=10` so we get ~6× more observations per market for the same calendar window. This improves the T vs N regime for the optimizer. With 10-min data, `rolling_window` is in 10-min steps (e.g. 144 ≈ 1 day, 288 ≈ 2 days). Walk-forward steps are scaled when fidelity ≤ 10: `walkforward_train_steps=1440`, `walkforward_test_steps=288` (~10 days train, 2 days test per fold) so fold count stays similar to the hourly setup.
- **Wider Optuna search:** Hyperparameter ranges were widened so Optuna can explore more of the space:
  - `learning_rates`: (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
  - `penalties_lambda`: (0.25, 0.5, 1.0, 2.0)
  - `rolling_windows`: (24, 48, 96, 144, 288) [10-min steps]
  - `domain_limits`: (0.08, 0.12, 0.18, 0.25)
  - `max_weights`: (0.04, 0.06, 0.10, 0.15)
  - `concentration_penalty_lambdas`: (2.0, 5.0, 10.0, 20.0, 50.0)
  - `covariance_penalty_lambdas`: (0.5, 1.0, 5.0, 10.0)
  - `covariance_shrinkages`: (0.02, 0.05, 0.10)
  - `entropy_lambdas`: (0.0, 0.01, 0.02)
  - `uniform_mixes`: (0.0, 0.05, 0.1, 0.2)

## How to improve performance (prioritized)

1. **Tune the mean-downside objective** — Make `variance_penalty` and `downside_penalty` searchable in Optuna. Lower downside_penalty lets the optimizer care more about mean return and less about penalizing losses; can help it tilt toward higher-return assets instead of collapsing to equal-weight. (Implemented: optional `variance_penalties` / `downside_penalties` tuples in config.)

2. **More gradient steps per window** — Increase `steps_per_window` from 3 to 5 or 8 so each time step does more optimization work before committing weights. (Implemented: pipeline uses 5.)

3. **Try Adam instead of SGD** — Use `torch.optim.Adam` for the weight logits; often more stable and can escape flat regions better than SGD. (Implemented: optional `optimizer_type="adam"` in config.)

4. **Fewer markets** — Reduce `max_markets` to 25–30. Fewer parameters with the same T gives a better-determined problem so the return gradient can win over the penalties sometimes.

5. **Loosen penalties on the margin** — Ensure the search grid includes low concentration_penalty_lambda and covariance_penalty_lambda (e.g. 0.5, 1.0) so some trials are more return-seeking; feasibility filter still enforces domain exposure for reporting.

6. **Longer rolling window** — Try rolling_window ≥ 144 (1 day in 10-min steps) so the gradient is estimated from more data and is less noisy.

7. **Bootstrap and stress-tests** — Add bootstrap confidence intervals on holdout Sortino so you can say whether constrained vs baseline is statistically different; add a Monte Carlo stress-test (e.g. correlated domain shock) to show that constraints improve tail risk even when average Sortino is similar.

8. **Two-stage: select then optimize** — First select a smaller set of markets (e.g. by liquidity or momentum), then optimize weights over that set so the optimizer has fewer, more impactful knobs.

## Current Project State (Latest Pipeline Configuration)

### What is now different from earlier runs
- **Optimization engine:** still online rolling-window optimization, but inner updates now default to **Adam** (`optimizer_type="adam"`) instead of only SGD.
- **More inner optimization work:** `steps_per_window` increased to **5** (from 3 in prior runs), so each window gets more than one quick update before advancing.
- **Objective flexibility added:** Optuna now tunes not only structural constraints/hyperparameters but also the mean-downside objective weights:
  - `variance_penalties=(0.5, 1.0, 2.0)`
  - `downside_penalties=(1.0, 2.0, 3.0)`
- **Data granularity changed:** price history uses **10-minute fidelity** (`history_fidelity=10`) instead of hourly, increasing observations per market by roughly ~6x for the same calendar span.
- **Walk-forward scaling updated for 10-minute data:** walk-forward blocks are scaled to keep fold counts in a comparable range:
  - `walkforward_train_steps=1440` (~10 days)
  - `walkforward_test_steps=288` (~2 days)
- **Wider Optuna search space:** search ranges now include broader learning-rate, penalty, window, and mixing settings to encourage non-uniform solutions when signal supports them.

### Practical impact of these changes
- **Pros:** richer data, more expressive tuning, and a more robust inner optimizer make it more likely to discover meaningful tilts when signal exists.
- **Tradeoff:** each completed trial is substantially slower than earlier hourly/SGD runs because:
  1) there are more timestamps (10-min data), and  
  2) each timestamp performs more gradient work (`steps_per_window=5`).
- **Observed behavior so far:** early completed trials now show wider Sortino dispersion than the previous "all-near-equal" regime, which indicates the search is exploring materially different parameter regimes rather than collapsing immediately to one solution.

### Interpretation note for ongoing runs
Higher trial-to-trial Sortino variation at this stage is expected and desirable. It suggests the optimization landscape is no longer effectively flat under the current settings, and Optuna is testing genuinely different risk/return trade-offs rather than repeatedly reproducing near-identical equal-weight behavior.

## Round 4 / Round 5 Post-Mortem and Round 6 Design

### Correction notice

An earlier version of this section claimed that every Round 4 / Round 5 pod tied or lost to baseline. That conclusion was derived from the `baseline_holdout_sortino` field inside `data/processed/*_constrained_best_metrics.json`, which is recomputed inside `run_optuna_search` using the same walk-forward slicing + top-K-bagged alignment as the constrained model — not the natural equal-weight-on-holdout comparison. The per-pod `docs/week9_diagnostics_report.md` files on each pod branch give the right comparison, and the numbers below supersede the earlier table.

### Summary of finished pods (from per-pod diagnostics reports)

| Pod | Branch | Baseline Sortino | Constrained Sortino | Δ Sortino | Baseline DD | Constrained DD | Δ DD | Config highlight |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **I4** | `cloud-runs-I4` | +0.0963 | +0.1040 | **+0.0077** | −29.70% | −28.54% | **+1.16 pp** | mom 20/5d, rw=24 (Round 4 best) |
| K4 | `cloud-runs-K4` | +0.0618 | +0.0615 | −0.0003 | −29.12% | −28.03% | **+1.09 pp** | mom 20/5d, rw=96 |
| L4 | `cloud-runs-L4` | +0.1123 | +0.0600 | **−0.0523** | −22.38% | −24.89% | −2.51 pp | mom 25/10d (aggressive momentum hurt) |
| M4 | `cloud-runs-M4` | +0.0179 | −0.0301 | **−0.0480** | −6.12% | −17.39% | **−11.27 pp** | no momentum (baseline flat, constrained over-concentrated) |
| **Q5** | `cloud-runs-Q5` | +0.0751 | +0.0808 | **+0.0057** | −35.10% | −32.34% | **+2.75 pp** | LR sweep 0.04–0.12 on Pod I recipe |
| G-seed42 | `cloud-runs-G-seed42` | +0.0238 | −0.0102 | **−0.0340** | −7.47% | −7.72% | −0.25 pp | G recipe, seed=42 |

### What the results actually tell us

**1. Two pods modestly beat baseline; three improved drawdown.** I4 won on both Sortino (+0.0077) and DD (+1.16 pp). Q5 also won on both (+0.0057, +2.75 pp). K4 tied Sortino but improved DD by +1.09 pp. Three of five Round-4/5 pods reduced holdout drawdown — the sizing/momentum story is adding *some* risk-adjusted value, it just isn't large in absolute terms, and absolute drawdowns are still severe (−28% to −32%) because the underlying universe simply has −29%-to-−35% baseline drawdowns to begin with.

**2. Seed noise is an order of magnitude larger than any win we've logged.** Pod G2 (Round 2): Δ = +0.0019. Pod G-seed42 (identical recipe, seed=42): Δ = **−0.0340**. |ΔΔ| ≈ 0.036 Sortino from seed alone. Every single-seed Δ we have — including I4's +0.0077 — is **inside that noise floor**. The sign of I4's result is consistent with other sizing-aware pods (Q5, K4 all improved DD; I4 and Q5 both improved Sortino), which is the strongest *indirect* evidence that there's a real effect underneath the noise. But to make a direct claim we need σ(Δ) measured on a recipe we actually want to validate. That is Pod S4 (revised: multi-seed I4, not multi-seed G).

**3. Drawdown is a universe property as much as a model property.** L4's baseline DD was −22%; M4's was −6%. M4 in particular warns against reading "−17% constrained DD" as an improvement — the baseline on that no-momentum universe was −6%, and the constrained model made it **worse** by 11 pp because it over-concentrated. The implication for Round 6: sizing caps need to be evaluated relative to the *constrained*-vs-*baseline* DD delta on the same universe, not against absolute DD targets. And tightening caps only helps if the optimizer was actually over-concentrating in the first place — I4/Q5/K4 evidence says it was.

**4. A richer objective beats a broader hyperparameter search.** Round 5 (Pod Q5) widened the LR ceiling to 0.12 while keeping the objective identical. Q5's best LR pinned at 0.0625 — inside the new window, not at the ceiling — and Q5 improved on I4 only on DD (+2.75 pp vs +1.16 pp), while Sortino Δ shrank (+0.0057 vs +0.0077). More LR granularity isn't the lever. The objective is. Pod S1 exposes all eight risk-aware levers simultaneously for the first time:

```python
# src/constrained_optimizer.py — existing inner loop (abbreviated)
mean_return - variance_penalty_t * variance
             - downside_penalty_t * downside_semivar
             - covariance_penalty_lambda_t * covariance_penalty
             - concentration_penalty * lambda_conc
             - domain_exposure_excess * lambda_penalty
```

All terms are active; what's been missing is the ability to sweep them from the CLI. The teammate's `stock-PM-combined-strategy` branch did sweep these, hit Δ = +0.1018 (well outside the noise floor), with DD = −9.4% on their universe. That's the benchmark S1 is targeting.

### Code changes landed on `cloud-runs-R6`

1. **CLI overrides on `script/polymarket_week8_pipeline.py`** — eight new flags mirroring the `--rolling-windows` / `--lr-values` pattern. Every flag overrides an existing `ExperimentConfig` field that was already consumed by the inner loop:

   - `--variance-penalty-values`, `--downside-penalty-values` enable Optuna search over the `mean − λ_var·var − λ_down·semivar` objective.
   - `--covariance-penalty-lambdas`, `--covariance-shrinkage-values` expose the covariance penalty and shrinkage target.
   - `--domain-limit-values`, `--max-weight-values`, `--concentration-penalty-lambdas` expose position-sizing caps.
   - `--seed-override` supports the multi-seed robustness study (Pod S4).

   Validation and the "apply after `--reduced-search`" semantics match the existing `--lr-values` exactly, so `--reduced-search` plus any Round 6 flag widens only the explicitly swept lever while keeping the reduced-search narrowings on everything else. Default behavior (no flag) is bit-identical to pre-Round-6 runs — no risk to in-flight pods.

2. **Port of `src/pm_risk_overlay.py` and `src/equity_signal.py`** — cherry-picked verbatim from `origin/stock-PM-combined-strategy` (teammate branch left untouched, per the user's explicit instruction). Provides `build_equity_domain_tilt_multiplier` (SPY-informed domain tilts), `pm_category_spread_returns` + `top_negative_correlation_pairs` (zero-investment PM-category pairs trading), and the equity-signal diagnostics they depend on. Also four `data/external/*_template.csv` files.

3. **Two new post-hoc overlay evaluators** modeled on the existing `script/posthoc_alpha_blend.py`:

   - `script/posthoc_overlay_tilt.py` applies the equity-domain tilt at a sweep of strengths on a domain-equal baseline, using actual PM per-asset returns from `_price_history.csv`. Used by Pod S2. Requires `yfinance` + network.
   - `script/posthoc_overlay_spread.py` identifies negatively correlated domain pairs from `_category_correlation.csv`, computes the zero-investment spread, sweeps `(max_pairs, spread_lambda)`. Used by Pod S3. Pure-CPU, no network. Smoke-tested on `week8` artifacts: interior argmax at `(max_pairs=5, λ=0.1)` improves on pure baseline by +0.0014 Sortino (real-but-small signal).

### Round 6 experiment design (5 CPU pods) — revised after corrected post-mortem

See [docs/cloud_runbook.md §16](cloud_runbook.md) for full recipes and CLI commands. Summary:

- **S4 — Pod I4 recipe × 5 seeds `{3, 7, 101, 202, 303}`** (noise-floor diagnostic, promoted to priority 1). Measures σ(Δ) around our individual-seed winner. Without this, no other pod's result is interpretable. Swapped from the G recipe (a Round-5 loser) to the I4 recipe (our only win we're trying to validate).
- **S1 — full risk-aware objective sweep** (highest upside, priority 2). Sweeps eight levers simultaneously with sizing bracketed between I4's defaults (≈0.10) and the teammate's optimum (0.04) rather than strictly tighter than both, and keeps I4's winning `rw=24` in the rolling-window search. Targets the teammate's Δ = +0.1018 benchmark.
- **S5 — sizing-frontier grid on Pod I4** (priority 3). 3×3×3 grid on `max_weight × domain_limit × concentration_penalty_lambda`, Optuna-sampled. Traces the Sortino/DD frontier between I4's defaults and the teammate's tight corner; replaces the earlier single-pin design.
- **S2 — post-hoc equity-domain tilt sweep** (priority 4, runs on S1/S5 outputs). Tests whether the SPY-informed tilt adds value on top of a domain-equal portfolio.
- **S3 — post-hoc PM-category spread sweep** (priority 5, runs on S1/S5 outputs). Tests whether zero-investment pairs trading on the top negatively correlated PM categories adds risk-adjusted value.

Priority order: **S4 > S1 > S5 > S2 > S3**. S2 and S3 are minutes-long CPU post-hoc evaluators and can piggyback on whichever pod frees up first.

### Why the design was revised

- **S1 was strangling the universe.** The first-draft S1 pinned `max_weight ∈ {0.03,0.04,0.05,0.06}` — tighter than any of our Round-4/5 winners, which used the default ≈0.10. The revised sweep `{0.04,0.06,0.08,0.10}` brackets both ends. Similarly `domain_limit` widened from `{0.06..0.12}` to `{0.08,0.12,0.16}`, and `rolling_windows` now includes `rw=24` (I4's winner) which the earlier `{96,144,288}` had dropped. `--momentum-screening` is back on, since M4 (no-momentum) was the worst pod on DD by a wide margin.
- **S5 was a single point.** First-draft S5 pinned `max_weight=0.04, domain_limit=0.08, cpl=2.0` to the teammate's optimum. On our universe, if that single point fails, we learn "doesn't work" with no actionable gradient. Revised S5 is a 27-cell grid (3×3×3) that traces the whole frontier with `rw=24` held fixed so sizing is the only lever moving.
- **S4 was measuring noise on the wrong recipe.** First-draft S4 ran five seeds of the G recipe — a known Round-5 loser (G-seed42: Δ = −0.034). Measuring σ on a losing recipe doesn't validate our winner. Revised S4 runs five seeds of the **I4** recipe. If σ(Δ) ≤ 0.003, I4's +0.0077 is ~2.5σ signal; if 0.003 < σ ≤ 0.008, the 5-seed mean Δ becomes the reportable; if σ > 0.008, I4 is indistinguishable from noise and only S1 + the teammate's benchmark remain live.

### Success criteria (revised)

1. **S4 produces a defensible σ(Δ).** The single most important number Round 6 needs to deliver. It gates the interpretation of every other pod's result. Reportable as `mean(Δ) ± std(Δ)` across the 5 seeds on I4's recipe.
2. **S1 produces Δ ≥ +0.02 Sortino vs its own diagnostic-report baseline** (well outside the 0.036 Round-5 noise floor and clearly separated from I4's +0.0077). Combined with S4's σ, this gives us either a confirmed alpha-vs-baseline claim or a precise upper bound.
3. **S5 traces a visible Sortino/DD frontier.** Best cell should improve I4's −28.5% DD by ≥ +8 pp while keeping Sortino ≥ +0.08. If the whole grid reads worse than I4 on both metrics, sizing is not the lever.
4. **S2 and/or S3 produce an interior Sortino argmax at non-zero overlay strength beating the no-overlay row by ≥ +0.01.** If yes, the overlay graduates to Round 7 integration inside `_run_online_pass`; if no, the overlay direction is closed.

### What Round 6 explicitly does not test

- **Kelly × tight caps.** The Round 7 K10D-tight experiment (re-run `polymarket_week10_kelly_pipeline.py` with `--max-weights 0.04 --concentration-penalty-lambdas 2.0` and dynamic copula on) requires a GPU pod — not in the current five-pod budget. Held for the next round.
- **Regime-dependent penalties.** A mixture objective switching penalty strengths on SPY z-score would use `src.equity_signal.compute_risk_regime_zscore` (ported in §14.8). Candidate for Round 7 if Round 6 validates the equity signal via Pod S2.
- **Tilt / spread inside the optimizer.** If Pods S2 / S3 clear their success bar, fold the relevant overlay into `_run_online_pass` so it's optimized jointly with the penalties rather than applied post-hoc. Until then, post-hoc evaluation is the right test.
