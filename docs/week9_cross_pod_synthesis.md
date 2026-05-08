# Week 9 Cross-Pod Synthesis (B, C, D, F)

This document synthesizes the four Week 9 cloud experiments (`week8_B`, `week8_C`,
`week8_D`, `week8_F`) into a single comparative analysis. The auto-generated
`docs/week9_diagnostics_report.md` is per-pod and is overwritten by every
pipeline run — this file is the cross-pod headline that survives subsequent runs.

Companion artifacts (one set per pod, all on `cloud-runs-X` branches):

| Pod | Constrained metrics JSON | Per-pod report (on its branch) |
|---|---|---|
| B | `data/processed/week8_B_constrained_best_metrics.json` | `docs/week9_diagnostics_report.md` on `cloud-runs-B` |
| C | `data/processed/week8_C_macro_explicit_constrained_best_metrics.json` | `docs/week9_diagnostics_report.md` on `cloud-runs-C` |
| D | `data/processed/week8_D_macro_both_constrained_best_metrics.json` | `docs/week9_diagnostics_report.md` on `cloud-runs-D` |
| F | `data/processed/week8_F_constrained_best_metrics.json` | `docs/week9_diagnostics_report.md` on `cloud-runs-F` |

Pods **A** (rescale, no ETF) and **E** (three back-to-back macro modes) were
still in flight at the time of writing and are therefore **not yet included**.
A and E should be added to the table below when they finish; their addition is
not expected to overturn the headline conclusion.

---

## 1. Bottom line

**All four pods underperform the equal-weight baseline on the held-out 20% of
data, on both Sortino ratio and maximum drawdown.** Sortino delta ranges from
−0.020 to −0.060; max drawdown delta ranges from −0.24% to −4.66%. No pod beat
the baseline on either headline metric. The interpretation that follows is
therefore a *negative result with diagnostic value*, not an alpha claim.

---

## 2. Headline comparison (sorted by Sortino delta)

| Pod | Macro | ETF λ | Macro λ | Sortino Δ | MaxDD Δ | Open Sortino Δ | Closed Sortino Δ | Realized max-domain |
|---|---|---|---|---|---|---|---|---|
| **F** (joint) | both | 0.10 | 4.61 | **−0.0202** ★ | −0.72% | −0.088 | −0.005 | 0.0263 |
| **D** | both | 0.50 | 0.47 | −0.0206 | **−0.24%** ★ | **−0.137** | −0.001 | 0.0254 |
| **B** | rescale | 0.05 | 0.00 | −0.0342 | −4.66% | −0.034 | −0.028 | 0.0255 |
| **C** | explicit | 0.50 | 2.11 | −0.0597 | −0.45% | **+0.055** | −0.089 | 0.0254 |

★ = best in column. "Realized max-domain" is the largest single-domain weight
the constrained portfolio actually held; for reference, exact equal-weight is
`1/40 = 0.0250`. **All four pods landed within 5% of equal-weight.**

For absolute holdout numbers (Sortino, MaxDD, mean return, volatility) and the
baseline-on-holdout values used to compute the deltas above, see each pod's
`week9_diagnostics_report.md` on its `cloud-runs-X` branch.

---

## 3. Per-pod takeaways

### F — joint macro search (smallest negative delta)
F was given complete freedom to pick any macro mode. It chose
`macro_integration="both"` with the **highest** explicit macro weight (4.61, vs
0.0–2.1 elsewhere) and the **lowest** ETF weight among the ETF-on pods (0.10,
vs 0.50). It also picked a **concentration penalty 7× higher than B/C/D**
(`λ_conc = 21.3` vs ~2.8). The combination forces the optimum to sit very
close to equal-weight, and that produced the smallest holdout loss.

**Interpretation:** when free to choose, the search collapsed toward the
"do nothing different" solution. Even that loses to actual equal-weight.

### D — `macro_integration=both` with high ETF
D had the **largest open-vs-closed Sortino swing** (−0.137 during US equity
hours, +0.0 during closed hours). With `λ_ETF = 0.5` and aggressive macro
penalties, D actively rotates portfolio weights toward SPY/QQQ/XLE during US
hours. That rotation hurt performance during the holdout window.

**Interpretation:** high ETF tracking weight is net-negative *during the hours
when it's most active*.

### B — `macro_integration=rescale`, near-zero ETF (worst performer)
B has a markedly worse holdout drawdown than every other pod (−13.73% vs
−8 to −9.6% elsewhere). The top contributor is **Will Dortmund win the
Bundesliga? at −290%** of total return — a single binary outcome going against
B's specific 2085-bar holdout window. The other pods (different ingest times →
different holdout windows) didn't catch this tail.

**Interpretation:** turning off ETF/macro integration didn't help. The
"rescale" macro alone is the weakest formulation, AND B drew a worse holdout
window. Without ETF tracking pulling weights toward broad equity, B leaned
harder on idiosyncratic positions and got burned by one resolution.

### C — `macro_integration=explicit`, high ETF (most counterintuitive)
C is the **only pod where the constrained model wins on the open-hours
subset** (Sortino +0.090 vs baseline +0.036, delta +0.055). But the closed
hours dominate: constrained +0.003 vs baseline +0.093 (delta −0.089). Because
closed bars outnumber open bars ~5.7:1, the headline result is the largest
negative Sortino delta in the set (−0.060).

**Interpretation:** macro/ETF coupling can be net-positive during US hours
when explicit macro weight is high enough to actually rotate the portfolio.
But ~85% of the data is closed hours, where the macro signals are stale and
the constraint structure leaks return relative to equal-weight.

---

## 4. Cross-cutting findings

### 4.1 Every pod collapsed to near-equal-weight
`max_domain_exposure` across all pods: 0.0254, 0.0254, 0.0255, 0.0263.
Equal-weight is exactly 0.0250 (= 1/40). The optimizer is making sub-percent
tilts around equal-weight in every configuration. The combined entropy +
concentration + covariance + ETF-tracking penalties crowd out the return
objective.

### 4.2 The data has effectively one risk factor
`top_eigenvalue_share` across pods: 0.83, 0.83, 0.84, 0.86. One principal
component captures 83–86% of cross-market variance. This is a property of the
data, not the model. With ~1 effective independent risk dimension, equal-weight
already exploits the only available bet. There is essentially no room for a
40-market constrained model to differentiate.

### 4.3 The same five markets drive returns across all pods
Top contributors recur in every pod:

- **Felix Protocol FDV** (positive, +85% to +257% of total return)
- **Claude 5** (negative, −44% to −83%)
- **Colombia election** (positive, +38% to +151%)
- **Fed rate decision** (negative, −30% to −100%)
- **Colorado midterm** (positive, +19% to +79%)

Because all pods held essentially equal-weight, they all caught the same
binary outcomes. The "model" had nothing to do with which markets drove
returns — these resolutions were the same across all four runs.

### 4.4 Hyperparameter search converges, but to wrong region
B and C both selected **Sobol trial #65** as best, producing identical
hyperparameters for everything except the macro-mode-specific terms. D
selected #30, F selected #63. Two independent runs landing on the same Sobol
index suggests the cross-validation objective is well-defined enough to
converge — but its optimum doesn't generalize to the holdout.

### 4.5 Tuning-to-holdout overfit gap is consistent and large
Best Optuna trial typically scored 0.08–0.13 Sortino in walk-forward CV but
landed at 0.006–0.033 on holdout — a 3–5× drop in every pod. Standard
selection bias from 100 trials × walk-forward CV plus regime shift between
the first 80% (tuning) and last 20% (holdout). **The CV objective is a poor
predictor of holdout Sortino on this dataset; only the held-out 20% counts.**

---

## 5. Why the baseline is hard to beat here (mechanism)

Combining the findings above:

1. With one dominant principal component, expected return is essentially
   `μ × wᵀv₁`, where `v₁` is the top eigenvector. Equal-weight already gives
   `wᵀv₁ ≈ 1/√N` projection; any tilt that improves this also concentrates
   risk in `v₁`, which our concentration/covariance penalties forbid.
2. Five binary contracts dominate realized returns. Their *resolution
   directions* are not predictable from the macro/ETF features the model has
   access to. Equal-weight's positions on these markets are roughly the same
   as the constrained model's positions, so the constrained model gains
   nothing on the dominant return drivers.
3. The macro/ETF penalty, if active during US hours, *actively pushes weights
   away* from the equal-weight optimum to track equity factors that don't
   forecast prediction-market returns at this frequency. This costs Sortino,
   especially during closed hours where the penalty is stale but its prior
   weight rotations remain in the portfolio state.

---

## 6. What to try next (prioritized)

The recommendations below are ordered by expected information gain per
compute hour. Items 1–3 are cheap re-runs of the existing pipeline with
different settings; items 4–6 require code changes; items 7–8 require data
or universe changes.

### 6.1 Cheap re-runs (no code changes; ~1 day each)

**(1) Drop ETF tracking entirely and tighten the macro search range.** The
Open vs Closed split shows ETF tracking is the most consistently harmful
term across pods. Run a single experiment with `--etf-tracking` flag *off*
and `lambda_macro_explicit ∈ {0, 0.5, 1.0}` instead of the current much wider
range. Hypothesis: removes the dominant negative term and gives the macro
component a fair test.

**(2) Reduce search space dimensionality.** Current Optuna search has 14+
hyperparameters with 100 trials. Sobol coverage of a 14-dim cube with 100
points is sparse. Fix the parameters that are not informative (entropy,
covariance shrinkage, uniform mix all picked similar values across pods) and
search only the 4–5 parameters that actually varied (`lr`, `domain_limit`,
`lambda_concentration`, `regime_k`, `lambda_macro_explicit`). With fewer
dimensions, 100 trials becomes a denser sweep and the best-trial
overfit gap should shrink.

**(3) Tighter holdout fraction or rolling-origin holdout.** Current
`holdout_fraction=0.2` gives ~2100 bars per holdout. Try 0.3 to widen the
out-of-sample window, or split the holdout into multiple rolling origins
(say 5 × 4% windows) and report mean ± std Sortino. This addresses the "B
got a bad window" problem and gives genuine confidence intervals on the
delta.

### 6.2 Modest code changes (~2–4 hours each + 1 day re-run)

**(4) Top-K bagging instead of single best.** Instead of using the single
best Optuna trial's weights, average the weights from the top-5 (or top-10)
trials by walk-forward CV Sortino. This is the standard fix for selection
bias from many-trial searches. Should reduce the 3–5× tuning-to-holdout
overfit gap to ~2×.

**(5) Walk-forward holdout with weight-state continuity.** The current
holdout is a single fixed slice. Replace it with a rolling-origin holdout
where the model's weight state from the previous fold continues forward.
This is closer to how the strategy would actually run live and gives a more
honest measure of out-of-sample Sortino over varied market regimes.

**(6) Per-domain or per-market shrinkage to baseline.** Add a hyperparameter
`α ∈ [0, 1]` that interpolates between the constrained model's weights and
equal-weight: `w_final = α · w_constrained + (1 - α) · w_baseline`. Optuna
searches `α`. Hypothesis: even with constraints the model adds *some* signal
on average, but it's swamped by noise; 50/50 blending should keep the
upside while damping the noise. This is essentially shrinkage-toward-uniform
applied at the portfolio level rather than the weight level.

### 6.3 Larger structural changes (~1 week)

**(7) Expand the universe to break the single-factor regime.** Top eigenvalue
share is 84% with 40 markets. With more diverse markets (more domains, longer
history, or markets that genuinely move on different drivers), this should
drop and give the constrained optimizer real degrees of freedom. Specific
suggestions: add at least 20 more markets from underweighted domains
(commodities, currencies, weather/disaster, regulatory deadlines), keep
total universe ≤ 80 to stay tractable.

**(8) Different objective: net of holdout-baseline.** Right now Optuna
maximizes Sortino on walk-forward CV folds. Change the objective to
`Sortino(constrained) − Sortino(equal-weight on same fold)`. This forces
the search to find configurations that *beat* equal-weight rather than
just maximizing absolute Sortino (which equal-weight already does well at).
Likely to push the search away from the near-equal-weight collapse we see
across all pods.

### 6.4 Out of scope but worth noting for future work

- **Add features the model can actually use.** Macro/ETF features at hourly
  frequency don't forecast prediction-market resolutions. Markets are driven
  by news events, rule clarifications, and resolution-date proximity.
  Features like time-to-resolution, news-event indicators, or order-book
  imbalance from CLOB API are likely to be more informative than SPY/QQQ
  prices.

- **Cross-platform arbitrage instead of single-platform allocation.** The
  original `multiplatform_pipeline.py` includes Kalshi data. Equivalent
  binary contracts on Polymarket vs Kalshi often diverge by 1–3%; an
  arbitrage strategy is mechanically positive-EV and doesn't depend on
  the universe having more than one risk factor.

- **Different evaluation metric.** Sortino at hourly frequency on
  prediction-market returns is dominated by a few binary resolutions per
  contract per holdout. Consider reporting both **Sortino at daily
  frequency** (averaging out intra-day noise) and **realized P&L per
  contract resolution** as alternative evaluation lenses.

---

## 7. What this means for the project narrative

This is a clean, defensible negative result with diagnostic value. The story
is:

> Across four constrained portfolio configurations varying macro mode,
> ETF-tracking weight, and search strategy, all configurations converged to
> near-equal-weight allocation and underperformed the equal-weight baseline
> on the held-out 20% of data on both Sortino ratio (Δ ∈ [−0.060, −0.020])
> and maximum drawdown (Δ ∈ [−4.66%, −0.24%]). Spectral analysis shows the
> data has effectively one principal-component risk factor (top eigenvalue
> share 83–86%), leaving no room for a 40-market constrained model to
> differentiate from equal-weight. The macro/ETF integration adds a tracking
> penalty that further degrades performance, particularly during US equity
> sessions where it is most active. The joint macro search produced the
> smallest negative delta by selecting the highest concentration penalty in
> the experiment, effectively engineering a near-equal-weight solution. We
> conclude that the constrained mean-downside formulation with
> macro/ETF integration, applied to a 40-market hourly Polymarket universe,
> cannot find a meaningful out-of-sample edge over equal-weight, and we
> identify single-factor variance dominance as the structural reason.

This framing is stronger than "we beat the baseline" because it is
falsifiable, mechanism-grounded, and corroborated across four independent
runs.

---

## 8. Round 2: implementation and experiment plan

Round 1 (B/C/D/F + the still-running A/E) gave us a clean diagnosis. Round 2
is a targeted attempt to **beat the baseline** by acting on the four
strongest signals from Section 4:

- ETF tracking is net-negative whenever it is active (Section 3, pods C/D).
  → Round 2 turns ETF tracking off everywhere.
- Selection bias from 100-trial Sobol search inflates CV Sortino ~3–5× over
  holdout (Section 4.5).
  → Round 2 averages the top-K trials' holdout returns instead of using a
  single best trial.
- The CV objective rewards configurations that maximize absolute Sortino,
  which equal-weight already does well (Section 4.1, 4.4).
  → Round 2 changes the search objective to *Sortino(constrained) −
  Sortino(equal-weight)* on each fold.
- 14-dimensional search × 100 trials is sparse; many parameters converged
  identically across pods.
  → Round 2 narrows the search space (drops outer extremes, fixes
  entropy_lambda at 0).

Optionally we also blend toward baseline at evaluation time (shrinkage with
α searched by Optuna) to test whether even a 50/50 mix of constrained and
equal-weight is preferable to either alone.

### 8.1 Code changes shipped on `cloud-runs`

Two files changed; both backward-compatible (defaults preserve Round 1
behavior):

`src/constrained_optimizer.py` — `run_optuna_search` gains three keyword
args:

| Argument | Default | Effect |
|---|---|---|
| `top_k_bagging: int` | `1` | Average holdout `portfolio_returns`, `eval_weights`, and `avg_weights` of the top-K feasible trials (sorted by CV objective). K=1 is the legacy single-best behavior. |
| `baseline_shrinkage: bool` | `False` | Adds `baseline_alpha ∈ [0,1]` as an Optuna hyperparameter; final per-step return is `α·r_constrained + (1−α)·r_equalweight` in both fold scoring and holdout. |
| `beat_baseline_objective: bool` | `False` | Optuna maximizes `Sortino_constrained_CV − Sortino_equalweight_CV` on the same walk-forward folds (and uses the same delta for `MedianPruner` reports). |

The best-metrics JSON now includes `baseline_holdout_sortino`,
`holdout_sortino_minus_baseline`, `top_k_bagging`, `baseline_shrinkage`,
`beat_baseline_objective`, `bagged_trial_numbers`, `bagged_alphas`, and
`best_trial_baseline_alpha`. Comparing pods is now a one-line diff on these
fields.

`script/polymarket_week8_pipeline.py` — four new flags, threaded through to
both call sites of `run_optuna_search` (single-mode and `--macro-modes`
sweep):

| Flag | Argument | Notes |
|---|---|---|
| `--top-k-bagging` | int (default 1) | Recommended K=5 for Round 2. |
| `--baseline-shrinkage` | flag | Enables Optuna search of `baseline_alpha`. |
| `--beat-baseline-objective` | flag | Changes the optimization target to baseline-relative Sortino. |
| `--reduced-search` | flag | Narrows `learning_rates`, `penalties_lambda`, `domain_limits`, `max_weights`, `variance_penalties`, `downside_penalties`, `uniform_mixes`; pins `entropy_lambdas=(0.0,)`. Same trial budget → denser Sobol coverage of the meaningful subspace. |

### 8.2 Experiment matrix (six pods, ~ablation design)

Every Round 2 pod **drops `--etf-tracking`** (Tier 1 #1) and applies
`--reduced-search` (Tier 1 #2). Pods then vary along three axes:

- **Macro mode**: `rescale`, `explicit`, `both`, joint.
- **Top-K bagging**: K=1, 5, or 10.
- **Optimization target / shrinkage**: plain Sortino, beat-baseline, or
  shrinkage α.

| Pod | Macro | Top-K | Beat-baseline obj | Baseline shrinkage | Hypothesis being tested |
|---|---|---|---|---|---|
| **A** | rescale | 5 | no | no | Pure top-K bagging on the simplest macro. Isolates the bagging effect. |
| **B** | rescale | 1 | no | yes | Pure shrinkage-α on the simplest macro. Isolates the shrinkage effect. |
| **C** | both | 5 | no | no | Top-K bagging on the macro mode that blends rescale + explicit. |
| **D** | explicit | 5 | yes | no | Bagging + objective change on the explicit-only macro. |
| **E** | both | 10 | yes | yes | Kitchen sink: bigger bag, beat-baseline objective, and shrinkage α together. |
| **F** | joint | 5 | yes | yes | Joint macro search with all three improvements on. Complements E by letting Optuna pick macro mode. |

If the consistent winner is a configuration with both `--top-k-bagging` and
`--baseline-shrinkage` enabled (E, F), Round 3 should focus there. If A
alone closes the gap, bagging-only is the best path; if B alone closes it,
the result was always portfolio-level shrinkage, not constraint design.

### 8.3 What "beat the baseline" looks like in artifacts

For a Round 2 pod X, after the run completes, a positive
`holdout_sortino_minus_baseline` in
`data/processed/week9_X<suffix>_constrained_best_metrics.json` means we
beat the baseline. Suffix is `""` for `--macro-integration rescale`,
`"_macro_explicit"` for explicit, `"_macro_both"` for both. For F (joint)
the suffix is empty and `holdout_macro` inside the JSON tells you which
mode Optuna picked.

Quick fan-in on the local machine after all six finish:

```bash
for pod in A B C D E F; do
  for f in data/processed/week9_${pod}*constrained_best_metrics.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json,sys
d=json.load(open('$f'))['best_params']
print(f\"$pod  delta={d['holdout_sortino_minus_baseline']:+.4f}  \" 
      f\"sortino={d['holdout_sortino_ratio']:.4f}  \"
      f\"baseline={d['baseline_holdout_sortino']:.4f}  \"
      f\"K={d['top_k_bagging']}  shrinkage={d['baseline_shrinkage']}  \"
      f\"beat_obj={d['beat_baseline_objective']}\"
"
  done
done
```

A negative result here is still informative: it tells us that even with the
four highest-leverage corrections from Round 1, the constrained mean-downside
formulation cannot beat equal-weight on this universe — at which point the
project narrative shifts from "fix the model" to "fix the universe" (Tier 3
items 7 and the cross-platform-arbitrage note).

