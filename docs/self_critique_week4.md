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
