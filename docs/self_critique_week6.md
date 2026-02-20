# Week 6 self-critique — Report draft 2

## OBSERVE

Draft 2 documented a working Polymarket pipeline and honest holdout gaps, but the constrained optimizer still read as a “failed tweak” rather than a controlled ablation. Figures lagged code, and the Sortino-ratio story was hard to defend under small rolling windows.

## ORIENT

### Strengths

- Reproducible `script/polymarket_week8_pipeline.py` gave graders a single entry point.
- We documented data-quality checks and lookahead fixes transparently.

### Areas for improvement

1. Objective: Sortino as a direct loss is poorly conditioned; we needed a mean–downside surrogate.
2. Evaluation: uniform-mix and frozen-weight reporting obscured what “online” meant.
3. Narrative: weak connection between API noise (tags, history) and optimizer failure modes.

### Critical risks

Tag-to-domain mapping can dominate results; without per-category quotas, “diversification” metrics lie.

## DECIDE

1. Replace Sortino loss with additive mean–downside decomposition and projected simplex.
2. Report true online holdout with multiple inner steps per bar.
3. Add cross-pod PCA / eigenvalue diagnostics before claiming MVO success.

## ACT

Need one focused sprint on `src/constrained_optimizer.py` plus Optuna wiring; skim `docs/week9_cross_pod_synthesis.md` once Week 9 results exist.
