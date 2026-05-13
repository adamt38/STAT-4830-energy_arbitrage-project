# Week 4 self-critique - prototype baseline

## OBSERVE

We have moved the project from a loose energy-arbitrage idea into a concrete Polymarket portfolio project. The best part of this draft is that it is no longer just a proposal: there is an end-to-end path that pulls market data, builds a cached dataset, computes an equal-weight baseline, runs a constrained optimizer, and saves figures.

The uncomfortable result is that the first constrained model does not beat the baseline. That is still useful, but the draft needs to be clearer that the baseline is the thing we are trying to beat, not a formality.

## ORIENT

### Strengths
- We have a real pipeline instead of scattered notebook cells.
- The baseline-vs-optimizer comparison gives us a concrete benchmark for the rest of the semester.
- Data-quality checks are already part of the workflow, which should help us avoid silently trusting bad Polymarket histories.

### Areas for improvement
1. The optimizer is still too much of a black box. We need to explain what each penalty is supposed to do and why it should help.
2. The selected markets are not balanced enough across domains, so the "cross-domain" claim is weaker than it should be.
3. The validation setup is too light. We need a cleaner walk-forward split before we can trust any improvement.

### Critical Risks / Assumptions
The biggest risk is that tag-based domains are noisier than they look. If the tags are wrong or too broad, then our exposure constraints may be measuring the wrong thing. Sparse price histories are the second risk: missing or uneven CLOB data can make one market look safer than it really is.

## DECIDE

### Concrete Next Actions
1. Rebuild the market universe with explicit per-domain quotas.
2. Add a walk-forward evaluation split and report holdout metrics separately from tuning metrics.
3. Tighten the explanation of the objective function so the code and write-up match.

## ACT

### Resource Needs
- Confirm the Polymarket endpoint and historical price conventions before expanding the data pull.
- Set aside one focused iteration for model stabilization before the next slides.
- Keep notes on failed settings instead of only saving the best run; those failures are probably where the next improvement will come from.
