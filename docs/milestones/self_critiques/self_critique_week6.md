# Week 6 self-critique - report draft 2

## OBSERVE

Draft 2 is stronger than the first draft because the project now has a clear Polymarket pipeline and a real holdout comparison. The problem is that the optimizer still reads like "we tried a constrained version and it lost" rather than a careful experiment. The figures also trail the code, so the report does not yet make the debugging path easy to follow.

## ORIENT

### Strengths

- Reproducible `script/polymarket_week8_pipeline.py` gave graders a single entry point.
- We documented the data-quality checks and the lookahead fix instead of hiding them.
- The baseline is now a serious comparison point, which makes the project more honest.

### Areas for improvement

1. Sortino as a direct loss is shaky over short rolling windows; we need a smoother mean-downside objective.
2. The difference between frozen weights and online updating is not explained clearly enough.
3. The report should connect API/data noise to optimizer behavior instead of treating data cleaning as separate plumbing.

### Critical risks

Tag-to-domain mapping can dominate the result. If one category quietly fills most of the universe, then the diversification numbers look cleaner than the portfolio actually is.

## DECIDE

1. Replace Sortino loss with additive mean-downside decomposition and projected simplex.
2. Report an online holdout path with multiple inner steps per bar, and explain why that matches deployment better than frozen weights.
3. Add covariance or PCA diagnostics before claiming that domain constraints are managing real risk.

## ACT

The next sprint should be mostly in `src/constrained_optimizer.py` and the pipeline arguments. After the next run, I should write down not only the best metrics but also whether the weights actually moved away from equal-weight.
