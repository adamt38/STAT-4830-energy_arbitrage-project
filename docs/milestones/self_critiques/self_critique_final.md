# Final self-critique - Week 15

## OBSERVE

The final version is much more honest than the early drafts. It does not claim that we found a deployable trading strategy. It says equal-weight is a hard baseline, Kelly/copula is the best gross result, fees are still a serious problem, and the stock overlays are mostly diagnostic.

The repo is also very full now. That helps the implementation grade, but it can hurt a skim reader. The new `docs/final_artifact_index.md` helps, but the final submission still depends on clear pointers more than on adding more material.

## ORIENT

### Strengths

- The strongest part is the honesty around fees, bootstrap uncertainty, and resolution-driven spikes.
- There is a clear artifact trail now: `docs/final_artifact_index.md`, `data/processed/*`, `data/round7_cache/*`, `docs/week11_round7_diagnostics_report.md`, and the Kelly scripts.
- The project shows real iteration: MVO, macro/ETF attempts, Kelly, turnover penalties, momentum screening, and stock overlays.

### Areas for improvement

1. K10E/K10F claims need to stay scoped because the default branch does not contain every in-optimizer sweep table.
2. The Kelly conclusion would be much stronger with three or more seeds on the same locked universe.
3. I do not see a peer-feedback response file in the repo. If peer feedback was assigned outside GitHub, it should be summarized before final submission.

### Critical risks

The main risk is that a reader sees the gross Kelly gain and misses the fee caveat. The second risk is treating single-seed Kelly or Pod M wins as decisive when they are really directional.

## DECIDE

1. Keep `docs/final_artifact_index.md`, `README.md`, and `docs/README.md` as the grader-facing navigation layer.
2. Do not add more experiments unless a result is clearly missing from the final story.
3. Keep `docs/development_log.md` as the living engineering diary without duplicating it into the report body.

## ACT

Final checks completed: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v --tb=short` passed with `41 passed, 1 skipped`. The final presentation figures were regenerated into `docs/figures/final_presentation/` using `script/build_final_presentation_figures.py`. The last useful step is a reader pass, not another modeling pass.
