# Final Artifact Index

This page is the fastest path through the non-report grading evidence: working code, tests, experiment artifacts, process notes, critiques, and presentation assets. The canonical graded report remains [`../report.md`](../report.md).

## Current Verification

- **Tests:** `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v --tb=short`
- **Latest local result:** `41 passed, 1 skipped` in 2.13s on May 12, 2026.
- **Skip reason:** `test_equity_domain_tilt_multiplier_changes_weights` is intentionally skipped until the equity-domain tilt is wired directly into `_run_online_pass`; overlay helpers are still covered by the surrounding risk/hedge tests.
- **Presentation figures:** regenerated with `MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/python script/build_final_presentation_figures.py`.

## Fast Grader Path

1. Start with [`../README.md`](../README.md) for the rubric map and runnable commands.
2. Run tests with `PYTHONPATH=. pytest tests/` or the exact command above.
3. Inspect implementation entrypoints in [`../src/`](../src/) and [`../script/`](../script/).
4. Use this page to jump to cached metrics instead of rerunning multi-hour Optuna or cloud jobs.
5. Read [`development_log.md`](development_log.md) and [`milestones/self_critiques/`](milestones/self_critiques/) for process and critique credit.
6. Use [`final_presentation_guide.md`](final_presentation_guide.md) and [`figures/final_presentation/`](figures/final_presentation/) for the final presentation narrative and Q&A prep.

## Implementation Evidence

- **Polymarket data and baseline:** [`../src/polymarket_data.py`](../src/polymarket_data.py), [`../src/baseline.py`](../src/baseline.py).
- **MVO / Sortino optimizer:** [`../src/constrained_optimizer.py`](../src/constrained_optimizer.py), orchestrated by [`../script/polymarket_week8_pipeline.py`](../script/polymarket_week8_pipeline.py).
- **Kelly / dynamic copula optimizer:** [`../src/kelly_copula_optimizer.py`](../src/kelly_copula_optimizer.py), orchestrated by [`../script/polymarket_week10_kelly_pipeline.py`](../script/polymarket_week10_kelly_pipeline.py).
- **Cross-platform / arbitrage path:** [`../script/multiplatform_pipeline.py`](../script/multiplatform_pipeline.py), [`../src/event_alignment.py`](../src/event_alignment.py), [`../src/arbitrage_signals.py`](../src/arbitrage_signals.py).
- **Equity and hedge overlays:** [`../src/equity_signal.py`](../src/equity_signal.py), [`../src/pm_risk_overlay.py`](../src/pm_risk_overlay.py), [`../src/stock_oil_hedge.py`](../src/stock_oil_hedge.py).
- **Tests:** [`../tests/`](../tests/) covers install smoke checks, macro integration, equity signals, risk-regime concentration, PM overlays, stock/oil hedges, seven-day hedge optimization, and figure contracts.

## Result Artifacts

- **MVO and cloud-run diagnostics:** [`week9_diagnostics_report.md`](week9_diagnostics_report.md), [`week11_round7_diagnostics_report.md`](week11_round7_diagnostics_report.md), and the corresponding `week9_*`, `week11_*`, and `week13_*` artifacts in [`../data/processed/`](../data/processed/).
- **Kelly K10A-D runs:** [`week10_kelly_diagnostics_report.md`](week10_kelly_diagnostics_report.md), [`week10_kelly_academic_summary.md`](week10_kelly_academic_summary.md), and `week10_kelly_*` metrics/timeseries in [`../data/processed/`](../data/processed/).
- **Round 7 cached Kelly comparison:** [`../data/round7_cache/`](../data/round7_cache/) stores K10A/K10B/K10C baseline and Kelly timeseries plus best metrics.
- **Fee and fractional-Kelly posthocs:** [`../data/processed/round7_fee_ranking.csv`](../data/processed/round7_fee_ranking.csv), [`../data/processed/week10_kelly_C_alpha_blend.csv`](../data/processed/week10_kelly_C_alpha_blend.csv), and [`../data/processed/week10_kelly_C_alpha_blend_summary.md`](../data/processed/week10_kelly_C_alpha_blend_summary.md).
- **Momentum x Kelly Pod M and fee-aware Pod MF:** [`option_B_results.md`](option_B_results.md), `week14_M_seed7_*`, and `week14_MF_seed7_*` artifacts in [`../data/processed/`](../data/processed/).
- **Stock/PM combined diagnostics:** [`week17_diagnostics_report.md`](week17_diagnostics_report.md), `week17_*` metrics in [`../data/processed/`](../data/processed/), and Week 17 figures in [`../figures/figures/`](../figures/figures/).

## Interpretation Hierarchy

- **Most robust negative result:** MVO / Sortino variants repeatedly fail to beat equal-weight once evaluated against the same holdout baseline.
- **Best gross positive result:** K10C Kelly + dynamic copula improves gross log-wealth, but its bootstrap significance is marginal and it is fee fragile.
- **Best practical fixes:** K10D turnover penalty and Pod M momentum x Kelly are directionally useful but still need multi-seed confirmation.
- **Stock overlay status:** Week 17 stock/PM overlays are diagnostic rather than a headline win; the best trial turned the direct equity-signal reward channel off.
- **Deployability caveat:** Gross edge is not the same as net-of-fees deployability; fee-aware training and richer features remain the clearest next steps.

## Process And Critiques

- **Development diary:** [`development_log.md`](development_log.md) records pivots, failed attempts, cloud-run design, branch integration, and test status.
- **Cloud methodology:** [`cloud_runbook.md`](cloud_runbook.md) documents parallel pod execution, artifact prefixes, and merge-safe experiment design.
- **Self-critiques:** [`milestones/self_critiques/`](milestones/self_critiques/) contains OODA reflections through the final critique.
- **LLM exploration:** [`llm_exploration/`](llm_exploration/) preserves AI-assisted exploration and final-structure planning.

## Presentation Assets

- **Slide guide:** [`final_presentation_guide.md`](final_presentation_guide.md).
- **Technical Q&A primer:** [`final_presentation_technical_primer.md`](final_presentation_technical_primer.md).
- **Speaker notes/script:** [`final_presentation_speaker_notes.md`](final_presentation_speaker_notes.md), [`final_presentation_speaker_script.md`](final_presentation_speaker_script.md).
- **Generated figures:** [`figures/final_presentation/`](figures/final_presentation/) now contains `fig01_*` through `fig07_*` plus `v1_*`, `v2_*`, `v3_*`, `v4_*`, `v5_*`, `v6_*`, `v7_*`, `v8_*`, `v9_*`, `v13_*`, and `v14_*` PNGs.
- **Figure builder:** [`../script/build_final_presentation_figures.py`](../script/build_final_presentation_figures.py) is extracted from the primer and can be rerun from the repo root.
