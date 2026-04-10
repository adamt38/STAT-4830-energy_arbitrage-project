# Week 12 self-critique — Report draft 5

## OBSERVE

Draft 5 finally integrated stock overlays, Week 17 attribution, and Kelly honesty checks, but length exploded and some Week 17 windows looked “too good to be true” without resolution disclaimers.

## ORIENT

### Strengths

- Single-factor / PCA diagnosis explains MVO behavior crisply.
- Fee-aware baseline and net Kelly tables align incentives with reporting.

### Areas for improvement

1. **Stock/PM figure discipline:** regenerate or clearly caption placeholder-style plots; cite JSON summaries.
2. **Round 7 pod fan-in:** either merge K10E/K10F artifacts or explicitly scope default-branch claims (done in `docs/week11_round7_diagnostics_report.md` §7).
3. **Bibliography:** add formal citations before PDF export if required.

### Critical risks

Attribution showing >90% contribution from one resolving market invalidates short-window Sharpe comparisons unless flagged as mechanical.

## DECIDE

1. Merge narrative into root `report.md` as the single canonical submission file.
2. Move extreme 7-day plots to appendix or surround with resolution warnings.
3. Final presentation rehearsal focused on Slide 9 “honesty” numbers (break-even bps, z-score).

## ACT

Run `pytest tests/` after dependency pin changes; export deck PDF offline per `docs/final_presentation_guide.md`.
