# Week 12 self-critique - report draft 5

## OBSERVE

Draft 5 has most of the pieces in one place: the MVO failures, the Kelly honesty checks, the stock/PM overlay work, and the Week 17 attribution. The cost is length. Some sections now feel like they are trying to prove every experiment mattered, when the cleaner story is that several experiments were useful because they failed.

The other issue is that some Week 17 windows look too good unless the report clearly explains the resolution-driven contribution.

## ORIENT

### Strengths

- Single-factor / PCA diagnosis explains MVO behavior crisply.
- Fee-aware and net-of-fees tables make the Kelly result much more honest.
- The stock overlay section is useful as a diagnostic even though it is not a headline win.

### Areas for improvement

1. Stock/PM figures need to be regenerated or clearly labeled if they are illustrative.
2. Claims about K10E/K10F need to be scoped carefully if the default branch does not contain full sweep tables.
3. The bibliography should be checked before PDF export if formal citations are required.

### Critical risks

The Week 17 attribution is the main risk: if more than 90% of a result comes from one resolving market, then short-window Sharpe or Sortino comparisons can mislead the reader unless we flag the mechanism.

## DECIDE

1. Merge narrative into root `report.md` as the single canonical submission file.
2. Move extreme short-window plots to the appendix or surround them with resolution warnings.
3. Rehearse the final presentation around the honesty numbers: break-even bps, bootstrap z-score, alpha blend, and single-factor diagnosis.

## ACT

Run `pytest tests/` after any dependency changes, regenerate the final figures, and export the deck offline. The final polish should remove clutter, not add another experiment.
