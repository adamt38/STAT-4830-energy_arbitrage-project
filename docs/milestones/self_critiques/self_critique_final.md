# Final self-critique — Week 15 (STAT 4830)

## OBSERVE

The final `report.md` now states a nuanced conclusion: equal-weight is a serious null; Kelly/copula is the best gross story but **not** fee-robust without further training; stock overlays are diagnostic. The repo is code-heavy and evidence-rich, which helps “Implementation” but challenges skim grading unless pointers are obvious.

## ORIENT

### Strengths

- Honest treatment of fees, bootstrap marginality, and resolution-driven Week 17 spikes.
- Clear artifact trail: `data/processed/*`, `docs/week11_round7_diagnostics_report.md`, Kelly scripts.
- Methodological breadth (MVO pods, Kelly, overlays) matches the course’s optimization focus.

### Areas for improvement

1. **K10E/K10F:** default branch lacks in-optimizer sweep tables; narrative must stay synced (see week11 §7).
2. **Figures:** a few submission-quality PNGs may still need regeneration from latest scripts for pixel-perfect PDF.
3. **Multi-seed:** Kelly conclusions would strengthen with 3+ seeds on identical universe locks.

### Critical risks

Readers may confuse **gross** Kelly significance with **net-of-fees** deployability; the abstract and §4.4 must stay aligned.

## DECIDE

1. Freeze `report.md` + week11 text after this merge; future pod fan-in updates week11 §4–5 only when data land.
2. Add formal references on export if the syllabus demands them.
3. Keep `docs/development_log.md` as the living engineering diary without duplicating it into the report body.

## ACT

Optional: run `script/backdate_milestone_commits.sh` **only** if team policy allows historical author dates; otherwise push with normal timestamps.
