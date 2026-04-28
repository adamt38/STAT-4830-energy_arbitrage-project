# LLM exploration log — final report structure (redacted)

**Purpose:** STAT 4830 “Development Process” artifact — excerpt of an assisted working session used to organize the final paper (not a verbatim transcript).

---

**User (paraphrased):** We have Kelly K10C gross win, MVO ties baseline, Week 17 stock overlay sets λ=0 at Optuna best. How should the final report avoid overstating deployability?

**Assistant outline (adopted):**

1. **Abstract** — lead with “equal-weight is hard to beat”; Kelly as best gross direction; fees/bootstrap caveats in one sentence.
2. **Methods** — separate subsections: fee-aware baseline math; mean–downside MVO; stock/PM sleeve accounting; Kelly + copula + straight-through estimator.
3. **Results** — Week 17 constrained vs baseline table first (honest underperformance on Sortino), then stock/PM 7-day with **alpha=0** caveat, then Kelly fee-aware table, then K10D turnover figure as **engineering response** to fee fragility, then “why equal weight” synthesis.
4. **Figures** — insist every PNG path exists in `figures/` or document JSON-only results.

**User follow-up:** Where do Round 7 post-hocs live?

**Assistant:** `data/processed/round7_fee_ranking.md`, `week10_kelly_C_alpha_blend_summary.md`, `round7_bootstrap_ci.md`; long-form in `docs/week11_round7_diagnostics_report.md`.

**Tooling notes:** User chose Markdown-first submission; export to PDF with serif 11 pt if instructors require print styling.

---

*Session lightly edited for length; no API keys or private URLs included.*
