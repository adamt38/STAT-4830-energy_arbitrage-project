# Week 10 self-critique — Report draft 4

## OBSERVE

Draft 4 pivoted to Kelly + copula with a credible gross result (K10C), but the write-up risked “headline chasing” without fee and bootstrap hygiene. Slides and report disagreed on which run was canonical.

## ORIENT

### Strengths

- Kelly objective matches binary payoffs better than Sortino/MVO proxies.
- Dynamic copula is a principled way to reuse macro features *as risk conditioning*, not ETF tracking.

### Areas for improvement

1. **Fees:** gross log-wealth is not economic without turnover and break-even bps analysis.
2. **Statistics:** single-seed Δ needs at least block bootstrap on paired residuals.
3. **Leverage:** max DD worsened versus baseline; fractional-Kelly story belongs in-report, not only appendix.

### Critical risks

High-turnover Kelly can be a variance harvest on resolution windows; attribution must name contributor concentration.

## DECIDE

1. Run `posthoc_fee_ranking.py`, `posthoc_alpha_blend.py`, `posthoc_bootstrap_ci.py` and fold into Round 7 doc.
2. Add K10D turnover penalty as engineering response before claiming deployability.
3. Keep MVO “negative result” as a full scientific contribution, not a footnote.

## ACT

Lock figure paths to `data/round7_cache/` inputs; coordinate GPU branch only for K10E/K10F once laptop story is airtight.
