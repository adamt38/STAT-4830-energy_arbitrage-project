# Week 10 self-critique - report draft 4

## OBSERVE

Draft 4 finally has a more interesting result: the Kelly + dynamic-copula run gives a credible gross improvement where the MVO variants mostly did not. That is exciting, but it also makes the draft easier to oversell. At this point the slides and report were not fully aligned on which run was canonical, and the fee/bootstrap checks were not yet strong enough.

## ORIENT

### Strengths

- Kelly objective matches binary payoffs better than Sortino/MVO proxies.
- Dynamic copula gives macro features a more believable role: risk conditioning, not forced ETF tracking.
- The negative MVO result now has a purpose. It explains why we pivoted instead of looking like wasted work.

### Areas for improvement

1. Fees need to be front and center. Gross log-wealth is not enough if turnover makes the strategy untradeable.
2. A single-seed win needs a block bootstrap or some other paired uncertainty check.
3. The Kelly run worsens drawdown, so fractional Kelly should be part of the main story, not just an appendix detail.

### Critical risks

The biggest risk is mistaking a resolution-window windfall for a durable edge. If a few markets drive most of the gain, the attribution has to say that plainly.

## DECIDE

1. Run the fee ranking, alpha-blend, and bootstrap post-hocs before calling K10C a win.
2. Add K10D turnover penalty as engineering response before claiming deployability.
3. Keep the MVO negative result as a scientific contribution, not a footnote.

## ACT

Lock the key figures to cached inputs so the deck and report cannot drift. Do not start another expensive branch until the current Kelly story has fees, bootstrap, and attribution in the same place.
