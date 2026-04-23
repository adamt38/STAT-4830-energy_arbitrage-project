# Final Presentation — Speaker Script (literal speaking copy)

Companion to:
- [`docs/final_presentation_guide.md`](final_presentation_guide.md) — slide-by-slide narrative + figure inventory.
- [`docs/final_presentation_technical_primer.md`](final_presentation_technical_primer.md) — technical concepts, plain English.

**Format:** 10 minutes total · 3 speakers · followed by ~2 minutes of Q&A.

**How to use this file.** Each slide gets a *target* (timing + word count), the *literal speaking copy*, and *stage directions* (`*italics in asterisks*`). Word counts are calibrated to a comfortable 150 words/min. Practise once with a stopwatch — you should land each slide ±3 seconds. The hand-off lines at the end of each speaker's section are non-negotiable: same exact wording every rehearsal so the transitions feel rehearsed and confident.

**Total spoken content: ≈ 9:00.** Targets a finish at ~9:30 with buffer for transitions and the headline-result pause on Slide 8.

---

## Speaker 1 — Slides 1 through 4 (~2:10)

> **Setup.** Speaker 1 is the framing speaker. Calm voice, slow pace on Slide 4 because it has the densest math. End by handing the clicker to Speaker 2.

---

### Slide 1 — Title (5 sec, ≈ 12 words)

*Click to title slide. Stand still. Make eye contact with the back row.*

> "Cross-Domain Portfolio Optimization on Polymarket. I'm [name], with [name two] and [name three]."

*Click to Slide 2.*

---

### Slide 2 — Why this problem (35 sec, ≈ 88 words)

*Point at the bipartite diagram, specifically at one of the four left-side latent drivers.*

> "Polymarket is real-money, real-time betting on binary events — elections, economic prints, sports. We picked forty contracts across forty different topics. The catch is hidden on this slide: forty *different* contracts can share **just a handful of underlying drivers**. Election day moves every election market in lockstep. A Fed announcement moves every rate-decision market together. So the obvious strategy — split your money evenly across all forty — diversifies *count*, not *risk*. Our project asks: can a constrained optimizer do better?"

*Pause one beat. Click.*

---

### Slide 3 — Data pipeline (35 sec, ≈ 88 words)

*Point at the leftmost mermaid box and trace the chain with the laser as you speak.*

> "Here's how the data gets to the optimizer. We pull sixty events from the Polymarket gamma API, flatten each into Yes-No binaries, tag every market to its category, drop the admin and rewards markets, then **round-robin select forty markets across forty distinct categories** so no single domain dominates. We pull hourly CLOB price history — fifty-seven thousand bars — and split eighty-twenty walk-forward, leaving us **eleven thousand four hundred twenty-nine holdout bars** to grade every strategy on. That's the universe everything from now on lives inside."

*Click.*

---

### Slide 4 — Baseline, MVO, training loop (55 sec, ≈ 138 words)

*Read the equation only if comfortable. Otherwise skip the symbols and explain by name.*

> "The null hypothesis is equal weight: one-fortieth in each market, every bar, zero hyperparameters. Anything we build has to beat that. Our optimization objective is mean-variance with two extras: a downside penalty so we punish losses harder than gains, and a soft cap on each *domain's* exposure. We parameterize on the **projected simplex** — meaning weights can hit exactly zero — which becomes critical two slides from now. The training loop is **Online Gradient Descent**: at every holdout bar, we run a few Adam steps on a rolling window, project, take the realized return, then advance one bar. State persists across all eleven thousand bars. This is the same loop we'll reuse for the Kelly model later. Hyperparameters are tuned by Optuna with scrambled Sobol — we'll explain that choice in Q&A if asked."

*Pause. Hand-off line:*

> "[Name two] will walk through everything we tried."

*Hand off the clicker. Click to Slide 5 as you sit down.*

---

## Speaker 2 — Slides 5 through 8 (~3:35)

> **Setup.** Speaker 2 covers the most material. Energy goes up on Slide 5 (five things tried — keep it punchy), then drops on Slide 6 (the diagnosis is the *aha* moment, deliver it slowly), then back up for the Kelly pivot. Headline number on Slide 8 must land like a punch.

---

### Slide 5 — Things we tried to break the baseline (75 sec, ≈ 188 words)

*The five-tile scorecard fills the slide. Point at each tile as you say its name. Move quickly — eye contact between tiles, not at the slide.*

> "We tried five distinct levers to beat baseline. Tile A: **add macro data** — SPY, QQQ, and energy-sector ETFs as features and as tracking targets. Result: net negative, with one neat side-finding — ETF tracking is *positive* during US equity hours but the equity market is closed eighty-five percent of the time. Tile B: **momentum pre-screening** — keep only the top twenty most-moving markets. Slightly positive but inside the seed-noise band, so not significant. Tile C is the Kelly model — we'll cover that in two slides, foreshadowing here. Tile D: a much richer **stock-PM combined strategy** — a VIX-driven risk-regime overlay plus a topic-to-ticker hedge layer using twelve different equities. Most ambitious thing we built. Best Optuna trial **set the equity reward weight to zero** — the optimizer itself decided the layer wasn't worth using. Tile E: **make the universe itself learnable** with differentiable inclusion gates. Added flexibility overfit; hand-picked top-twenty momentum beat it. Five distinct levers, none broke baseline at significance."

*Click.*

---

### Slide 6 — Diagnosis: single-factor dominance (35 sec, ≈ 88 words)

*Slow down. This is the explanation slide. Point at the tallest blue bar in the PCA stacked chart.*

> "Here's why none of it worked. We did a PCA on the residuals from each MVO run, and across **every variant we tried**, the **top eigenvalue captures eighty-three to eighty-six percent** of total variance. Eighty-six percent. Our optimizer was *free to concentrate* — the projected simplex can reach corners — but it kept landing near uniform because there's only one real risk factor in this universe. With one factor, equal weight already exploits the only bet. **It's a structural property of the universe, not a tuning issue.**"

*Pause. Click.*

---

### Slide 7 — Pivot to Kelly + dynamic copula (65 sec, ≈ 163 words)

*Use the Kelly architecture diagram (`v2_kelly_architecture.png`). Point at the pink blocks first.*

> "So we changed the objective. Three ideas to take away. First: when payoffs are binary and multiplicative, the right objective is **expected log-wealth — Kelly — not Sortino**. Sortino is an arithmetic-variance proxy that breaks down on big or non-Gaussian returns. Second: **macro data re-enters here, but differently** — not as a tracking target, not as an extra reward term, but as the **input to a small neural network that emits a correlation matrix every step**. The MLP reads SPY, QQQ, and BTC returns and outputs a state-dependent risk model. That's the dynamic copula. Third: we keep **the same Adam-OGD outer loop and the same projected simplex** as before — only the objective and the network parameters change. The forward pass stacks an MLP, a positive-definite shrinkage, a Cholesky, the Gaussian CDF, and a straight-through Bernoulli — four non-convex layers, but they train end-to-end."

*Click.*

---

### Slide 8 — K10C headline result (40 sec, ≈ 100 words)

*Cumulative log-wealth chart. Point at the gap on the right edge as you say the number. Pause after "+0.46".*

> "And it works. The Kelly model — we call it K10C — beats baseline by **plus zero point four-six total log-wealth on the holdout** ... that's roughly **a fifty-eight percentage-point gain in compounded annual growth**. The first thing in the project that exits the seed-noise band. The catch is on the next slide: the drawdown more than doubles, from minus four-and-a-half percent on baseline to minus twelve percent. So this looks great — and three honesty tests say *be careful*."

*Hand-off line:*

> "[Name three] will walk through those three tests."

*Hand off clicker. Sit. Click to Slide 9.*

---

## Speaker 3 — Slides 9 through 12 + closing (~3:08)

> **Setup.** Speaker 3 owns the credibility of the whole talk. Slide 9 is the longest single slide and the strongest signal that we understand our own results. Read each panel slowly. Don't rush the takeaway on Slide 11.

---

### Slide 9 — Three honesty post-hocs (90 sec, ≈ 225 words)

*The three-panel slide. Walk left to right, panel by panel. Point at each as you discuss it.*

> "Three tests, three panels. **Left:** a **circular-block bootstrap** — we re-sample the holdout in twenty-four-hour blocks one thousand times to preserve autocorrelation. The histogram of resampled deltas barely keeps zero out of the ninety-five-percent confidence interval — the lower bound is **plus zero point zero-zero-five**. Z-statistic is **plus one point nine-one**. So the gross edge is real but **only marginally significant**. **Middle:** a **fee ladder**. We re-grade Kelly with proportional fees from zero up to two hundred basis points per trade. The break-even fee for K10C is **three point seven-six basis points** — and Polymarket's effective frictions exceed that. At ten basis points the strategy actively *loses* zero point seven-six log-wealth. **Right:** an **alpha-blend** between full Kelly and equal-weight. The risk-adjusted-return curve peaks at **alpha equals zero point six**, not at one. So full Kelly is over-levered — fractional Kelly at sixty percent maximizes Sortino, and shrinks max drawdown from minus twelve percent to minus seven point six percent while keeping seventy-five percent of the gross edge. So: gross edge is real but marginal, **destroyed by fees**, and **over-levered on risk-adjusted terms**. Three problems, three fixes."

*Click.*

---

### Slide 10 — Two fixes that closed the gap (55 sec, ≈ 138 words)

*Three-subplot grouped bar. Point at the turnover bars first, then the log-wealth bars.*

> "Two fixes. **Fix one — K10D** — we add an L1 turnover penalty inside the Kelly loss. Average daily turnover drops from zero point one-zero-seven to **zero point zero-two-zero**, **roughly five-fold**, and we still keep **plus zero point two-seven** log-wealth out of the original zero point four-six. That's the directional answer to the fee-fragility problem. **Fix two — Pod M-seed seven** — we run the Kelly machinery on the **twenty-market momentum-screened universe**, the one cross-term no prior pod had tested. Sortino delta plus zero point zero-two-seven, log-wealth delta plus zero point two-two, and a bootstrap probability-positive of seventy-six percent. Both fixes are directionally correct. Honest caveat: **both are single-pod single-seed results** — multi-seed confirmation is future work."

*Click.*

---

### Slide 11 — Honest takeaway + what we'd do next (40 sec, ≈ 100 words)

*Three bullets. Hold up one finger per bullet — physical signal that you're closing.*

> "Three takeaways. **One:** mean-variance never beat equal-weight on this universe — single-factor dominance is structural. **Two:** Kelly with a dynamic, macro-conditioned copula has a real *gross* edge — about z equals one point nine — precisely because it uses macro data as a *risk-model input* rather than a tracking target. But without fee-awareness at train time and multi-seed confirmation, we won't call it deployable. **Three:** the directionally-correct fixes already exist — turnover-aware Kelly, Kelly times momentum. Next is fee-aware training, multi-seed replication, and a wider universe to break the single-factor regime."

*Land the closing line slowly.*

> "We didn't ship a winner. We shipped a clean, falsifiable map of where winning lives."

*Click to Slide 12.*

---

### Slide 12 — Q&A handoff and course-takeaway round (3 sec + 3 × 30 sec)

> "Questions."

*Hold for any direct questions. Then segue into the course-takeaway round.*

> "Before we close, each of us has thirty seconds on what we found most useful or surprising about the course material."

*Each speaker delivers one 30-second take. **Pick honestly** from your own experience — the bullets below are seed ideas, not scripts. Don't read them out word-for-word.*

#### Speaker 1 — course-takeaway (30 sec, ≈ 75 words)

*Suggested seed: walk-forward CV vs holdout overfit.*

> "The most useful thing for me was watching how big the gap got between our walk-forward CV Sortino and the realized holdout Sortino — we routinely saw three-to-five-times overfitting. It made selection bias *concrete*. I came in trusting my best-trial number; I leave thinking of every best-trial number as an upper bound that needs a holdout test before I believe it."

#### Speaker 2 — course-takeaway (30 sec, ≈ 75 words)

*Suggested seed: Kelly ≠ Sortino, the importance of objective choice.*

> "What surprised me most was that the *objective* mattered more than the *optimizer*. We spent weeks tuning Adam, schedulers, projection geometry — none of that broke baseline. We changed the objective from Sortino to Kelly and finally got an edge. Lesson: when an optimizer plateaus, ask what loss it's actually minimizing before reaching for a fancier optimizer."

#### Speaker 3 — course-takeaway (30 sec, ≈ 75 words)

*Suggested seed: circular-block bootstrap on autocorrelated time series.*

> "The thing I'll keep from the course is the **circular-block bootstrap**. A naive bootstrap on hourly returns would have given us false-confidence confidence intervals — way too tight, ignoring autocorrelation. The block version produced confidence intervals that actually *barely* contained our edge — which is what triggered the whole honesty-slide line of investigation. One specific tool that paid back its complexity immediately."

*Pause. Smile. Wait for moderator.*

> "Thank you."

---

## Practice protocol

1. **Solo runs.** Each speaker reads their own section out loud with a stopwatch **three times**. Target the per-slide word count to within ±10 percent.
2. **Hand-off rehearsal.** Run the two transitions (S1→S2 at end of Slide 4, S2→S3 at end of Slide 8) **five times in a row** as a separate drill. Use the exact hand-off lines printed above. The hand-off should be invisible — no pause, no "uh, your turn."
3. **Full-team run-through.** Three full passes with the actual deck. After each pass, the off-mic speakers note the *one* sentence the on-mic speaker fumbled. Fix it before next pass.
4. **Q&A drill.** Use [`final_presentation_guide.md`](final_presentation_guide.md) Appendices A through H — every appendix is a pre-written answer to a likely question. Speaker 3 owns honesty-slide questions; Speaker 2 owns Kelly / copula / "where is your true probability" (Appendix H); Speaker 1 owns data-pipeline and OGD questions (Appendix G).
5. **Backup if you go long.** If at the 8-minute mark Speaker 3 hasn't started Slide 11, **drop the alpha-blend panel narration on Slide 9** (cut ≈ 20 seconds) and keep moving. Never cut Slide 11 — the closing line *"We didn't ship a winner..."* is the whole talk's payoff.

---

## Word-count and timing summary

| Slide | Speaker | Time target | Words | Cumulative |
|---|---|---:|---:|---:|
| 1 Title | S1 | 0:05 | 12 | 0:05 |
| 2 Why | S1 | 0:35 | 88 | 0:40 |
| 3 Data pipeline | S1 | 0:35 | 88 | 1:15 |
| 4 Baseline + MVO + training loop | S1 | 0:55 | 138 | 2:10 |
| 5 Things we tried | S2 | 1:15 | 188 | 3:25 |
| 6 Diagnosis | S2 | 0:35 | 88 | 4:00 |
| 7 Pivot to Kelly + copula | S2 | 1:05 | 163 | 5:05 |
| 8 K10C headline | S2 | 0:40 | 100 | 5:45 |
| 9 Honesty: three post-hocs | S3 | 1:30 | 225 | 7:15 |
| 10 Two fixes | S3 | 0:55 | 138 | 8:10 |
| 11 Takeaway + next | S3 | 0:40 | 100 | 8:50 |
| 12 Q&A handoff | S3 | 0:03 | 3 | 8:53 |
| Course-takeaway × 3 | All | 1:30 | 225 | 10:23 |

The course-takeaway round is **post-clock** by your guide's note (does not count against the 10 min). The talk itself comes in at **8:53**, leaving ~1:00 of comfort buffer.

---

## One last reminder

You are not memorizing a monologue. **Memorize the headline number per slide** (e.g., "+0.46 log-wealth," "−4.5% baseline drawdown," "3.76 bps break-even fee"), and let the surrounding sentences flow naturally. The script above is the *spine* — your job is to put the muscles on. If you find yourself reading verbatim, that's a sign you haven't rehearsed enough.
