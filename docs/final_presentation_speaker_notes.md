# Final Presentation — Speaker Notes (mapped to the actual 22-slide deck)

Companion to:
- [`docs/final_presentation_guide.md`](final_presentation_guide.md) — narrative and figure inventory.
- [`docs/final_presentation_technical_primer.md`](final_presentation_technical_primer.md) — plain-English concept explanations for Q&A.
- [`docs/final_presentation_speaker_script.md`](final_presentation_speaker_script.md) — full literal script (the earlier version, mapped to the original 12-slide plan). This document supersedes it for the actual deck.

**Deck:** `STAT4830_Week17.pptx.pdf`, 22 slides, 10-minute cap, followed by 2 min Q&A.

**Suggested speaker assignment** (swap if it feels wrong — the split balances content load):

| Speaker | Slides | Role |
|---|---|---|
| **Adam Thomson** | 1 – 7 (Title → First Results) | Sets up the problem, data, baseline, and MVO failure. |
| **Allen Xia** | 8 – 13 (Breaking Baseline → Kelly Headline) | Covers everything tried, the single-factor diagnosis, the pivot to Kelly, and the headline result. |
| **Xinkai Yu** | 14 – 22 (Post-Hocs → Limitations) | Honesty post-hocs, attempted fixes, takeaways, and limitations. |

**How to use.** Each slide lists: suggested figure on slide, timing target, and 3–6 **cue-card bullets** (not a literal script). Bold text = *say this number out loud*. Italic text = *stage direction*. Do not read the bullets verbatim; rehearse until you can say them in your own voice.

---

## Speaker 1 — Adam Thomson (Slides 1–7, ≈ 2:45)

### Slide 1 — Title (3 sec)

*Figure:* team names, course, title on screen.

- Stand still. Eye contact with the back of the room.
- One sentence only: *"Cross-Domain Portfolio Optimization on Polymarket. I'm Adam, with Allen Xia and Xinkai Yu."*
- Click immediately to Slide 2.

---

### Slide 2 — Why Prediction Markets? (30 sec)

*Figure:* text slide with three headers (New Market / Cross-Correlation / Main Takeaways).

- Prediction markets = **real-money, real-time probability bets**.
- Our universe = **40 binary contracts across 40 different topics** — politics, crypto, sports, economics.
- Catch: *forty different contracts can share one macro driver* — an election, a Fed decision, a crypto rally.
- One-line takeaway to land: *"Naive equal-weight diversifies **count**, not **risk**."*
- Close with: *"Can a constrained optimizer cap **domain** exposure and beat it? That's the project."*

---

### Slide 3 — Why Prediction Markets? (visual, 20 sec)

*Figure:* `v9_event_clustering.png` — bipartite chart, ~4 latent drivers on the left, 40 contracts on the right.

- *Point at one of the left-side clusters as you speak.*
- One line: *"Forty contracts, roughly four effective drivers. When election night hits, every election market moves together."*
- Don't over-explain — the figure does the work.

---

### Slide 4 — Data Pipeline (35 sec)

*Figure:* flow diagram (mermaid) of the pipeline.

- Start at the left box, trace with laser to the right as you speak.
- *"Pull 60 events from the gamma API → flatten to binary Yes/No → tag each market to a category → drop admin and rewards markets → round-robin select **40 markets across 40 categories**."*
- Key numbers: **57,141 total hourly bars**, **80/20 walk-forward split**, **11,429 holdout bars** — *this is the universe every strategy gets graded on.*
- Don't rush — the audience only sees this flow once.

---

### Slide 5 — Baseline, MVO Objective (45 sec)

*Figure:* the equation slide (the full MVO loss with domain caps + entropy bonus).

- Baseline is **equal-weight, 1/40 in each market, zero hyperparameters** — the null hypothesis.
- Walk the equation by *name*, not symbols:
  - "**Expected-return reward**, plus **variance** and **downside** penalties — so Sortino-style, losses punished more than symmetric volatility."
  - "**Covariance-quadratic term** for smoother gradients."
  - "**Two soft-hinge cap layers** — one per domain, one per contract — this is our original contribution."
  - "**Entropy bonus** to pull weights back toward uniform and avoid corner collapse."
- Key mechanism: **projected simplex, not softmax** — means weights can reach *exactly zero*. Flag this; it matters two slides later.
- Key blend: **40% uniform-mix floor** for stability.

---

### Slide 6 — Training Loop (35 sec)

*Figure:* OGD ribbon + Optuna migration note.

- **Online Gradient Descent, not offline SGD.** At every holdout bar: 3 Adam steps on a rolling window, project to simplex, realize the return, advance one bar. *State persists across all 11,429 bars.*
- Why online: Polymarket contracts *expire* — the return distribution is non-stationary. Offline train-once-deploy-forever would break within weeks.
- **Optuna:** migrated from TPE/Bayesian to **scrambled Sobol + MedianPruner on April 16** because 14 conditional dims × 100 trials is below Bayesian's viability threshold. Sobol just covers the cube uniformly; we get deterministic replication for free.
- Q&A readiness: Adam is itself an OGD algorithm (Kingma & Ba Theorem 4.1) — we're not stacking.

---

### Slide 7 — First Results (30 sec)

*Figure:* MVO vs baseline comparison (likely `fig01_mvo_sortino_delta_bar.png` or similar).

- Lead with the punchline: *"Our initial constrained model **did not** beat baseline."*
- Three observations to make:
  1. **Constrained portfolio has lower volatility** than baseline — so it worked as a risk controller.
  2. But it **isn't ahead on holdout Sortino** — risk-adjusted return is flat or worse.
  3. **Weights in many regimes sit near equal-weight** — despite being free to concentrate.
- Honest framing: *"The punchline from here on is iterative diagnosis — dilution, penalties, objective conditioning — under richer data and Optuna search."*
- **Hand-off line (memorize exactly):** *"Allen will walk through everything we tried to beat baseline."*
- *Pass clicker. Sit.*

---

## Speaker 2 — Allen Xia (Slides 8–13, ≈ 3:15)

### Slide 8 — Breaking Baseline (text) (50 sec)

*Figure:* text slide listing five levers A–E.

- **Pace yourself — five items in 50 seconds.** ~10 sec each. Don't get stuck on any one.
- **A. ETF tracking + macro features** — added SPY/QQQ/XLE as tracking targets and features. Net-negative overall, but **positive during US equity hours**. Closed bars outnumber open bars 5.7:1, which is why it lost in aggregate.
- **B. Momentum pre-screening** — shrink universe to top-20 markets by 5-day return. Sortino Δ **+0.008 to +0.016** — positive but **inside the noise band**.
- **C. Kelly + dynamic copula** — *foreshadow*: "the only lever that exits the noise band, we'll cover it in three slides."
- **D. Stock-PM combined strategy** — full VIX + 11-ticker risk-regime overlay with a domain→ticker hedge layer. Optuna's best trial **set `equity_signal_lambda = 0`** — *the optimizer itself rejected the reward channel*.
- **E. Learnable market inclusion** — differentiable inclusion gates on a 40-market universe. **Overfit**; hand-picked momentum beat it by **+0.051 Sortino**.
- Close with: *"Five distinct levers, none broke baseline at significance."*

---

### Slide 9 — Breaking Baseline (visual) (30 sec)

*Figure:* `v6_things_we_tried_scorecard.png` — 5-tile scorecard with mini risk-return scatters.

- Point at each tile briefly as you name it — you already named the levers; let the figure reinforce.
- Call out the **grey ±0.036 seed-noise band** — *"any pod inside this band is indistinguishable from random."*
- Only tile C (Kelly) clearly exits.
- Transition: *"One question remained: **why** did none of these work?"*

---

### Slide 10 — Single-Factor Dominance (text) (25 sec)

*Figure:* text slide with the two bullet findings.

- **Top eigenvalue across every MVO variant: 83–86% of total variance.**
- Our optimizer was *free to concentrate* — projected simplex can reach corners — **but it chose near equal-weight**.
- Explain the logic in one sentence: *"With roughly one real risk factor, equal-weight already exploits the only bet. Adding more hyperparameters doesn't help."*
- Frame this as the *diagnosis slide*, not the *failure slide*: *"This is a **structural property of the universe**, not a tuning issue."*

---

### Slide 11 — Single-Factor Dominance (visual) (25 sec)

*Figure:* `fig02_pca_eigenvalue_stacked.png` — horizontal stacked bar of top-5 PCA eigenvalue shares per pod.

- Point at the first blue bar: *"Eighty-six percent in one factor, across every MVO pod we ran."*
- Key line: *"We're fighting a universe that only has **one bet to make**."*
- Transition to Kelly: *"This told us we had the wrong objective — we needed to change **what** we optimize, not **how**."*

---

### Slide 12 — Pivot to Kelly (55 sec)

*Figure:* `v2_kelly_architecture.png` — full Kelly + copula block diagram, non-convex blocks in pink.

- Three ideas to name, *in order*:
  1. **New objective: expected log-wealth (Kelly).** The right growth-optimal objective for multiplicative, binary-payoff problems. Sortino is an arithmetic-variance proxy that breaks down on non-Gaussian returns.
  2. **Macro data re-enters — differently.** Not a tracking target, not a reward term. **The MLP reads SPY / QQQ / BTC returns and emits a correlation matrix $R_t$ every step.** Macro conditions the *risk model*, not the return forecast.
  3. **Same outer loop.** Same Adam-OGD, same projected simplex as before. What changes: the objective becomes log-wealth + MC-sampled Bernoulli, and we jointly train the MLP parameters $\theta$ and weights $w$.
- Name the four non-convex blocks briefly (pink): **MLP → PD shrinkage → Cholesky → Φ → straight-through Bernoulli**. Four layers, end-to-end differentiable.
- Honesty flag: *"The loss landscape is severely non-convex. We tolerate that."*

---

### Slide 13 — Kelly Headline Result (30 sec)

*Figure:* `fig03_k10c_cumulative_log_wealth.png` — cumulative log-wealth K10C vs baseline on 11,429 holdout bars.

- *Point at the gap on the right edge as you speak.*
- **"Plus zero point four-six log-wealth on the holdout"** — *pause* — *"that's roughly a **58 percentage-point CAGR gain** over baseline."*
- **Drawdown caveat:** max DD went from **−4.5% on baseline to −11.7% on K10C**. More than 2× worse on drawdown.
- Framing: *"This looks great. Three honesty tests on the next slides say: be careful."*
- **Hand-off line (memorize exactly):** *"Xinkai will walk through those three tests."*
- *Pass clicker. Sit.*

---

## Speaker 3 — Xinkai Yu (Slides 14–22, ≈ 3:30)

### Slide 14 — Post-Hocs: Bootstrap CI (30 sec)

*Figure:* `fig04_bootstrap_ci_histogram.png` — 1000 circular-block bootstrap replicates of Δ log-wealth.

- Why *circular block* bootstrap and not vanilla: **hourly returns are autocorrelated** — a naive bootstrap would give false-confident CIs.
- **1000 resamples, block size 24 hours**, re-grade K10C vs baseline on each.
- Headline: **95% CI = [+0.005, +0.973]** — zero barely excluded.
- **Z = +1.91, Pr(Δ > 0) = 97.5%.**
- Close: *"Statistically significant, but **marginally**. We wouldn't pass a two-sided α = 0.05 test."*

---

### Slide 15 — Post-Hocs: Fee Ladder (30 sec)

*Figure:* `fig05_net_of_fees_ladder.png` — grouped bars at fee ∈ {0, 10, 50, 200} bps.

- Re-grade K10A, K10B, K10C net of proportional trading fees.
- **Break-even fee for K10C = 3.76 bps.** *Pause.*
- Polymarket's effective frictions are at least 5 bps, likely 10+. **At 10 bps, Δ log-wealth flips to −0.76.**
- Close: *"The gross edge is real — but we cannot actually collect it at realistic fee levels."*
- Flag for Q&A: K10B break-even is higher at **10.93 bps**, but K10B's gross edge is smaller (+0.23, not +0.46).

---

### Slide 16 — Post-Hocs: α-Blend (35 sec)

*Figure:* `fig06_alpha_blend_with_dd.png` — Sortino vs α with MaxDD overlay.

- The test: **blend K10C weights with equal-weight** as $\alpha \cdot w_{\text{Kelly}} + (1-\alpha) \cdot w_{\text{uniform}}$, sweep α from 0 to 1.
- **Sortino peaks at α ≈ 0.60**, not at α = 1.
- What this means: *K10C is over-levered*. Full Kelly is risk-maximal, not Sortino-maximal.
- At **α = 0.5**: keeps **75% of the gross edge**, drawdown shrinks from **−11.7% → −7.6%**.
- Close: *"This is exactly fractional Kelly from the finance literature — Kelly tells you the growth-optimal bet size, Sortino tells you the risk-adjusted bet size, and we land at roughly **60% of full Kelly**."*

---

### Slide 17 — Post-Hocs: Summary (20 sec)

*Figure:* likely a 3- or 4-panel summary, possibly `v7_underwater_drawdown.png` or a composite.

- One spoken sentence to land the section: **"Gross edge is real but marginal (z = 1.91), destroyed by 10 bps of fees (break-even 3.76 bps), and over-levered on risk-adjusted terms (α ≈ 0.60)."**
- *Hold up three fingers as you list them — physical signal.*
- Transition: *"Three problems. We ran **two targeted fixes** on the first two."*

---

### Slide 18 — Attempted Fixes (25 sec)

*Figure:* `fig07_k10d_podm_grouped_bar.png` — 3-subplot grouped bar: log-wealth / turnover / max-DD for K10C, K10D, Pod M-seed7.

- **K10D: add L1 turnover penalty inside the Kelly loss.** $+ \lambda_{\text{turn}} \|w_t - w_{t-1}\|_1$.
- Turnover drops **0.107 → 0.020 — roughly 5× reduction.**
- Log-wealth Δ still **+0.27** (out of original +0.46). MaxDD −10.8%, essentially unchanged.
- Direct answer to the fee-fragility problem. Lower turnover = fewer fees per unit of edge.

---

### Slide 19 — Attempted Fixes (25 sec)

*Figure:* either the same grouped bar (Pod M column), or `v8_turnover_histogram.png`.

- **Pod M-seed7: Kelly machinery × momentum top-20/5d pre-filter.**
- *Framing:* **the one cross-term no prior pod had tested** — we had Kelly alone, we had momentum-screened MVO alone; we hadn't run Kelly on the momentum-screened universe.
- Results: Sortino Δ **+0.027**, log-wealth Δ **+0.22**, **Pr(Δ > 0) = 75.8%** via bootstrap.
- Honest caveat: *"Both K10D and Pod M-seed7 are **single-seed** results. Multi-seed confirmation is future work."*

---

### Slide 20 — Takeaways + Next Steps (visual) (25 sec)

*Figure:* `v13_next_steps_map.png` (or a text summary).

- Use the figure to frame what we have vs what's next — *"this map shows every pod we tried as a point in the features × universe plane."*
- One big takeaway line: *"We didn't find a deployable strategy. We found a map of where deployable strategies live."*
- Let the figure carry; don't over-explain.

---

### Slide 21 — Takeaways + Next Steps (text) (35 sec)

*Figure:* the three text bullets already on-slide.

- *Hold up one finger per bullet — physical signal that you're closing.*
- **Bullet 1. MVO never beat equal-weight** — not with macro, not with ETF tracking, not with risk-regime overlays, not with learnable universe. **Single-factor dominance is structural to this universe.**
- **Bullet 2. Kelly + dynamic copula has a real gross edge** — z ≈ 1.9 — precisely because it uses macro data as a *risk-model input*, not a tracking target. But without fee-awareness at train time and multi-seed CIs, it is not deployable.
- **Bullet 3. Directionally-correct fixes exist** — turnover penalty in K10D, Kelly × momentum in Pod M-seed7. **Next:** fee-aware Kelly (add λ_fee > 0 at train time), multi-seed replication, expand universe to break single-factor regime, richer features (resolution proximity, event calendars).

---

### Slide 22 — Limitations (40 sec)

*Figure:* three text items (Resolution Risk / Selection Metric / Single PC Dominance).

- **1. Resolution risk.** *"One market can dominate realized returns."* — when a contract nears resolution, its price swings violently, and if we have meaningful weight on it, it dominates holdout PnL. We did not risk-adjust for time-to-resolution.
- **2. Selection-metric mismatch.** *"Excess Sortino on walk-forward can stay negative while mean return rises."* — Optuna tuning Sortino does not guarantee a mean-return improvement; our 3–5× overfit between tuning and holdout bears this out.
- **3. Single-PC dominance.** *"First PC accounts for >80% of variance on the initial 40 markets."* — every optimizer we tried has to live with this. A wider or more heterogeneous universe is the structural fix.
- Close with: *"We didn't ship a winner. We shipped a **clean, falsifiable map** of where winning lives. Thank you."*
- *Smile. Wait for applause / Q&A.*

---

## Q&A — Handoff rules

- **Adam** owns: data pipeline, MVO math, OGD vs offline SGD, Optuna choice, baseline semantics. Appendices A, B, C, D, E, G.
- **Allen** owns: MVO failure, single-factor diagnosis, Kelly objective, dynamic copula, straight-through Bernoulli, **"where is your true probability?"** — Appendix H is the backup slide (`v14_kelly_flow_classical_vs_ours.png`) for that last question.
- **Xinkai** owns: bootstrap methodology, fee sensitivity, α-blend interpretation, K10D / Pod M results, honest limitations.

Each speaker answers their own domain. If a question spans two, the owner of the *first* concept in the question answers, then hands to the other: *"Xinkai, do you want to add anything on the bootstrap side?"*

**Default when unsure:** *"Great question — I want to give you the careful answer rather than a wrong one. Can we follow up by email after?"* Never bluff a number.

---

## 30-second course-takeaway round (post-clock)

Each of you: **30 seconds** on *"What did you find most useful or surprising about the course material?"* The guide notes this is *after* the 10-minute cap; it doesn't cost you time. Pick honestly from your own experience — seeds below are starting points, not scripts.

- **Adam:** *The gap between walk-forward CV Sortino and realized holdout Sortino.* We saw 3–5× overfit between tuning score and realized holdout. Selection bias became concrete for me through this project.
- **Allen:** *The right **objective** matters more than the right **optimizer**.* We changed the objective (Sortino → Kelly) — not the optimizer — and finally beat baseline. Lesson: when your optimizer plateaus, ask what loss it's actually minimizing before reaching for a fancier optimizer.
- **Xinkai:** *The circular-block bootstrap on autocorrelated time series.* A naive bootstrap would have given us false-confidence CIs. This one tool is what triggered the whole honesty-slide investigation.

---

## Timing table (for final rehearsal stopwatch check)

| Slide | Speaker | Target | Cumulative |
|---|---|---:|---:|
| 1 Title | Adam | 0:03 | 0:03 |
| 2 Why PM (text) | Adam | 0:30 | 0:33 |
| 3 Why PM (visual) | Adam | 0:20 | 0:53 |
| 4 Data pipeline | Adam | 0:35 | 1:28 |
| 5 Baseline + MVO | Adam | 0:45 | 2:13 |
| 6 Training loop | Adam | 0:35 | 2:48 |
| 7 First results | Adam | 0:30 | 3:18 |
| 8 Breaking baseline (text) | Allen | 0:50 | 4:08 |
| 9 Breaking baseline (visual) | Allen | 0:30 | 4:38 |
| 10 Single factor (text) | Allen | 0:25 | 5:03 |
| 11 Single factor (visual) | Allen | 0:25 | 5:28 |
| 12 Pivot to Kelly | Allen | 0:55 | 6:23 |
| 13 Kelly headline | Allen | 0:30 | 6:53 |
| 14 Post-hocs: bootstrap | Xinkai | 0:30 | 7:23 |
| 15 Post-hocs: fees | Xinkai | 0:30 | 7:53 |
| 16 Post-hocs: α-blend | Xinkai | 0:35 | 8:28 |
| 17 Post-hocs: summary | Xinkai | 0:20 | 8:48 |
| 18 Attempted fix — K10D | Xinkai | 0:25 | 9:13 |
| 19 Attempted fix — Pod M | Xinkai | 0:25 | 9:38 |
| 20 Takeaways (visual) | Xinkai | 0:25 | *over 10:00* — consider cutting |
| 21 Takeaways (text) | Xinkai | 0:35 | |
| 22 Limitations | Xinkai | 0:40 | |

**The deck as-is totals ≈ 10:43.** You are ~40 seconds over the 10-minute cap. Three ways to trim:

1. **Merge Slides 20 + 21** into one spoken block (25 sec total instead of 60 sec), keeping both on-screen but not narrating the visual panel separately. **Saves ≈ 35 sec.** *Recommended.*
2. Cut Slide 17 (post-hocs summary) from 20 sec → 10 sec by skipping the "three fingers" theatre and going straight to Slide 18. **Saves ≈ 10 sec.**
3. Cut Slide 3 narration from 20 sec → 10 sec — the visual speaks for itself. **Saves ≈ 10 sec.**

Doing #1 alone lands you at ~10:08. Doing #1 and #2 lands you at ~9:58 with comfortable buffer.

---

## Pre-flight checklist (day-of)

- [ ] All three speakers have the slide deck **open on their laptops** before the session — not just the presenter's machine.
- [ ] Clicker advance tested, laser pointer tested.
- [ ] Backup PDF of the deck on a USB key.
- [ ] Backup PDF of `v14_kelly_flow_classical_vs_ours.png` (Q&A spare figure) ready to pull up on laptop if someone asks the "true probability" question.
- [ ] Each speaker has the **headline number for their section memorized cold**: Adam = 11,429 holdout bars; Allen = +0.46 log-wealth / −11.7% DD; Xinkai = z = +1.91, 3.76 bps break-even, α = 0.60.
- [ ] Hand-off lines rehearsed 5× in a row the morning of.
- [ ] No one has caffeine-crashed 15 min before showtime.
