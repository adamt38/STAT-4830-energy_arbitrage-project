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

- **Three ideas to name, in order.**
  1. **New objective: expected log-wealth (Kelly).** The right objective for multiplicative, binary-payoff problems. Sortino is an arithmetic-variance proxy that breaks down on non-Gaussian returns.
     - *Plain English:* a 50% loss needs a 100% gain to recover. Arithmetic mean hides that asymmetry; log-wealth doesn't. Our contracts can move 20+ points in a day, so Kelly is the mathematically correct objective — Sortino systematically over-leverages here.
  2. **Macro data re-enters — differently.** The MLP reads SPY / QQQ / BTC returns and **emits a correlation matrix $R_t$ every step**. Macro conditions the *risk model*, not the return forecast.
     - *Plain English:* Slide 8's macro tries failed because we asked macro to *predict returns* (it can't). Here, macro **outputs the 40×40 correlation matrix directly** — on calm days $R_t$ is near-identity (diversification is real); on stormy days near-rank-1 (everything co-moves). Same data, fundamentally different use.
  3. **Same outer loop.** Same Adam-OGD, same projected simplex. Objective becomes log-wealth + MC-sampled Bernoulli; we jointly train MLP parameters $\theta$ **and** weights $w$.
- **The four pink (non-convex) blocks, and why each exists:**
  - **MLP** outputs raw correlation from macro features.
  - **PD shrinkage** forces positive-definiteness (you can't Cholesky a raw MLP output).
  - **Cholesky** gives us $L_t$ so that $L_t z \sim \mathcal{N}(0, R_t)$ from unit Gaussians.
  - **Φ (Gaussian CDF)** maps correlated Gaussians to correlated uniforms.
  - **Straight-through Bernoulli** turns uniforms into differentiable 0/1 payoffs — hard Bernoulli on forward pass, sigmoid-surrogate gradient on the backward. Without it the whole pipeline would have zero gradients everywhere.
- **Honesty flag:** *"The loss landscape is severely non-convex — four non-linearities stacked. Adam finds *a* local optimum, not *the* global one. That is why multi-seed replication is on the next-steps list."*
- **One-line headline to land:** *"Kelly gives us the right objective; the dynamic copula gives us a risk model that finally reflects market state; Adam-OGD trains both on the same simplex we used for MVO."*

**Deeper plain-English — what Kelly does, and how it actually differs from MVO:**

- **What MVO is doing, in one sentence.** *"Given today's estimated mean return vector and covariance matrix, pick weights that maximize mean-minus-lambda-times-variance."* It is a **one-period, arithmetic-mean** objective. There is no notion of compounding; no penalty for going broke; every day is graded as if it were independent and the same statistics will repeat tomorrow.

- **What Kelly is doing, in one sentence.** *"Given that you will bet over and over and let wealth compound, pick weights that maximize the **geometric** growth rate of your bankroll."* Formally: maximize $E[\log(1 + w^\top r)]$. Because the sum of per-bar log-returns equals the log of compounded wealth, maximizing the per-bar expectation is **mathematically equivalent** to maximizing long-run compounded terminal wealth (Kelly 1956; Thorp 1971).

- **The one example that makes the difference visceral.** Imagine a bet with 50% chance of **+100%** and 50% chance of **−50%**.
  - *MVO sees:* arithmetic mean = **+25%**, modest variance — *"bet looks great, size it up."*
  - *Kelly sees:* $E[\log(1+r)] = 0.5 \log(2) + 0.5 \log(0.5) = 0$ — *"the bet compounds to flat over the long run; do not oversize."*
  - *This is exactly the failure mode on binary contracts:* a contract at price 0.50 pays +100% on YES and −100% on NO, which Kelly correctly flags and MVO does not.

- **Why this matters in *our* setting specifically.** When returns are small (low-single-digit percent like equities), $\log(1+r) \approx r$, and MVO ≈ Kelly — which is why MVO is fine for most finance textbook problems. **Polymarket returns are 20+ points per day.** The approximation breaks badly. MVO systematically over-bets high-mean-high-variance contracts, because its arithmetic-mean objective doesn't see that a 50% loss is twice as destructive as a 50% gain is constructive, in compounding terms.

- **Three things Kelly bakes in structurally that MVO does not.**
  - **Asymmetric downside awareness.** $\log(0) = -\infty$ — Kelly's objective literally **explodes** if you take a position that could wipe you out. MVO's variance treats +50% and −50% symmetrically, which is exactly wrong for compounding.
  - **Natural leverage control.** Kelly's optimum bet size *already encodes* how aggressive you should be. You do not have to hand-tune a risk-aversion coefficient like MVO's $\alpha_v$ — the log curvature does it for you.
  - **Growth-rate semantics.** Kelly's objective is the *long-run annualized growth rate* of your money, which is what any real deployer actually cares about. MVO's objective is a one-period trade-off between arithmetic mean and variance — a mathematical abstraction that only matches real money at the small-return limit.

- **What stays identical between our MVO and Kelly pods.** Optimizer (Adam-OGD), parameterization (projected simplex), outer loop (rolling window, 3 inner steps per bar), constraints (per-domain caps, per-contract caps, uniform-mix floor), universe (40 markets), hyperparameter infrastructure (Optuna scrambled Sobol). **Only two things change:** (a) the objective swaps from arithmetic mean–variance to expected log-wealth, and (b) we replace the rolling-window covariance with the MLP-emitted macro-conditioned copula $R_t$. *That is the entire architectural difference* — we reused ~90% of the MVO code.

- **The honest caveat** *(covered in depth in primer §13a, Appendix H for Q&A)*: because we plug the market price back in as the Bernoulli probability, the simulator's expected per-market payoff is *zero*. So our implementation is Kelly in name but **minimum-variance in effect** — Jensen expansion around $E[\pi] = 0$ gives $E[\log(1 + w^\top \pi)] \approx -\tfrac{1}{2} w^\top \Sigma(R_t) w$. The genuine Kelly advantages above are *available* to our framework; our holdout edge comes from min-var + drift, not from a true probability edge. Real Kelly would require plugging in a separately-estimated $\hat p \ne q$ (next-steps bullet).

- **The one-sentence Q&A punchline:** *"MVO optimizes the shape of today's return distribution; Kelly optimizes the long-run multiplicative growth of wealth. When bets are small they agree; when bets are large — Polymarket is large — Kelly is the correct objective. MVO systematically over-bets contracts that look high-mean but are actually compounding-neutral or worse."*

**Architecture block-by-block walkthrough (point at `v2_kelly_architecture.png` left→right):**

- **Step 0 — Inputs (top-left).** Every hour we read three things: (a) **market prices** $p_t \in [0,1]^{40}$ for our 40 contracts; (b) **macro features** $m_t$ = that hour's SPY / QQQ / BTC returns; (c) **current weights** $w_t$ carried over from the previous bar.
  - *Key design choice to call out:* macro is input to the **risk model**, not the alpha model. This is the single most important architectural decision on the slide.

- **Step 1 — MLP: macro → raw correlation (pink block #1).** A 2-layer net reads the 3 macro features and outputs the 780 off-diagonal entries of a 40×40 correlation matrix. One matrix per hour.
  - *Why:* on calm days we want $R_t$ near identity (diversification real); on stormy days near rank-1 (everything co-moves). The MLP **learns that mapping** from data — we don't hand-code regimes.

- **Step 2 — PD shrinkage (pink block #2).** Shrink the raw matrix toward identity: $R_t \leftarrow (1-\alpha)\tilde R_t + \alpha I$, picking the minimum $\alpha$ that keeps the smallest eigenvalue $\geq 10^{-3}$.
  - *Why:* Cholesky (next step) requires positive-definite input. Without shrinkage the pipeline crashes any time the MLP accidentally produces a near-singular matrix — which happens constantly in early training.

- **Step 3 — Cholesky: $R_t = L_t L_t^\top$ (pink block #3).** Standard trick: $L_t$ is the "square root" of the correlation. Multiplying independent unit Gaussians by $L_t$ yields a draw from $\mathcal{N}(0, R_t)$.

- **Step 4 — Build correlated uniforms (Gaussian copula).**
  1. Sample $z \sim \mathcal{N}(0, I_{40})$ — 40 independent standard Gaussians.
  2. Multiply by $L_t$: $x = L_t z$ → now correlated Gaussians with correlation $R_t$.
  3. Apply the Gaussian CDF element-wise: $u = \Phi(x)$ → correlated **uniforms** on $[0,1]$.
  - *Plain English:* "the copula is a machine that takes 'how correlated things are' and spits out 'a batch of simulated worlds that actually exhibit that correlation.'" We do this 128× per bar (Monte-Carlo).

- **Step 5 — Straight-through Bernoulli (pink block #4).** For each simulated world, $y_i = \mathbb{1}\{u_i \leq p_i\}$ gives a hard 0/1 outcome per contract on the **forward** pass. On the **backward** pass we replace the indicator with a sigmoid surrogate $\sigma(k(p - u))$ so gradients can flow.
  - *Why:* indicator functions are flat almost everywhere → zero gradient → nothing trains. The surrogate passes the hard value forward but gives a usable gradient back.
  - *Honesty note:* this is the one place we accept biased gradients. Documented on the next-steps slide.

- **Step 6 — Payoffs and expected log-wealth.** Centered, normalized per-contract payoff: $\pi_i = (y_i - p_i) / [p_i(1 - p_i)]$ — i.e. $+1/(1-p)$ if YES hits, $-1/p$ if NO hits. Log-wealth per world: $\log(1 + w_t^\top \pi)$. Average over the 128 worlds → **expected log-wealth**, the objective.
  - *Why this shape:* log-wealth summed across time = log of compounded terminal wealth. Max that expectation and you max long-run growth — the entire point of Kelly.

- **Step 7 — Turnover penalty (K10D only).** $\mathcal{L}_t = -\mathbb{E}[\log\text{-wealth}] + \lambda_{\text{turn}} \|w_t - w_{t-1}\|_1$.
  - *Why:* K10C had no penalty and was fee-fragile; K10D added this and survived the 5bp fee ladder. Small change, big robustness win.

- **Step 8 — Two-headed gradient update (the training loop).** One Adam-OGD step computes $\partial\mathcal{L}/\partial w$ **and** $\partial\mathcal{L}/\partial\theta_{\text{MLP}}$ in the *same* backward pass. Three inner steps per bar. Both heads move together.
  - **Weight head:** projected onto the simplex (sum to 1, per-domain caps, per-contract caps, uniform-mix floor) — **same projector we built for MVO**.
  - **MLP head:** unconstrained; just learns to emit better $R_t$.
  - *The key conceptual point:* we are **jointly** training the risk model *and* the allocation. Most finance pipelines estimate covariance first, allocate second. We collapse both into one end-to-end gradient pass.

- **Step 9 — Roll forward one hour, repeat.** Store new weights; move to $t+1$; repeat 11,429 times for the holdout. **Online learning** — every bar is both a learning bar and a deployment bar. No look-ahead.
  - *Connects to class material:* this is the OCO framing from lecture — same outer structure as MVO, now with a non-convex inner objective.

- **20-second verbal summary (memorize this).** *"Macro features go into an MLP that outputs a correlation matrix. Cholesky turns that into a sampler. We simulate 128 possible worlds each hour, compute expected log-wealth across those worlds, and take one Adam step that updates both our portfolio weights and the MLP at the same time. Then we roll forward an hour. That's it."*

- **Four probes to pre-arm for** (pointer → step):
  - *"Where's the 'true' probability?"* → Step 5. *"We plug market price back in as $p$; simulator expectation is zero; min-var in effect. See primer §13a / Appendix H."*
  - *"Why copula instead of rolling covariance?"* → Steps 1–4. *"Rolling gives one fixed matrix per window; MLP emits a new $R_t$ every hour, conditioned on macro — identity on calm days, rank-1 on storm days."*
  - *"What's non-convex?"* → Steps 1, 2, 3, 5. *"Four stacked non-linearities: MLP, PD shrinkage, Cholesky, Bernoulli. Adam finds a local optimum — multi-seed replication is on the next-steps list."*
  - *"Why log-wealth over Sortino?"* → Step 6. *"Log-wealth matches compounding; Sortino is arithmetic, breaks down on 20-point-per-day moves."*

---

### Slide 13 — Kelly Headline Result (30 sec)

*Figure:* `fig03_k10c_cumulative_log_wealth.png` — cumulative log-wealth K10C vs baseline on 11,429 holdout bars.

- *Point at the gap on the right edge as you speak.*
- **"Plus zero point four-six log-wealth on the holdout"** — *pause* — *"that's roughly a **58 percentage-point CAGR gain** over baseline."*
  - *What the number means in dollars:* $e^{0.46} \approx 1.58$ — K10C ended the holdout with ~58% more money per dollar started than baseline.
  - *Why log-wealth, not arithmetic return:* log adds instead of multiplies, matches Kelly's training objective, and honestly represents compounding (50% gain + 50% loss = 75¢, not $1).
- **Drawdown caveat:** max DD **−4.5% baseline → −11.7% K10C**. More than 2× worse.
  - *What this feels like:* there was a point during holdout where K10C was 11.7% underwater from its own prior high. We collect the edge by **tolerating more than twice the interim pain**.
  - *Is this trade worth it?* Exactly what the α-blend on Slide 16 answers — the answer is "about 60% of full leverage, not 100%."
- **Why the curve diverges and then plateaus:** the dynamic copula adapts to macro state. On high-signal bars (big SPY moves, etc.) K10C cashes in variance reduction; on quiet days K10C ≈ baseline. The gap is the accumulated result of many state-dependent decisions.
- **The meta-point to say out loud:** *"K10C is the **first strategy in the entire project** that clearly exits the seed-noise band."*
- Framing: *"This looks great. Three honesty tests on the next slides say: be careful."*
- **Hand-off line (memorize exactly):** *"Xinkai will walk through those three tests."*
- *Pass clicker. Sit.*

---

## Speaker 3 — Xinkai Yu (Slides 14–22, ≈ 3:30)

### Slide 14 — Post-Hocs: Bootstrap CI (30 sec)

*Figure:* `fig04_bootstrap_ci_histogram.png` — 1000 circular-block bootstrap replicates of Δ log-wealth.

- **What the bootstrap does, in one sentence:** we resample our one holdout run with replacement to simulate what *other plausible holdouts* could have looked like, then read off the 95% CI from that distribution. It converts a point estimate (+0.46) into an interval that reflects real uncertainty.
- **Why *circular block* and not vanilla:** hourly returns are autocorrelated — today's return is correlated with yesterday's. Vanilla bootstrap draws individual bars with replacement, which assumes independence — would give **artificially tight** CIs. Circular-block bootstrap draws contiguous **24-hour chunks** that preserve within-day structure; "circular" just means blocks wrap the end back to the start. Standard method (Politis & Romano 1994).
- **The protocol:** **1000 resamples, block size 24 hours**, re-grade K10C vs baseline on each, plot the distribution.
- **Headline:** **95% CI = [+0.005, +0.973]** — zero is *barely* excluded. **Z = +1.91, Pr(Δ > 0) = 97.5%.**
  - *Translation:* under a normal approximation, one-sided p ≈ 0.028. Would reject "K10C = baseline" at α = 5% but fail at α = 2.5%.
- **Why this test matters:** without it, we'd be reporting "+0.46" as *the* number. With it, we see the interval nearly includes zero — the honest framing is "marginally significant," not "significant." **Reporting +0.46 without this test would be overclaiming.**
- Close: *"Statistically significant, but **marginally**. We wouldn't pass a two-sided α = 0.05 test."*

---

### Slide 15 — Post-Hocs: Fee Ladder (30 sec)

*Figure:* `fig05_net_of_fees_ladder.png` — grouped bars at fee ∈ {0, 10, 50, 200} bps.

- **What "bps" means:** 1 bp = 0.01%. "10 bps of fees" = 0.10% of dollar value paid on every trade (covers bid-ask spread + slippage + exchange fees).
- **Why fees hurt Kelly more than baseline:** baseline never rebalances (turnover ≈ 0), so its fee bill is ~zero. K10C has **daily turnover 0.107** → at 10 bps per unit of turnover × 365 days ≈ **390 bps/year of fee drag** that compounds against the gross edge.
- **The test:** re-grade K10A, K10B, K10C at fee ∈ {0, 10, 50, 200} bps; bar heights are net-of-fee Δ log-wealth.
- **Break-even fee for K10C = 3.76 bps.** *Pause.*
  - *What "break-even" means:* the fee level where net-of-fee Δ = exactly 0. Below it, K10C wins; above it, K10C loses. The break-even *is* the edge-per-unit-of-turnover.
- **Polymarket reality:** bid-ask spreads are 10–50 bps on less-liquid contracts, plus price-impact cost. Effective frictions are at least 5 bps, likely 10+. **At 10 bps, Δ log-wealth flips to −0.76** — the strategy actively *loses* more than half a log-wealth unit.
- **Q&A landmine:** K10B break-even is higher (**10.93 bps**) than K10C's, but K10B's gross edge is smaller (+0.23 vs +0.46). In the realistic 5–10 bps range, **neither** strategy wins net.
- **Forward pointer:** K10D (two slides from now) directly attacks this by penalizing turnover at train time — the fix for fee fragility.
- Close: *"The gross edge is real — but we cannot actually collect it at realistic fee levels."*

---

### Slide 16 — Post-Hocs: α-Blend (35 sec)

*Figure:* `fig06_alpha_blend_with_dd.png` — Sortino vs α with MaxDD overlay.

- **The test, in math:** sweep α from 0 to 1 and re-run the holdout at each step using $w_\alpha = \alpha \cdot w_{\text{Kelly}} + (1-\alpha) \cdot w_{\text{uniform}}$.
  - α = 1 is pure K10C; α = 0 is pure baseline; between is a blend. Record Sortino, log-wealth, and max-DD at each α.
- **Result: Sortino peaks at α ≈ 0.60**, not at α = 1.
- **Why the peak is interior (not at α = 1):** Sortino = return / downside deviation. Both grow with α, but not linearly — the ratio has an interior maximum where marginal return stops compensating for marginal downside. Above 60%, **we're losing Sortino by adding more Kelly**.
- **What this tells us:** K10C is **over-levered**. Full Kelly is growth-maximal, not risk-adjusted-maximal. This is exactly the "fractional Kelly" result from MacLean-Thorp-Ziemba — in the literature, deployers almost always use half-Kelly or quarter-Kelly.
- **At α = 0.5:** keeps **75% of the gross edge**; drawdown shrinks **−11.7% → −7.6%**.
  - Rule of thumb: half-Kelly gives up ~25% of growth and cuts drawdown in half. We reproduce this exactly.
- **Why max-DD is monotonic in α:** more Kelly ⇒ more directional exposure ⇒ more drawdown, always. So the Sortino-optimal α ≈ 0.6 is *also* a drawdown-reduction (−11.7% → roughly −9% at α = 0.6).
- **Why K10C is over-levered in the first place** *(links to primer §13a)*: because we plug the market price back in as the Bernoulli probability, simulated $E[\pi_i] = 0$ per market. Kelly's natural scale comes from the *edge*, which is zero in our simulator — so the optimizer has no natural magnitude and pushes leverage against the concentration caps. The α-blend rescues the risk-adjusted result after the fact.
- **Why this matters for the story:** our reported +0.46 headline is at a *sub-optimal* point on the risk-adjusted frontier. A realistic deployment uses only ~60% of the Kelly signal.
- Close: *"Kelly tells you the growth-optimal bet size, Sortino tells you the risk-adjusted bet size — we land at **60% of full Kelly**."*

---

### Slide 17 — Post-Hocs: Summary (20 sec)

*Figure:* likely a 3- or 4-panel summary, possibly `v7_underwater_drawdown.png` or a composite.

- **The three post-hocs are separate problems, not alternative measures of the same thing:**
  1. **Bootstrap (Slide 14)** — *Is the edge real?* Yes, but marginally (z = 1.91). Edge exists in the data.
  2. **Fees (Slide 15)** — *Can we collect it?* Not at realistic Polymarket frictions (break-even 3.76 bps).
  3. **α-blend (Slide 16)** — *Are we collecting it efficiently?* No, over-levered — optimum is ~60% of full Kelly.
- One spoken sentence to land: **"Gross edge is real but marginal (z = 1.91), destroyed by 10 bps of fees (break-even 3.76 bps), and over-levered on risk-adjusted terms (α ≈ 0.60)."**
- *Hold up three fingers as you list them — physical signal.*
- **Which problems get fixed next:** K10D (Slide 18) directly attacks #2. α-blend deployment implicitly fixes #3. **#1 needs multi-seed replication** — we didn't have time; it's in the next-steps list on Slide 21.
- **Why keeping them distinct matters:** any one of these would be a red flag; all three together are what justify "marginal," not "working." One fix at a time is the right research strategy.
- Transition: *"Three problems. We ran **two targeted fixes** on the first two."*
- **If the slide carries `v7_underwater_drawdown.png`:** point at K10C's deepest trough (≈ −11.7%) as a visual for "this is why α-blend matters." Underwater curves make drawdown viscerally obvious.

---

### Slide 18 — Attempted Fixes (25 sec)

*Figure:* `fig07_k10d_podm_grouped_bar.png` — 3-subplot grouped bar: log-wealth / turnover / max-DD for K10C, K10D, Pod M-seed7.

- **What "L1 turnover" is:** $\|w_t - w_{t-1}\|_1 = \sum_i |w_{t,i} - w_{t-1,i}|$ — the sum of absolute weight changes across all 40 markets. Double turnover = double fees.
- **K10D adds one term to the Kelly loss:** $\mathcal L_{\text{K10D}} = \mathcal L_{\text{K10C}} + \lambda_{\text{turn}} \|w_t - w_{t-1}\|_1$ — a direct tax on trading. Every rebalance pays a proportional cost against whatever Kelly gain it brings.
- **Why L1, not L2:** L1 is **non-smooth at zero** → creates a "dead zone" where no-trade strictly beats a tiny trade → **sparse rebalances**. L2 would give many small adjustments everywhere, which is exactly what we don't want. (Same reason Lasso is sparse and Ridge isn't.)
- **Why Adam handles the L1 kink without a proximal step:** Adam's per-coordinate running moment estimates smooth the discontinuity in practice — same trick that makes Lasso trainable with standard deep-learning libraries. No soft-thresholding step needed.
- **Results:** turnover **0.107 → 0.020 — roughly 5× reduction**. Log-wealth Δ still **+0.27** (was +0.46). MaxDD **−10.8%** (was −11.7%), essentially unchanged.
- **Why this trade-off is great:** cut 80% of trading, lose only 41% of edge. Savings-to-cost ratio is what makes K10D the right direction for fee-aware deployment — at 10 bps fees, the preserved edge nearly survives.
- **Why max-DD barely moves:** drawdown is about *directional exposure*, not trading frequency. K10D still bets the same direction as K10C, just updates less often. **Turnover and drawdown are fundamentally different axes of risk**; L1 fixes one, not the other.
- **What K10D does NOT fix:** bootstrap significance (#1) and over-levering (#3) — targeted at fees only. Fixing all three requires fee- and α-aware Kelly + multi-seed training (in next-steps list).

---

### Slide 19 — Attempted Fixes (25 sec)

*Figure:* either the same grouped bar (Pod M column), or `v8_turnover_histogram.png`.

- **What momentum pre-filtering does:** every bar, rank the 40 markets by **5-day absolute return** (120-bar lookback), keep only the **top-20 movers**, let Kelly optimize within that smaller universe. Re-rank and swap markets in/out each bar. Flags: `--momentum-top-n 20 --momentum-lookback-days 5`.
- **Why this could plausibly help Kelly:** Slide 10's diagnosis — 83–86% of variance in one factor. Markets with big recent moves are the factor-exposed ones. Top-20 by momentum **implicitly concentrates the universe on factor-exposed markets** while still letting the optimizer choose weights among them. Plain English: **less noise for Kelly to chew through**, smaller non-convex loss surface, fewer bad local minima.
- **The 2×2 factorial framing (say this out loud):** *"Objective × universe: {MVO, Kelly} × {full-40, top-20}. Prior pods covered three cells. Pod M is **the missing cell — Kelly × top-20**. This was the one cross-term no prior pod had run."*
- **Results:** Sortino Δ **+0.027**, log-wealth Δ **+0.22**, **Pr(Δ > 0) = 75.8%** via the same circular-block bootstrap from Slide 14.
- **What the combination tells us:** *sub-additive on gross edge* (+0.22 < K10C's +0.46) but *super-additive on risk-adjusted* (Sortino higher than either axis alone). Consistent with the "less noise" story — momentum screening reduces edge but improves risk-adjusted performance.
- **Why Pr(>0) is only 75.8% (vs K10C's 97.5%):** smaller edge is harder to distinguish from zero under the same bootstrap. Pod M is more interesting on *risk-adjusted* grounds, not raw-edge grounds.
- **Why the single-seed caveat is non-trivial:** Kelly's loss is non-convex (four non-linearities from Slide 12). Different seeds can hit different local optima. Pod M is literally `seed=7`. A different seed could give +0.10 or +0.30 or even negative. Elsewhere in the project (Slide 7 footnote) we ran 3-seed tests on MVO and got Δ Sortino −0.017 ± 0.0003 — very tight. We haven't done the same for Pod M. **That's the honest caveat.**
- Honest caveat to say out loud: *"Both K10D and Pod M-seed7 are **single-seed** results. Multi-seed confirmation is future work."*

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
