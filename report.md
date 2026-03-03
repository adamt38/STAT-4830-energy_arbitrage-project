# Project Report: Cross-Domain Portfolio Optimization on Polymarket

## Problem Statement

### What are you optimizing?
We optimize a portfolio of Polymarket prediction contracts to maximize risk-adjusted return while penalizing over-allocation to a single event domain (for example politics or crypto).

### Why does this matter?
Prediction markets are exposed to event clustering risk. A portfolio that appears diversified by the number of positions may still be concentrated in one real-world driver. Our project tests whether explicit domain-level constraints reduce tail risk versus a naive equal-weight allocation.

### How do we measure success?
Success is measured against an equal-weight baseline using:
- Sortino ratio (risk-adjusted return),
- maximum drawdown,
- domain exposure concentration.

### Data and constraints
- **Data source:** `gamma-api.polymarket.com/events` (market discovery and tags) + `clob.polymarket.com/prices-history` (token price history).
- **Optimization constraint:** penalize any domain exposure above a limit `L_k` via a differentiable quadratic penalty.
- **Project objective:** maximize `Sortino_t(R_p) - lambda * sum_k max(0, S_k - L_k)^2`.

### Risks
- API coverage may be uneven by market and domain.
- Domain mapping from tags is noisy and needs normalization.
- First-pass online optimization may be unstable and overfit.

---

## Technical Approach

### Data pipeline
We implemented a reproducible pipeline in `src/polymarket_data.py` that:
1. pulls paginated active events,
2. flattens event/market records,
3. maps event tags to coarse domains,
4. selects binary market tokens,
5. fetches token-level historical prices,
6. caches outputs in `data/raw` and `data/processed`.

### Baseline
`src/baseline.py` computes:
- equal-weight market allocation,
- portfolio return series,
- cumulative return,
- max drawdown,
- Sortino ratio,
- domain exposure shares.

### Constrained optimizer
`src/constrained_optimizer.py` implements a first OGD/SGD-style online routine in PyTorch:
- rolling-window updates,
- softmax weights over markets,
- Sortino-based objective with domain-penalty term,
- small grid search over learning rate, penalty lambda, and window length.

---

## Week 4 Prototype Results (Completed Work So Far)

Artifacts were generated through `script/polymarket_week8_pipeline.py`.

### Dataset quality snapshot
- markets retained: **19**
- history points: **12,509**
- missing-history markets: **0**
- non-monotonic token series: **0**
- duplicate timestamp-token points: **0**

### Baseline (equal-weight)
- Sortino: **0.0706**
- Max drawdown: **-14.35%**
- Mean return per step: **0.000309**
- Volatility: **0.01451**
- Category exposure now spans many categories, including: `us-presidential-election`, `world-elections`, `global-elections`, `sports`, `soccer`, `serie-a`, `la-liga`, `movies`, `jerome-powell`, `airdrops`, `politics`, `world`, and `elections`.

### Constrained first iteration (best grid point)
- learning rate: **0.05**
- lambda penalty: **10.0**
- rolling window: **48**
- Sortino: **0.0148**
- Max drawdown: **-84.18%**
- Mean return per step: **0.000729**
- Volatility: **0.03591**

### Interpretation
The constrained model currently does **not** beat baseline on risk-adjusted quality or drawdown. This is our Week 4 prototype baseline and establishes a working benchmark for the next iteration.

---

## Week 8 Iteration (Current Work)

### Week 8 achieved updates
1. Expanded from coarse domains to high-liquidity tag categories.
2. Rebuilt dataset with category-balanced selection targeting 50-100 categories.
3. Implemented category-equal baseline weighting (equal total allocation per category).
4. Regenerated artifacts with renamed Week 8 files (`week8_*` and `week8_iteration_*`).

### Week 8 latest metrics
- markets retained: **80**
- categories retained: **80**
- baseline category exposure: **equal at 1.25% per category**
- baseline Sortino: **0.0730**
- baseline max drawdown: **-4.73%**
