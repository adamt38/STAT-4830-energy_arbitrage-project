# Momentum Pre-Screening: Mechanism and Initial Results

## 1. Problem Recap

The constrained portfolio optimizer consistently converges to near-equal-weight allocation across all configurations tested (Week 4 through Week 9). The holdout Sortino delta between constrained and baseline has been negative in every run:

| Run | N Markets | Sortino Delta |
|-----|-----------|---------------|
| Week 4 | 19 | −0.056 |
| Week 8 | 177 | −0.149 |
| Week 9 v1 | ~260 | −0.008 |
| Week 9 v2 | ~260 | −0.014 |
| Week 9 latest | 40 | −0.009 |

Three structural reasons explain this:

1. **Near-zero cross-asset correlation.** Average absolute category correlation is 0.0008, maximum is 0.035. For uncorrelated assets with similar expected returns, equal-weight (1/N) is already the minimum-variance portfolio. There is no covariance structure for the optimizer to exploit.

2. **Weak return signal relative to penalty gradients.** With 40 markets and ~24 days of 10-minute data, each market carries 2.5% of the portfolio. The penalty terms (domain lambda ~1.9, concentration lambda ~32, covariance lambda ~10) create gradients that consistently overpower the return signal at every update step.

3. **Most markets contribute nothing.** Of the current 40 markets, 10 had zero price movement over the last 5 days, and a further 10 had absolute momentum below 0.06. Meanwhile, a single market (Databricks IPO) drove 41.7% of total portfolio return. The majority of the universe is dead weight — contributing noise but no signal.

## 2. What Momentum Screening Does

Momentum screening is a pre-filter that selects markets based on recent price movement before passing them to the optimizer. It does not change how the optimizer works — it changes what data the optimizer sees.

### The mechanism

Given a set of candidate markets with price histories:

**Step 1 — Compute momentum for each market.**
For each market, take the price series over the last N days (the "lookback window"). Compute the cumulative return:

```
momentum = (last_price - first_price) / first_price
```

A market that went from $0.50 to $0.60 has momentum +0.20. A market that went from $0.80 to $0.70 has momentum −0.125. A market that stayed at $0.50 has momentum 0.

**Step 2 — Rank by absolute momentum.**
We rank markets by |momentum|, not by directional momentum. We want markets that are *moving*, regardless of direction. Whether a market is surging or collapsing, it has signal. A flat market has none. The optimizer's job is to decide which direction to bet — the screener's job is to find markets where there is something happening.

**Step 3 — Select the top K.**
Keep the top K markets (e.g., 15 or 20) and discard the rest. Pass only these markets to the optimizer.

### What this changes

| Property | Before (40 markets, liquidity-selected) | After (top-20, momentum-screened) |
|----------|----------------------------------------|-----------------------------------|
| Selection criterion | Liquidity + history length | Recent price movement |
| Per-market weight | 2.5% (1/40) | 5.0% (1/20) |
| Signal quality | Mixed (includes 10 flat markets) | All markets have recent movement |
| Optimizer degrees of freedom | 40 parameters | 20 parameters |

## 3. How This Addresses the Equal-Weight Problem

Each of the three structural problems identified in Section 1 maps to a specific improvement from momentum screening:

**Problem: Weak return signal.**
Momentum screening removes markets with zero or near-zero recent movement. The 10 flat markets in the current universe (Arkansas Senate, NBA MVP, Champions League, Formula 1, etc.) contribute exactly zero return signal — they are priced at a stable level and their prices do not move within the data window. By removing them, the surviving universe has a mean absolute momentum of 1.32 (top-20, 5-day) vs. 0.66 for the full set. The return gradient the optimizer receives is stronger because every remaining market has actual price dynamics to learn from.

**Problem: Too many parameters for the data.**
Reducing from 40 to 15–20 markets means each weight parameter controls a larger share of the portfolio. A tilt from 5% to 10% on a single market is a portfolio-level effect the optimizer can detect above step-to-step noise. With 40 markets, the same relative tilt (2.5% to 5%) is half the magnitude and more easily overwhelmed by noise.

**Problem: Penalty gradients dominate the return gradient.**
With fewer markets carrying stronger signal, the return gradient has a better chance of competing against the penalty terms. The constraint thresholds also become meaningful: with 20 markets, a 15% domain limit can actually bind (equal weight is 5%), whereas with 40 markets a 12% limit never binds (equal weight is 2.5%).

## 4. Screening Results on Current Data

The following results were computed on the existing `week8_price_history.csv` dataset (174,750 price observations, 40 markets, 10-minute fidelity, ~28 days of history).

### Full 40-market ranking by 5-day absolute momentum

| Rank | Market | Domain | Momentum | Direction |
|------|--------|--------|----------|-----------|
| 1 | Will Databricks' market cap be >= $250B at IPO close? | databricks | 20.1429 | UP |
| 2 | Will Claude 5 be released by March 31, 2026? | claude-5 | 0.9677 | DOWN |
| 3 | Trump out as President by March 31? | epstein | 0.7500 | DOWN |
| 4 | Will Bayrou win the 2027 French presidential election? | france | 0.7500 | UP |
| 5 | Will Janet Mills be Dem nominee for Senate in Maine? | democratic-primary | 0.5526 | DOWN |
| 6 | US national Bitcoin reserve before 2027? | bitcoin | 0.5000 | DOWN |
| 7 | Will Stuttgart win the 2025-26 Bundesliga? | bundesliga | 0.5000 | UP |
| 8 | Will Steve Hilton win CA Governor Election 2026? | california-midterm | 0.4298 | UP |
| 9 | Will AfD win most seats in 2026 Berlin state elections? | german-elections | 0.2687 | UP |
| 10 | Will Alphabet be largest company by market cap on Dec 31? | business | 0.2400 | UP |
| 11 | Will Anthropic market cap be < $100B at IPO close? | anthropic | 0.2000 | DOWN |
| 12 | MicroStrategy sells any Bitcoin by June 30, 2026? | 2025-predictions | 0.1842 | DOWN |
| 13 | Will Trump pardon Ghislaine Maxwell by end of 2026? | ghislaine-maxwell | 0.1538 | UP |
| 14 | Opensea FDV above $500M one day after launch? | fdv | 0.1386 | DOWN |
| 15 | Will Bitmine hold more than 7M ETH before 2027? | ethereum | 0.1333 | UP |
| 16 | Will LA Clippers make the NBA Playoffs? | basketball | 0.0873 | UP |
| 17 | New pandemic in 2026? | climate-science | 0.0870 | UP |
| 18 | Elon Musk trillionaire before 2027? | elon-musk | 0.0855 | DOWN |
| 19 | Will Bitcoin reach $250K by Dec 31, 2026? | crypto-prices | 0.0690 | DOWN |
| 20 | Gensyn FDV above $800M one day after launch? | gensyn | 0.0667 | UP |
| 21 | Extended FDV above $500M one day after launch? | extended | 0.0606 | UP |
| 22 | Felix Protocol FDV above $25M one day after launch? | felix | 0.0526 | UP |
| 23 | Will North America win the 2026 FIFA World Cup? | fifa-world-cup | 0.0392 | DOWN |
| 24 | Will GOP hold exactly 54 Senate seats after 2026? | congress | 0.0220 | DOWN |
| 25 | MegaETH FDV > $1B one day after launch? | airdrops | 0.0189 | UP |
| 26 | Will Bitcoin have the best performance in 2026? | commodities | 0.0185 | DOWN |
| 27 | Will Zohran Mamdani announce Presidential run before 2027? | celebrities | 0.0143 | UP |
| 28 | Will Brazil win the 2026 FIFA World Cup? | 2026-fifa-world-cup | 0.0114 | DOWN |
| 29 | Will Tarcisio de Frietas qualify for Brazil runoff? | brazil | 0.0103 | UP |
| 30 | Will Dems win CO Senate race in 2026? | colorado-midterm | 0.0054 | DOWN |
| 31 | Will Dems win AR Senate race in 2026? | arkansas-midterm | 0.0000 | FLAT |
| 32 | Will Tyrese Maxey win 2025-26 NBA MVP? | awards | 0.0000 | FLAT |
| 33 | Will Lucas rank #1 boy name on SSA list for 2025? | best-of-2025 | 0.0000 | FLAT |
| 34 | Will Tesla be largest company by market cap on March 31? | big-tech | 0.0000 | FLAT |
| 35 | Will Club Brugge win 2025-26 Champions League? | champions-league | 0.0000 | FLAT |
| 36 | Will David Luna Sanchez win 2026 Colombian election? | colombia-election | 0.0000 | FLAT |
| 37 | Will GOP win DE Senate race in 2026? | delaware-midterm | 0.0000 | FLAT |
| 38 | Will Lance Stroll be 2026 F1 Drivers' Champion? | formula1 | 0.0000 | FLAT |
| 39 | Will Mistral have the best AI model end of March 2026? | gemini-3 | 0.0000 | FLAT |
| 40 | Will Trump win Nobel Peace Prize in 2026? | geopolitics | 0.0000 | FLAT |

### Screening summary statistics

| Metric | Full 40 | Top-20 (5d) | Top-15 (3d) |
|--------|---------|-------------|-------------|
| Market count | 40 | 20 | 15 |
| Unique domains | 40 | 20 | 15 |
| Flat markets (zero momentum) | 10 | 0 | 0 |
| Mean absolute momentum | 0.664 | 1.315 | — |
| Median absolute momentum (non-zero) | 0.110 | — | — |
| Min absolute momentum in selected set | — | 0.067 | 0.071 |

### Key observations

1. **10 of 40 markets (25%) are completely flat.** These markets had zero price change over the 5-day lookback. They are typically long-dated outcomes with low trading activity (e.g., Formula 1 championship, Champions League, Nobel Peace Prize). Including them in the optimizer adds parameters to estimate but contributes no return signal.

2. **The screener correctly identifies resolving events.** Markets approaching their resolution date show high momentum: Claude 5 (March 31 deadline, momentum −0.97 as price collapses toward 0), Trump out as President by March 31 (momentum −0.75), Databricks IPO (momentum +20.14 as price surges toward 1). These are exactly the markets with strong directional signal.

3. **Category diversity is preserved.** The top-20 selection spans 20 unique domains (each market in this dataset belongs to its own domain). No domain concentration is introduced by the screening because high-momentum markets are distributed across categories (politics, sports, crypto, tech, elections).

4. **The 3-day and 5-day lookbacks produce different rankings.** For example, the Congress market (GOP Senate seats) ranks #24 on 5-day lookback but #5 on 3-day lookback, indicating a very recent price move. This confirms the lookback parameter captures different time horizons of momentum.

## 5. What Remains To Be Done

The momentum screening mechanism is implemented and verified. The following steps require running the full optimizer pipeline:

1. **Run the constrained optimizer (Optuna, 100 trials) on the momentum-screened universes** — top-20 with 5-day lookback and top-15 with 3-day lookback. Each run takes approximately 10–18 hours on an M2 Air.

2. **Compare holdout performance** — Sortino ratio, max drawdown, and mean return for baseline vs. constrained under each screening variant.

3. **Measure weight dispersion** — compute the standard deviation of optimizer weights. If momentum screening provides enough signal for the optimizer to learn non-trivial tilts, weight dispersion should be materially higher than the near-zero values observed in the current 40-market setup.

4. **Bootstrap confidence intervals** — resample holdout returns to determine whether any Sortino difference is statistically significant or within noise.

## Data Sources

All numbers in this report come from:
- `data/processed/week8_markets_filtered.csv` — 40 markets selected by liquidity + history length
- `data/processed/week8_price_history.csv` — 174,750 price observations at 10-minute fidelity
- `data/processed/week8_covariance_summary.json` — correlation structure
- `docs/week9_diagnostics_report.md` — attribution and holdout comparison
- `docs/development_log.md` — historical run results

Implementation code:
- `src/polymarket_data.py` — `compute_momentum_scores()` and `select_by_momentum()` functions
- `script/momentum_experiment_pipeline.py` — full experiment orchestration
