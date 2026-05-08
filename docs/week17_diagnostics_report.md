# Week 17 Diagnostics Report

## Run Context

- **Artifact prefix:** `week17`
- **Data lineage:** Processed artifacts cloned from `week11_v2` (same market universe as the Week 16 risk-regime run); pipeline stages applied under prefix `week17`.
- **Min history days used (after backoff):** `24.0`
- **Market count:** `40`
- **Tuning steps:** `48474`
- **Holdout steps:** `12119`
- **Holdout fraction:** `0.2`
- **Walk-forward (pipeline config):** `walkforward_train_steps` = `1440`, `walkforward_test_steps` = `288` (10‑minute bars)
- **Optimizer objective:** `excess_mean_downside` with **best-trial** surrogate weights **variance_penalty** = `0.5`, **downside_penalty** = `1.0` (see `week17_constrained_best_metrics.json`).
- **Selection metric:** `excess_sortino` (walk-forward tuning vs equal-weight baseline returns).
- **Large diagnostic export:** `data/processed/week17_domain_equity_signal_timeseries.csv.gz` (gzip’d CSV of per-step, per-domain aligned equity signals).

---

## Equity signal as a hedge: ETFs, single names, and PM topics

This project uses **listed equities in two complementary roles**: (1) **macro / regime conditioning** that tightens or loosens **portfolio concentration rules** in the optimizer, and (2) **topic-aligned reference returns** that let each Polymarket **category (domain)** be associated with an ETF or equity proxy—so the PM sleeve can be **conditioned on real-world risk** without assuming that stocks are a separate traded leg inside every experiment.

### A. Macro hedge layer — ETFs + volatility + defensive stocks (risk regime)

The scalar **`compute_risk_regime_zscore()`** combines **daily** yfinance returns over rolling windows (default **48 / 96 / 288 trading days**) into a score in **\[−1, 1\]** (positive ≈ risk-on). Components:

| Role | Tickers | Weight in score | Interpretation |
|------|---------|-----------------|----------------|
| Broad equity trend | **SPY** | **+0.3** | Higher recent return z vs history → loosen concentration (risk-on). |
| Risk / fear | **^VIX** | **−0.3** | Higher VIX return z → tighten diversification (risk-off). |
| Sector “momentum” basket | **XLE, XLV, XLF, QQQ, XLK** | **+0.2** (average z across names) | Energy, health care, financials, growth/tech proxy; positive z supports allowing winners more room when aligned with PM tilt logic. |
| Defensive / quality tilt | **WMT, COST, LLY, PG, KO** | **−0.2** (average z) | When defensives outperform strongly, score shifts **risk-off** → **tighter** `max_domain_exposure_threshold` and **stronger** concentration penalty—**hedging behavior** in the sense of **forcing diversification** when flight-to-quality dominates. |

The score maps to **`get_dynamic_concentration_params()`**: e.g. risk-on band → higher domain cap and lower concentration λ; risk-off → lower cap and higher λ (see `src/equity_signal.py`). This is **not** a separate portfolio return stream; it **reshapes the feasible set** for the Polymarket optimizer so that **ETFs and defensive single names jointly act as a regime filter**—analogous to using equity factors as a **risk overlay** on the PM book.

### B. Topic-aligned layer — “what stocks/ETFs go with which PM topics?”

For **per-market, per-domain** alignment inside **`build_asset_equity_signal_matrix`**, each **domain** (Polymarket category tag) is mapped to **one primary ticker**. The mapping comes from optional CSV `EquitySignalConfig.mapping_csv`, merged with code defaults **`DEFAULT_DOMAIN_TICKER_MAP`** in `src/equity_signal.py`:

| Coarse domain (ontology) | Default ETF / proxy |
|--------------------------|---------------------|
| `finance` | **XLF** |
| `economy` | **XLI** |
| `crypto` | **IBIT** |
| `politics`, `sports`, `other` | **SPY** |
| `science` | **XLK** |
| `culture` | **XLC** |

**Week 17 run (this artifact):** the saved map `data/processed/week17_domain_equity_ticker_map.csv` assigns **every listed category tag** (e.g. `epstein`, `formula1`, `crypto-prices`, `best-of-2025`, …) to **SPY**. So for this submission build, **all PM topics share the same broad U.S. equity hedge proxy**—**SPY’s previous-day-to-bar aligned simple returns**—rather than a separate ETF per tag. That is a deliberate **single-index hedge** choice for stability and data availability across heterogeneous topics; it can be overridden by supplying a richer CSV mapping.

**Optional richer mapping (used elsewhere in the repo):** `data/processed/pm_equity_domain_tilt_map.csv` illustrates **topic-specific** proxies for overlay scripts (e.g. **NVDA** for AI-related tags, **QQQ** for `big-tech`). Those rows are **not** the Week 17 per-domain series unless wired via `equity_signal_mapping_csv` in `ExperimentConfig`.

### C. How the equity signal enters the **combined** constrained PM strategy

Two mechanisms interact:

1. **`equity_signal_matrix` (aligned to PM steps):** Built from the domain→ticker map above. When `enable_equity_signal` is true, the optimizer can apply a **positive-part floor** on weights by domain using **`equity_signal_floor_scale`** (post-softmax **renormalization**), so domains with **stronger aligned equity signals** receive a **minimum weight share**—a **soft hedge / tilt** toward sectors that look supportive in cash equities. **`equity_signal_lambda`** scales an additional term in the objective path when enabled in Optuna.

2. **Best trial for Week 17:** **`equity_signal_lambda` = `0.0`** — the selected hyperparameters **turned off** the λ-weighted equity-signal reward channel for the winning trial, while the **risk-regime concentration policy** above still applied at the **experiment** level. The equity signal matrix and gzip’d time series remain **diagnostics** and support **alternative** trials where λ > 0.

Together, **ETF/defensive baskets** govern **how tightly** we allow PM concentration; **SPY (and optionally XLF, IBIT, NVDA, …)** govern **which directions** get a **minimum tilt** when λ and floor are active. That is the **combined strategy**: Polymarket optimization **plus** equity-based **regime overlay** **plus** optional **topic→proxy** alignment.

---

## Risk regime & concentration (applied to this run)

Values saved at pipeline start from `compute_risk_regime_zscore()` / `get_dynamic_concentration_params()` and merged into `ExperimentConfig` (see `data/processed/week17_pipeline_equity_regime.json`):

| Component | Value |
|-----------|--------|
| VIX-based regime | `low_vol` |
| Spot VIX (snapshot) | `17.94` |
| Regime scales (variance / downside / max-domain cap from `REGIME_PENALTY_SCALES`) | `0.5` / `1.0` / `0.18` |
| **ETF risk-regime z-score** | `0.0505` (neutral band → default concentration policy) |
| **Dynamic max domain exposure threshold** | `0.12` |
| **Dynamic concentration λ anchor** | `32.0` (merged into Optuna grid with `{2, 5, 10, 20, 32, 50}`) |
| Early-prune feasibility cap | `0.12` × factor (see optimizer); **feasible solution found:** yes |

---

## Optuna search summary

From `week17_constrained_best_metrics.json` → `optuna_summary`:

| Field | Value |
|-------|--------|
| **Sampler** | `QMCSampler` + `MedianPruner` (see `run_optuna_search` in `src/constrained_optimizer.py`) |
| **Trials requested** | `32` |
| **Completed** | `16` |
| **Pruned** | `16` |
| **Best trial index** | `1` |
| **Wall-clock search time** | ~`33088` s (~9.2 h) |

**Best trial (holdout champion):** `equity_signal_lambda` = **`0.0`** — equity signal paths were available, but this trial **did not** put weight on the λ-scaled equity signal term; **regime-based concentration** still reflects the ETF/defensive/VIX construction above.

---

## Holdout Performance Comparison

**Equal-weight baseline vs constrained optimizer on the holdout segment** (aligned evaluation window).

| Metric | Baseline | Constrained | Delta |
|--------|----------|-------------|-------|
| Sortino ratio | 0.1798 | 0.1538 | −0.0260 |
| Max drawdown | −5.4547% | −9.3999% | −3.9452 pp |
| Mean return | 0.00018792 | 0.00028244 | **+0.00009452** |
| Volatility | 0.00522424 | 0.00905327 | +0.00382903 |

**Reading:** Sortino **lags** the baseline on holdout; **mean step return is higher** with **higher volatility and deeper drawdowns** — a mean–risk trade, not a uniform win on risk-adjusted return.

---

## Full-Series Baseline Reference

From `week17_baseline_metrics.json` (equal-weight over full cached series):

- **Baseline Sortino (full):** `0.0520`
- **Baseline max drawdown (full):** `−8.2541%`
- **Baseline mean step return (full):** `0.00012349`
- **Baseline volatility (full):** `0.004439`

---

## Attribution — What Drove Returns

Holdout attribution from `week17_constrained_best_attribution_summary.json` / market contributions CSV.

**Biggest single market:** Will Lucas rank #1 among boy names on the SSA’s official list for 2025? (`best-of-2025`) — **98.2%** of total portfolio contribution (resolution-driven dominance).

**Biggest domain:** `best-of-2025` — **98.2%** of total contribution.

### Top 10 Market Contributors

| # | Market | Domain | Contribution | Share | Weight |
|---|--------|--------|---------------|-------|--------|
| 1 | Will Lucas rank #1 among boy names on the SSA’s official list for 2025? | `best-of-2025` | 3.362122 | 98.2% | 0.0329 |
| 2 | Will Juan Manuel Galán win the 2026 Colombian presidential election? | `colombia-election` | 0.085205 | 2.5% | 0.0285 |
| 3 | Will the Republicans win the Colorado Senate race in 2026? | `colorado-midterm` | 0.059443 | 1.7% | 0.0223 |
| 4 | Will Claude 5 be released by April 30, 2026? | `claude-5` | −0.040266 | −1.2% | 0.0077 |
| 5 | Felix Protocol FDV above $300M one day after launch? | `felix` | −0.030915 | −0.9% | 0.0229 |
| 6 | Will Haiti win the 2026 FIFA World Cup? | `2026-fifa-world-cup` | −0.013928 | −0.4% | 0.0199 |
| 7 | Epstein client list released by June 30? | `epstein` | 0.011220 | 0.3% | 0.0759 |
| 8 | Extended FDV above $300M one day after launch? | `extended` | 0.007832 | 0.2% | 0.0308 |
| 9 | Will Bernie endorse Kshama Sawant for WA-09 by Nov 2 2026 ET? | `bernie-sanders` | 0.006218 | 0.2% | 0.0112 |
| 10 | Will Arsenal win the 2025–26 Champions League? | `champions-league` | −0.006113 | −0.2% | 0.0136 |

### Top 10 Domain Contributors

| # | Domain | Contribution | Share |
|---|--------|---------------|-------|
| 1 | `best-of-2025` | 3.362122 | 98.2% |
| 2 | `colombia-election` | 0.085205 | 2.5% |
| 3 | `colorado-midterm` | 0.059443 | 1.7% |
| 4 | `claude-5` | −0.040266 | −1.2% |
| 5 | `felix` | −0.030915 | −0.9% |
| 6 | `2026-fifa-world-cup` | −0.013928 | −0.4% |
| 7 | `epstein` | 0.011220 | 0.3% |
| 8 | `extended` | 0.007832 | 0.2% |
| 9 | `bernie-sanders` | 0.006218 | 0.2% |
| 10 | `champions-league` | −0.006113 | −0.2% |

### Top 5 Correlated Contributor Pairs (token-level)

From `week17_constrained_best_top_market_correlation_pairs.csv` (first rows by \|corr\|):

| Market A (short label) | Market B (short label) | Correlation |
|------------------------|------------------------|-------------|
| Bernie / WA-09 endorsement | Connecticut Gov. primary (Erin Stewart) | −0.0723 |
| Bernie / WA-09 endorsement | Europe wins 2026 FIFA World Cup | −0.0270 |
| Claude 5 release by Apr 30, 2026 | Brazil runoff — Lula qualifies | 0.0196 |
| Extended FDV > \$300M (launch day) | Arsenal wins 2025–26 Champions League | −0.0162 |
| 2026 second-hottest year on record | Base FDV > \$2B one day after launch | −0.0114 |

*(Labels resolved from token IDs via `week17_constrained_best_market_return_contributions.csv`.)*

---

## Correlation and Risk Structure

From `week17_covariance_summary.json`:

- **Category count:** `40`
- **Avg abs category correlation:** `0.00088`
- **Max abs category correlation:** `0.0543`
- **Top eigenvalue share:** `0.5408`
- **Variance ratio constrained vs baseline:** `1.1640`

---

## Figure Index (Week 17)

Pipeline-style figures (baseline vs constrained / “equity hedge portfolio” labeling):

- `figures/week17_equity_hedge_portfolio_equity_curve_comparison.png`
- `figures/week17_equity_hedge_portfolio_drawdown_comparison.png`
- `figures/week17_equity_hedge_portfolio_category_exposure_comparison.png`
- `figures/week17_equity_hedge_portfolio_rolling_mean_return_comparison.png`
- `figures/week17_equity_hedge_portfolio_return_distribution_comparison.png`
- `figures/week17_equity_hedge_portfolio_top_exposure_deltas.png`
- `figures/week17_equity_hedge_portfolio_risk_return_snapshot.png`

---

## Interpretation Checklist

- [ ] Constrained holdout Sortino beats baseline (−0.0260)
- [ ] Constrained holdout drawdown better than baseline (−3.9452 pp)
- [x] Top contributor pairs not excessively correlated (max abs corr: 0.0543)
- [ ] No single domain dominates returns (top domain share: 98.2%)
- [x] ETF / VIX / defensive regime + dynamic concentration recorded (`week17_pipeline_equity_regime.json`)
- [x] Domain→equity proxy map archived (`week17_domain_equity_ticker_map.csv`; **Week 17:** all mapped categories → **SPY**)
- [x] Optuna QMC search completed with feasible domain-exposure filter

---

*Generated from `data/processed/week17_*` metrics and attribution artifacts; holdout comparison aligns with `week17_constrained_best_metrics.json`. Equity-instrument definitions follow `src/equity_signal.py`.*
