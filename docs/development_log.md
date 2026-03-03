# Development Log

## Week 4 Prototype (Completed)

### Objective of this cycle
- Pivot from battery arbitrage to the proposal topic: cross-domain portfolio optimization on Polymarket.
- Follow execution-first order: code and experiments first, then notebook figures, then documentation.

### Work completed

1. **Data engineering pipeline (completed)**
   - Added `src/polymarket_data.py` with:
     - paginated event retrieval from Gamma API,
     - event-market flattening,
     - tag-to-domain mapping,
     - token-level price history pulls from CLOB API,
     - cached outputs in `data/raw` and `data/processed`,
     - quality checks (missingness, duplicates, monotonic timestamps, domain coverage).
   - Added package marker `src/__init__.py`.

2. **Baseline metrics (completed)**
   - Added `src/baseline.py` with equal-weight benchmark and metrics:
     - Sortino ratio,
     - max drawdown,
     - return volatility,
     - domain exposure shares.
   - Outputs persisted to:
     - `data/processed/baseline_metrics.json`
     - `data/processed/baseline_timeseries.csv`

3. **First constrained experiments (completed)**
   - Added `src/constrained_optimizer.py`:
     - rolling-window OGD/SGD-style updates in PyTorch,
     - differentiable domain-overexposure penalty,
     - grid search over learning rate, penalty lambda, and window length.
   - Outputs persisted to:
     - `data/processed/constrained_experiment_grid.csv`
     - `data/processed/constrained_best_metrics.json`
     - `data/processed/constrained_best_timeseries.csv`

4. **Reproducible orchestration + figures (completed)**
   - Added `script/polymarket_week8_pipeline.py`:
     - runs data build -> baseline -> constrained experiments,
     - generates Week 8 figure artifacts:
       - `figures/week8_equity_curve_comparison.png`
       - `figures/week8_drawdown_comparison.png`
       - `figures/week8_domain_exposure_comparison.png`

5. **Notebook packaging (completed)**
   - Added `notebooks/week8_flash_results.ipynb` to display metrics and figures from cached artifacts.

### Current empirical snapshot
- Markets retained: 7
- Price history points: 4,547
- Baseline: Sortino 0.0274, max drawdown -26.30%
- Best constrained run: Sortino 0.0115, max drawdown -61.92%

### What failed / limitations observed
- First constrained setup is currently less stable than baseline on risk metrics.
- Domain coverage is skewed toward crypto in this first data slice.
- Small number of retained markets limits diversification quality.

## Week 8 Iteration (Now In Progress)

### Planned Week 8 upgrades
1. Expand from coarse domains to many specific category tags.
2. Increase event universe and use category-balanced sampling.
3. Re-run baseline and constrained experiments with expanded categories.
4. Refresh notebook and figure outputs for flash presentation.

### Week 8 update (applied)
- Category mapping now uses specific tag slugs (not only coarse buckets).
- Latest run retained 80 markets and 80 categories selected from high-liquidity groups.
- Baseline now uses equal category weights by construction (each category gets the same total allocation).
- Artifacts are now consistently prefixed and renamed for Week 8 iteration:
  - `data/processed/week8_*`
  - `figures/week8_iteration_*`
  - `notebooks/week8_iteration_flash_results.ipynb`
