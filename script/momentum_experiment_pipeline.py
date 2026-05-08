"""Momentum pre-screening experiment pipeline.

Runs the constrained optimizer on momentum-screened market universes
and compares against the existing liquidity-selected control.

Variants:
  - control   : reuses existing week8 artifacts (40 markets, liquidity-selected)
  - mom_20_5d : top-20 markets by 5-day absolute momentum
  - mom_15_3d : top-15 markets by 3-day absolute momentum

Usage:
  # Quick sanity check (5 Optuna trials per variant)
  QUICK_SANITY_CHECK=1 python script/momentum_experiment_pipeline.py

  # Full run (100 trials per variant)
  python script/momentum_experiment_pipeline.py
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import (
    BaselineResult,
    bootstrap_sortino_ci,
    pretty_float,
    pretty_pct,
    run_equal_weight_baseline,
    save_baseline_outputs,
)
from src.constrained_optimizer import ExperimentConfig, run_optuna_search
from src.covariance_diagnostics import run_covariance_diagnostics
from src.polymarket_data import BuildConfig, NoMarketsAfterHistoryFilterError, build_dataset


QUICK = os.environ.get("QUICK_SANITY_CHECK", "0") == "1"
OPTUNA_N_TRIALS = 5 if QUICK else 100


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _control_build_config() -> BuildConfig:
    """40-market liquidity-selected control (same as the existing pipeline)."""
    return BuildConfig(
        max_events=1000,
        max_closed_events=4000,
        include_closed_events=True,
        events_page_limit=60,
        min_event_markets=1,
        min_history_points=24,
        min_history_days=24.0,
        max_markets=40,
        max_categories=120,
        per_category_market_cap=4,
        min_category_liquidity=0.0,
        excluded_category_slugs=(
            "hide-from-new", "parent-for-derivative",
            "earn-4", "pre-market", "rewards-20-4pt5-50",
        ),
        artifact_prefix="week8",
        history_interval="max",
        history_fidelity=10,
        use_cached_events_if_available=True,
        history_priority_enabled=True,
        history_priority_oversample_factor=5,
        momentum_screening_enabled=False,
    )


def _momentum_build_config(
    *,
    top_n: int,
    lookback_days: float,
    prefix: str,
) -> BuildConfig:
    """Momentum-screened variant: fetch wide, screen narrow."""
    return BuildConfig(
        max_events=1000,
        max_closed_events=4000,
        include_closed_events=True,
        events_page_limit=60,
        min_event_markets=1,
        min_history_points=24,
        min_history_days=24.0,
        max_markets=top_n,
        max_categories=120,
        per_category_market_cap=4,
        min_category_liquidity=0.0,
        excluded_category_slugs=(
            "hide-from-new", "parent-for-derivative",
            "earn-4", "pre-market", "rewards-20-4pt5-50",
        ),
        artifact_prefix=prefix,
        history_interval="max",
        history_fidelity=10,
        use_cached_events_if_available=True,
        history_priority_enabled=False,
        history_priority_oversample_factor=8,
        momentum_screening_enabled=True,
        momentum_lookback_days=lookback_days,
        momentum_top_n=top_n,
    )


def _experiment_config_for_n_markets(n_markets: int) -> ExperimentConfig:
    """Scale optimizer hyper-parameter ranges to the market count."""
    equal_w = 1.0 / max(n_markets, 1)

    if n_markets <= 15:
        max_weights = (0.12, 0.18, 0.25)
        domain_limits = (0.18, 0.25, 0.35)
        exposure_thresh = 0.25
    elif n_markets <= 20:
        max_weights = (0.10, 0.15, 0.20)
        domain_limits = (0.15, 0.20, 0.30)
        exposure_thresh = 0.20
    else:
        max_weights = (0.04, 0.06, 0.10, 0.15)
        domain_limits = (0.08, 0.12, 0.18, 0.25)
        exposure_thresh = 0.12

    fidelity = 10
    walkforward_train = 1440
    walkforward_test = 288

    return ExperimentConfig(
        learning_rates=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        penalties_lambda=(0.25, 0.5, 1.0, 2.0),
        rolling_windows=(24, 48, 96, 144, 288),
        steps_per_window=5,
        objective="mean_downside",
        variance_penalty=1.0,
        downside_penalty=2.0,
        variance_penalties=(0.5, 1.0, 2.0),
        downside_penalties=(1.0, 2.0, 3.0),
        optimizer_type="adam",
        evaluation_modes=("online",),
        primary_evaluation_mode="online",
        enable_two_stage_search=False,
        stage2_top_k=8,
        max_parallel_workers=1,
        early_prune_enabled=True,
        early_prune_exposure_factor=1.5,
        domain_limits=domain_limits,
        max_weights=max_weights,
        concentration_penalty_lambdas=(1.0, 3.0, 8.0, 15.0),
        covariance_penalty_lambdas=(0.5, 1.0, 5.0, 10.0),
        covariance_shrinkages=(0.02, 0.05, 0.10),
        entropy_lambdas=(0.0, 0.01),
        uniform_mixes=(0.0, 0.05, 0.1, 0.2),
        max_domain_exposure_threshold=exposure_thresh,
        holdout_fraction=0.2,
        walkforward_train_steps=walkforward_train,
        walkforward_test_steps=walkforward_test,
        seed=7,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_series(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sortino_np(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(returns) / (downside_dev + 1e-8))


def _build_with_backoff(
    project_root: pathlib.Path,
    base_config: BuildConfig,
    day_candidates: tuple[float, ...] = (24.0,),
) -> tuple[dict[str, pathlib.Path], float]:
    last_error: RuntimeError | None = None
    for min_days in day_candidates:
        cfg = BuildConfig(**{**base_config.__dict__, "min_history_days": float(min_days)})
        try:
            artifacts = build_dataset(project_root=project_root, config=cfg)
            return artifacts, float(min_days)
        except NoMarketsAfterHistoryFilterError as exc:
            last_error = exc
            print(f"  No markets at min_history_days={min_days}; retrying...")
    if last_error is not None:
        raise last_error
    raise RuntimeError("Dataset build failed.")


def _run_variant(
    project_root: pathlib.Path,
    label: str,
    build_cfg: BuildConfig,
    experiment_cfg: ExperimentConfig,
    n_trials: int,
    run_started: float,
) -> dict[str, Any]:
    """Run one full variant: data build → baseline → Optuna → diagnostics."""
    prefix = build_cfg.artifact_prefix
    processed = project_root / "data" / "processed"

    def _banner(name: str) -> None:
        elapsed = time.perf_counter() - run_started
        print(f"\n{'#'*60}", flush=True)
        print(f"  [{label}] {name}", flush=True)
        print(f"  (total elapsed: {elapsed / 60:.1f}m)", flush=True)
        print(f"{'#'*60}\n", flush=True)

    # 1. Data build
    _banner("Data Build")
    t0 = time.perf_counter()
    data_artifacts, used_min_days = _build_with_backoff(project_root, build_cfg)
    data_sec = time.perf_counter() - t0
    print(f"  Data build: {data_sec / 60:.1f}m, min_history_days_used={used_min_days}")

    # 2. Baseline
    _banner("Baseline")
    t0 = time.perf_counter()
    bl = run_equal_weight_baseline(project_root, artifact_prefix=prefix)
    save_baseline_outputs(project_root, bl, artifact_prefix=prefix)
    print(f"  Baseline: {bl.market_count} markets, sortino={pretty_float(bl.sortino)}, "
          f"max_dd={pretty_pct(bl.max_drawdown)}")

    # 3. Optuna
    _banner(f"Optuna ({n_trials} trials)")
    t0 = time.perf_counter()
    constrained_artifacts = run_optuna_search(
        project_root, artifact_prefix=prefix,
        config=experiment_cfg, n_trials=n_trials,
    )
    optuna_sec = time.perf_counter() - t0
    print(f"  Optuna: {optuna_sec / 60:.1f}m ({optuna_sec / 3600:.1f}h)")

    # 4. Covariance diagnostics
    _banner("Covariance Diagnostics")
    run_covariance_diagnostics(project_root, artifact_prefix=prefix)

    # 5. Extract holdout results for comparison
    constrained_metrics = _read_json(processed / f"{prefix}_constrained_best_metrics.json")
    baseline_metrics = _read_json(processed / f"{prefix}_baseline_metrics.json")
    baseline_ts = _read_series(processed / f"{prefix}_baseline_timeseries.csv")
    constrained_ts = _read_series(processed / f"{prefix}_constrained_best_timeseries.csv")
    cov_summary = _read_json(processed / f"{prefix}_covariance_summary.json")

    best_params = constrained_metrics.get("best_params", {})
    split_info = constrained_metrics.get("data_split", {})
    holdout_steps = int(split_info.get("holdout_steps_total", 0))

    bl_holdout_rets = np.array(
        [float(r["portfolio_return"]) for r in baseline_ts[-holdout_steps:]],
        dtype=float,
    ) if holdout_steps > 0 else np.array([], dtype=float)
    co_holdout_rets = np.array(
        [float(r["portfolio_return"]) for r in constrained_ts],
        dtype=float,
    )

    bl_sortino = _sortino_np(bl_holdout_rets)
    co_sortino = float(best_params.get("holdout_sortino_ratio", 0.0))

    # 6. Bootstrap CI
    _banner("Bootstrap CI")
    ci = bootstrap_sortino_ci(bl_holdout_rets, co_holdout_rets, n_bootstrap=2000, seed=42)
    print(f"  Sortino delta: {ci['mean_delta']:+.4f}  "
          f"95% CI: [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  "
          f"frac_positive: {ci['frac_positive']:.2%}")

    ci_path = processed / f"{prefix}_bootstrap_sortino_ci.json"
    ci_path.write_text(json.dumps(ci, indent=2), encoding="utf-8")

    # 7. Weight dispersion
    co_weights_raw = constrained_metrics.get("domain_exposure", {})
    co_weight_values = list(co_weights_raw.values()) if co_weights_raw else []
    bl_weight_values = list(baseline_metrics.get("exposure_by_domain", {}).values())

    return {
        "label": label,
        "prefix": prefix,
        "n_markets": bl.market_count,
        "baseline_sortino": bl_sortino,
        "constrained_sortino": co_sortino,
        "sortino_delta": co_sortino - bl_sortino,
        "baseline_max_dd": float(best_params.get("holdout_max_drawdown", bl.max_drawdown)),
        "constrained_max_dd": float(best_params.get("holdout_max_drawdown", 0.0)),
        "bootstrap_ci": ci,
        "avg_abs_corr": float(cov_summary.get("avg_abs_correlation", 0.0)),
        "max_abs_corr": float(cov_summary.get("max_abs_correlation", 0.0)),
        "variance_ratio": float(cov_summary.get("variance_ratio_constrained_vs_baseline", 0.0)),
        "constrained_weight_std": float(np.std(co_weight_values)) if co_weight_values else 0.0,
        "baseline_weight_std": float(np.std(bl_weight_values)) if bl_weight_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _write_comparison_report(
    project_root: pathlib.Path,
    results: list[dict[str, Any]],
) -> pathlib.Path:
    docs = project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# Momentum Screening Experiment Report")
    lines.append("")
    lines.append("## Holdout Performance Comparison")
    lines.append("")
    lines.append("| Variant | N | BL Sortino | CO Sortino | Δ | 95% CI | Frac+ |")
    lines.append("|---------|---|-----------|-----------|---|--------|-------|")
    for r in results:
        ci = r["bootstrap_ci"]
        lines.append(
            f"| {r['label']} | {r['n_markets']} "
            f"| {r['baseline_sortino']:.4f} | {r['constrained_sortino']:.4f} "
            f"| {r['sortino_delta']:+.4f} "
            f"| [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] "
            f"| {ci['frac_positive']:.1%} |"
        )

    lines.append("")
    lines.append("## Weight Dispersion")
    lines.append("")
    lines.append("| Variant | Baseline weight σ | Constrained weight σ |")
    lines.append("|---------|-------------------|---------------------|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['baseline_weight_std']:.6f} | {r['constrained_weight_std']:.6f} |"
        )

    lines.append("")
    lines.append("## Correlation Structure")
    lines.append("")
    lines.append("| Variant | Avg abs corr | Max abs corr | Variance ratio |")
    lines.append("|---------|-------------|-------------|----------------|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['avg_abs_corr']:.4f} | {r['max_abs_corr']:.4f} "
            f"| {r['variance_ratio']:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Sortino delta > 0** means the constrained optimizer outperformed equal-weight")
    lines.append("- **95% CI containing 0** means the difference is not statistically significant")
    lines.append("- **Higher weight σ** means the optimizer is deviating more from equal-weight")
    lines.append("- **Higher avg abs correlation** means there is more covariance structure to exploit")
    lines.append("")

    report_path = docs / "momentum_experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = REPO_ROOT
    run_started = time.perf_counter()
    results: list[dict[str, Any]] = []

    variants: list[tuple[str, BuildConfig, int]] = [
        ("control (40 mkts, liquidity)", _control_build_config(), 40),
        ("momentum top-20, 5d lookback", _momentum_build_config(top_n=20, lookback_days=5.0, prefix="mom_20_5d"), 20),
        ("momentum top-15, 3d lookback", _momentum_build_config(top_n=15, lookback_days=3.0, prefix="mom_15_3d"), 15),
    ]

    for label, build_cfg, n_markets in variants:
        print(f"\n{'='*60}", flush=True)
        print(f"  VARIANT: {label}", flush=True)
        print(f"{'='*60}", flush=True)

        exp_cfg = _experiment_config_for_n_markets(n_markets)
        result = _run_variant(
            project_root=project_root,
            label=label,
            build_cfg=build_cfg,
            experiment_cfg=exp_cfg,
            n_trials=OPTUNA_N_TRIALS,
            run_started=run_started,
        )
        results.append(result)

    report_path = _write_comparison_report(project_root, results)

    total = time.perf_counter() - run_started
    print(f"\n{'#'*60}", flush=True)
    print(f"  EXPERIMENT COMPLETE — {total / 3600:.1f}h total", flush=True)
    print(f"  Report: {report_path}", flush=True)
    print(f"{'#'*60}", flush=True)

    print("\n--- Quick Summary ---")
    for r in results:
        ci = r["bootstrap_ci"]
        print(f"  {r['label']:40s}  Δ={r['sortino_delta']:+.4f}  "
              f"CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]  "
              f"weight_σ={r['constrained_weight_std']:.4f}")


if __name__ == "__main__":
    main()
