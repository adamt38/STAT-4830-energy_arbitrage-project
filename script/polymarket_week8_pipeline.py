"""End-to-end Week 8 pipeline after Week 4 prototype."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import pretty_float, pretty_pct, run_equal_weight_baseline, save_baseline_outputs
from src.constrained_optimizer import ExperimentConfig, run_experiment_grid
from src.covariance_diagnostics import run_covariance_diagnostics
from src.polymarket_data import BuildConfig, build_dataset


def _read_series(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _make_figures(project_root: pathlib.Path) -> dict[str, pathlib.Path]:
    processed = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_series = _read_series(processed / "week8_baseline_timeseries.csv")
    constrained_series = _read_series(processed / "week8_constrained_best_timeseries.csv")
    if constrained_series:
        # Align baseline and constrained plots to the same holdout horizon.
        baseline_series = baseline_series[-len(constrained_series) :]

    baseline_ret = [float(row["portfolio_return"]) for row in baseline_series]
    baseline_cum: list[float] = []
    running = 1.0
    for value in baseline_ret:
        running *= 1.0 + value
        baseline_cum.append(running)
    peak = 1.0
    baseline_dd: list[float] = []
    for value in baseline_cum:
        peak = max(peak, value)
        baseline_dd.append((value / peak) - 1.0)
    constrained_cum = [float(row["cumulative_return"]) for row in constrained_series]
    constrained_dd = [float(row["drawdown"]) for row in constrained_series]
    constrained_ret = [float(row["portfolio_return"]) for row in constrained_series]

    fig_equity, ax_equity = plt.subplots()
    ax_equity.plot(baseline_cum, label="Equal-weight baseline")
    ax_equity.plot(constrained_cum, label="Constrained OGD/SGD")
    ax_equity.set_title("Portfolio Cumulative Return")
    ax_equity.set_xlabel("Step")
    ax_equity.set_ylabel("Growth of $1")
    ax_equity.legend()
    equity_path = figures_dir / "week8_iteration_equity_curve_comparison.png"
    fig_equity.savefig(equity_path, dpi=120)
    plt.close(fig_equity)

    fig_dd, ax_dd = plt.subplots()
    ax_dd.plot(baseline_dd, label="Equal-weight baseline")
    ax_dd.plot(constrained_dd, label="Constrained OGD/SGD")
    ax_dd.set_title("Portfolio Drawdown")
    ax_dd.set_xlabel("Step")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.legend()
    dd_path = figures_dir / "week8_iteration_drawdown_comparison.png"
    fig_dd.savefig(dd_path, dpi=120)
    plt.close(fig_dd)

    with (processed / "week8_baseline_metrics.json").open("r", encoding="utf-8") as handle:
        baseline_metrics = json.load(handle)
    with (processed / "week8_constrained_best_metrics.json").open("r", encoding="utf-8") as handle:
        constrained_metrics = json.load(handle)

    baseline_exp = baseline_metrics.get("exposure_by_domain", {})
    constrained_exp = constrained_metrics.get("domain_exposure", {})
    all_domains = sorted(set(baseline_exp.keys()) | set(constrained_exp.keys()))
    exposure_rows = [
        {
            "category": domain,
            "baseline_exposure": float(baseline_exp.get(domain, 0.0)),
            "constrained_exposure": float(constrained_exp.get(domain, 0.0)),
            "delta_exposure": float(constrained_exp.get(domain, 0.0)) - float(baseline_exp.get(domain, 0.0)),
        }
        for domain in all_domains
    ]
    exposure_table_path = processed / "week8_category_exposure_table.csv"
    with exposure_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "baseline_exposure", "constrained_exposure", "delta_exposure"],
        )
        writer.writeheader()
        writer.writerows(
            sorted(exposure_rows, key=lambda row: row["constrained_exposure"], reverse=True)
        )

    # Use top categories so the chart remains readable.
    top_n = 25
    ranked_domains = [
        row["category"]
        for row in sorted(
            exposure_rows,
            key=lambda row: max(row["baseline_exposure"], row["constrained_exposure"]),
            reverse=True,
        )[:top_n]
    ]
    baseline_vals = [float(baseline_exp.get(domain, 0.0)) for domain in ranked_domains]
    constrained_vals = [float(constrained_exp.get(domain, 0.0)) for domain in ranked_domains]

    x = range(len(ranked_domains))
    width = 0.4
    fig_exp, ax_exp = plt.subplots(figsize=(13, 5))
    ax_exp.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
    ax_exp.bar([i + width / 2 for i in x], constrained_vals, width=width, label="Constrained")
    ax_exp.set_title("Category Exposure Comparison")
    ax_exp.set_ylabel("Portfolio Weight")
    ax_exp.set_xticks(list(x))
    ax_exp.set_xticklabels(ranked_domains, rotation=35, ha="right")
    ax_exp.legend()
    exposure_path = figures_dir / "week8_iteration_category_exposure_comparison.png"
    fig_exp.tight_layout()
    fig_exp.savefig(exposure_path, dpi=120)
    plt.close(fig_exp)

    # Rolling average step returns (holdout-aligned) for smoother trend comparison.
    rolling_window = 96

    def _rolling_mean(values: list[float], window: int) -> list[float]:
        if not values:
            return []
        if len(values) < window:
            m = float(np.mean(values))
            return [m for _ in values]
        arr = np.asarray(values, dtype=float)
        kernel = np.ones(window, dtype=float) / float(window)
        rolled = np.convolve(arr, kernel, mode="valid")
        prefix = [float(rolled[0])] * (window - 1)
        return prefix + rolled.tolist()

    baseline_roll = _rolling_mean(baseline_ret, rolling_window)
    constrained_roll = _rolling_mean(constrained_ret, rolling_window)
    fig_roll, ax_roll = plt.subplots()
    ax_roll.plot(baseline_roll, label="Baseline", linewidth=2.0)
    ax_roll.plot(constrained_roll, label="Constrained", linewidth=2.0)
    ax_roll.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax_roll.set_title("Rolling Mean Step Return (Holdout)")
    ax_roll.set_xlabel("Step")
    ax_roll.set_ylabel("Mean Return")
    ax_roll.legend()
    rolling_path = figures_dir / "week8_iteration_rolling_mean_return_comparison.png"
    fig_roll.tight_layout()
    fig_roll.savefig(rolling_path, dpi=120)
    plt.close(fig_roll)

    # Return distribution comparison for risk shape.
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(baseline_ret, bins=60, alpha=0.5, label="Baseline")
    ax_hist.hist(constrained_ret, bins=60, alpha=0.5, label="Constrained")
    ax_hist.axvline(float(np.mean(baseline_ret)), linestyle="--", linewidth=1.0, color="tab:blue")
    ax_hist.axvline(float(np.mean(constrained_ret)), linestyle="--", linewidth=1.0, color="tab:orange")
    ax_hist.set_title("Step Return Distribution (Holdout)")
    ax_hist.set_xlabel("Step Return")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    hist_path = figures_dir / "week8_iteration_return_distribution_comparison.png"
    fig_hist.tight_layout()
    fig_hist.savefig(hist_path, dpi=120)
    plt.close(fig_hist)

    # Top category exposure shifts (constrained - baseline).
    exposure_sorted = sorted(exposure_rows, key=lambda row: abs(row["delta_exposure"]), reverse=True)[:20]
    exp_labels = [row["category"] for row in exposure_sorted]
    exp_deltas = [row["delta_exposure"] for row in exposure_sorted]
    fig_delta, ax_delta = plt.subplots(figsize=(12, 5))
    colors = ["tab:green" if val >= 0 else "tab:red" for val in exp_deltas]
    ax_delta.bar(range(len(exp_labels)), exp_deltas, color=colors)
    ax_delta.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax_delta.set_xticks(list(range(len(exp_labels))))
    ax_delta.set_xticklabels(exp_labels, rotation=35, ha="right")
    ax_delta.set_title("Top Category Exposure Deltas (Constrained - Baseline)")
    ax_delta.set_ylabel("Delta Weight")
    exposure_delta_path = figures_dir / "week8_iteration_top_exposure_deltas.png"
    fig_delta.tight_layout()
    fig_delta.savefig(exposure_delta_path, dpi=120)
    plt.close(fig_delta)

    # Two-point risk/return snapshot for quick slide communication.
    baseline_mean = float(baseline_metrics.get("mean_return", 0.0))
    baseline_vol = float(baseline_metrics.get("volatility", 0.0))
    constrained_mean = float(constrained_metrics.get("best_params", {}).get("holdout_mean_return", 0.0))
    constrained_vol = float(constrained_metrics.get("best_params", {}).get("holdout_volatility", 0.0))
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter([baseline_vol], [baseline_mean], s=130, label="Baseline")
    ax_scatter.scatter([constrained_vol], [constrained_mean], s=130, label="Constrained (holdout)")
    ax_scatter.set_title("Risk-Return Snapshot")
    ax_scatter.set_xlabel("Volatility")
    ax_scatter.set_ylabel("Mean Step Return")
    ax_scatter.legend()
    risk_return_path = figures_dir / "week8_iteration_risk_return_snapshot.png"
    fig_scatter.tight_layout()
    fig_scatter.savefig(risk_return_path, dpi=120)
    plt.close(fig_scatter)

    return {
        "equity_curve": equity_path,
        "drawdown_curve": dd_path,
        "domain_exposure": exposure_path,
        "category_exposure_table": exposure_table_path,
        "rolling_mean_return": rolling_path,
        "return_distribution": hist_path,
        "top_exposure_deltas": exposure_delta_path,
        "risk_return_snapshot": risk_return_path,
    }


def main() -> None:
    project_root = REPO_ROOT

    data_artifacts = build_dataset(
        project_root=project_root,
        config=BuildConfig(
            max_events=500,
            events_page_limit=60,
            min_history_points=12,
            min_history_days=21.0,
            max_markets=100,
            max_categories=80,
            per_category_market_cap=1,
            min_category_liquidity=150000.0,
            excluded_category_slugs=(
                "hide-from-new",
                "parent-for-derivative",
                "earn-4",
                "pre-market",
                "rewards-20-4pt5-50",
            ),
            artifact_prefix="week8",
            history_interval="max",
            history_fidelity=60,
        ),
    )
    print("Data artifacts:")
    for key, value in data_artifacts.items():
        print(f"- {key}: {value}")

    baseline_result = run_equal_weight_baseline(project_root, artifact_prefix="week8")
    baseline_artifacts = save_baseline_outputs(project_root, baseline_result, artifact_prefix="week8")
    print("\nBaseline summary:")
    print(f"- markets: {baseline_result.market_count}")
    print(f"- sortino: {pretty_float(baseline_result.sortino)}")
    print(f"- max_drawdown: {pretty_pct(baseline_result.max_drawdown)}")
    for key, value in baseline_artifacts.items():
        print(f"- {key}: {value}")

    constrained_artifacts = run_experiment_grid(
        project_root,
        artifact_prefix="week8",
        config=ExperimentConfig(
            learning_rates=(0.05,),
            penalties_lambda=(1.0,),
            rolling_windows=(24,),
            steps_per_window=1,
            domain_limit=0.12,
            max_weight=0.05,
            concentration_penalty_lambda=80.0,
            entropy_lambda=0.02,
            uniform_mix=0.88,
            max_domain_exposure_threshold=0.05,
            holdout_fraction=0.2,
            walkforward_train_steps=120,
            walkforward_test_steps=24,
            seed=7,
        ),
    )
    print("\nConstrained experiment artifacts:")
    for key, value in constrained_artifacts.items():
        print(f"- {key}: {value}")

    covariance_artifacts = run_covariance_diagnostics(project_root, artifact_prefix="week8")
    print("\nCovariance diagnostics artifacts:")
    for key, value in covariance_artifacts.items():
        print(f"- {key}: {value}")

    figure_artifacts = _make_figures(project_root)
    print("\nFigure artifacts:")
    for key, value in figure_artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

