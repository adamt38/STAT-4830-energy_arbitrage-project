"""End-to-end Week 8 pipeline after Week 4 prototype."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import os
import pathlib
import shutil
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

from src.baseline import pretty_float, pretty_pct, run_equal_weight_baseline, save_baseline_outputs
from src.constrained_optimizer import ExperimentConfig, run_experiment_grid, run_optuna_search
from src.covariance_diagnostics import run_covariance_diagnostics
from src.equity_signal import (
    EquitySignalConfig,
    REGIME_PENALTY_SCALES,
    compute_risk_regime_zscore,
    get_dynamic_concentration_params,
    get_regime,
)
from src.baseline_vs_hedge_figures import figure_title_banner
from src.polymarket_data import BuildConfig, NoMarketsAfterHistoryFilterError, build_dataset
from src.stock_oil_hedge import StockOilHedgeConfig, run_stock_oil_hedge_experiment


def _read_series(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _read_json(path: pathlib.Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sortino_np(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(returns) / (downside_dev + 1e-8))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _config_hash(build_cfg: BuildConfig, experiment_cfg: ExperimentConfig) -> str:
    payload = {
        "build": build_cfg.__dict__,
        "experiment": experiment_cfg.__dict__,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_run_manifest(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    config_hash: str,
    stage_durations_sec: dict[str, float],
    used_min_history_days: float,
    artifact_groups: dict[str, dict[str, str]],
) -> pathlib.Path:
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / f"{artifact_prefix}_run_manifest.json"
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_prefix": artifact_prefix,
        "config_hash": config_hash,
        "min_history_days_used": used_min_history_days,
        "stage_durations_sec": stage_durations_sec,
        "artifacts": artifact_groups,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _make_week9_diagnostics_report(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    min_history_days_used: float,
) -> pathlib.Path:
    """Create a compact week 9 markdown report from latest artifacts."""
    processed = project_root / "data" / "processed"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _read_json(processed / f"{artifact_prefix}_baseline_metrics.json")
    constrained_metrics = _read_json(processed / f"{artifact_prefix}_constrained_best_metrics.json")
    covariance_summary = _read_json(processed / f"{artifact_prefix}_covariance_summary.json")
    attribution_summary = _read_json(
        processed / f"{artifact_prefix}_constrained_best_attribution_summary.json"
    )
    market_contrib = _read_series(
        processed / f"{artifact_prefix}_constrained_best_market_return_contributions.csv"
    )
    domain_contrib = _read_series(
        processed / f"{artifact_prefix}_constrained_best_domain_return_contributions.csv"
    )
    corr_pairs = _read_series(
        processed / f"{artifact_prefix}_constrained_best_top_market_correlation_pairs.csv"
    )
    baseline_ts = _read_series(processed / f"{artifact_prefix}_baseline_timeseries.csv")

    split_info_obj = constrained_metrics.get("data_split", {})
    split_info = split_info_obj if isinstance(split_info_obj, dict) else {}
    holdout_steps_total = _to_int(split_info.get("holdout_steps_total", 0), default=0)
    baseline_holdout_rows = baseline_ts[-holdout_steps_total:] if holdout_steps_total > 0 else []
    baseline_holdout_rets = np.array(
        [float(row["portfolio_return"]) for row in baseline_holdout_rows], dtype=float
    )
    if baseline_holdout_rets.size > 0:
        baseline_holdout_cum = np.cumprod(1.0 + baseline_holdout_rets)
        baseline_holdout_peak = np.maximum.accumulate(baseline_holdout_cum)
        baseline_holdout_dd = baseline_holdout_cum / np.clip(baseline_holdout_peak, 1e-8, None) - 1.0
        baseline_holdout_max_dd = float(np.min(baseline_holdout_dd))
    else:
        baseline_holdout_max_dd = 0.0

    best_params_obj = constrained_metrics.get("best_params", {})
    best_params = best_params_obj if isinstance(best_params_obj, dict) else {}
    top_market_obj = attribution_summary.get("top_market_by_abs_contribution", {})
    top_domain_obj = attribution_summary.get("top_domain_by_abs_contribution", {})
    top_market_summary = top_market_obj if isinstance(top_market_obj, dict) else {}
    top_domain_summary = top_domain_obj if isinstance(top_domain_obj, dict) else {}
    top_markets = market_contrib[:10]
    top_domains = domain_contrib[:10]
    top_corr_pairs = corr_pairs[:5]

    token_lookup: dict[str, dict[str, str]] = {}
    for row in market_contrib:
        token_lookup[row.get("token_id", "")] = {
            "question": row.get("question", ""),
            "slug": row.get("market_slug", ""),
            "domain": row.get("domain", "other"),
        }

    def _market_label(token_id: str) -> str:
        info = token_lookup.get(token_id, {})
        question = info.get("question", "")
        if question:
            return question
        slug = info.get("slug", "")
        return slug if slug else token_id[:16] + "..."

    top_market_question = top_market_summary.get("question", "") or top_market_summary.get("market_slug", "")
    top_market_domain = top_market_summary.get("domain", "other")
    top_market_contrib_val = _to_float(top_market_summary.get("contribution_share_of_total_return", 0.0))

    constrained_sortino = _to_float(best_params.get("holdout_sortino_ratio", 0.0))
    baseline_sortino_holdout = _sortino_np(baseline_holdout_rets)
    sortino_delta = constrained_sortino - baseline_sortino_holdout
    constrained_dd = _to_float(best_params.get("holdout_max_drawdown", 0.0))
    dd_delta = constrained_dd - baseline_holdout_max_dd

    lines = [
        "# Week 9 Diagnostics Report",
        "",
        "## Run Context",
        f"- artifact prefix: `{artifact_prefix}`",
        f"- min history days used after backoff: `{min_history_days_used}`",
        f"- market count: `{_to_int(baseline_metrics.get('market_count', 0), default=0)}`",
        f"- tuning steps: `{_to_int(split_info.get('tuning_steps', 0), default=0)}`",
        f"- holdout steps: `{holdout_steps_total}`",
        f"- objective: `{_to_float(best_params.get('variance_penalty', 0.0)):.1f}`-var / `{_to_float(best_params.get('downside_penalty', 0.0)):.1f}`-downside mean-downside surrogate"
        if "holdout_sortino_ratio" in best_params
        else "",
        "",
        "## Holdout Performance Comparison",
        "",
        "| Metric | Baseline | Constrained | Delta |",
        "|--------|----------|-------------|-------|",
        f"| Sortino ratio | {baseline_sortino_holdout:.4f} | {constrained_sortino:.4f} | {sortino_delta:+.4f} |",
        f"| Max drawdown | {baseline_holdout_max_dd:.4%} | {constrained_dd:.4%} | {dd_delta:+.4%} |",
        f"| Mean return | {float(np.mean(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0:.8f} | {_to_float(best_params.get('holdout_mean_return', 0.0)):.8f} | {_to_float(best_params.get('holdout_mean_return', 0.0)) - (float(np.mean(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0):+.8f} |",
        f"| Volatility | {float(np.std(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0:.8f} | {_to_float(best_params.get('holdout_volatility', 0.0)):.8f} | {_to_float(best_params.get('holdout_volatility', 0.0)) - (float(np.std(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0):+.8f} |",
        "",
        "## Full-Series Baseline Reference",
        f"- baseline sortino (full): `{_to_float(baseline_metrics.get('sortino_ratio', 0.0)):.4f}`",
        f"- baseline max drawdown (full): `{_to_float(baseline_metrics.get('max_drawdown', 0.0)):.4%}`",
        "",
        "## Attribution — What Drove Returns",
        "",
        f"**Biggest single market:** {top_market_question} (`{top_market_domain}`) — {top_market_contrib_val:.1%} of total return",
        f"**Biggest domain:** `{top_domain_summary.get('domain', 'other')}` — {_to_float(top_domain_summary.get('contribution_share_of_total_return', 0.0)):.1%} of total return",
        "",
        "### Top 10 Market Contributors",
        "",
        "| # | Market | Domain | Contribution | Share | Weight |",
        "|---|--------|--------|-------------|-------|--------|",
    ]
    for rank, row in enumerate(top_markets, 1):
        question = row.get("question", "") or row.get("market_slug", "")
        lines.append(
            "| {} | {} | `{}` | {:.6f} | {:.1%} | {:.4f} |".format(
                rank,
                question,
                row.get("domain", "other"),
                float(row.get("total_contribution", 0.0)),
                float(row.get("contribution_share_of_total_return", 0.0)),
                float(row.get("mean_weight", 0.0)),
            )
        )

    lines.extend([
        "",
        "### Top 10 Domain Contributors",
        "",
        "| # | Domain | Contribution | Share |",
        "|---|--------|-------------|-------|",
    ])
    for rank, row in enumerate(top_domains, 1):
        lines.append(
            "| {} | `{}` | {:.6f} | {:.1%} |".format(
                rank,
                row.get("domain", "other"),
                float(row.get("total_contribution", 0.0)),
                float(row.get("contribution_share_of_total_return", 0.0)),
            )
        )

    lines.extend([
        "",
        "### Top 5 Correlated Contributor Pairs",
        "",
        "| Market A | Market B | Correlation |",
        "|----------|----------|-------------|",
    ])
    for row in top_corr_pairs:
        label_a = _market_label(row.get("token_a", ""))
        label_b = _market_label(row.get("token_b", ""))
        lines.append(
            "| {} | {} | {:.4f} |".format(
                label_a,
                label_b,
                float(row.get("corr", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Correlation and Risk Structure",
            f"- category count: `{_to_int(covariance_summary.get('category_count', 0), default=0)}`",
            f"- avg abs category correlation: `{_to_float(covariance_summary.get('avg_abs_correlation', 0.0)):.4f}`",
            f"- max abs category correlation: `{_to_float(covariance_summary.get('max_abs_correlation', 0.0)):.4f}`",
            f"- top eigenvalue share: `{_to_float(covariance_summary.get('top_eigenvalue_share', 0.0)):.4f}`",
            f"- variance ratio constrained vs baseline: `{_to_float(covariance_summary.get('variance_ratio_constrained_vs_baseline', 0.0)):.4f}`",
            "",
            "## Interpretation Checklist",
            f"- [{'x' if sortino_delta > 0 else ' '}] Constrained holdout Sortino beats baseline ({sortino_delta:+.4f})",
            f"- [{'x' if dd_delta > 0 else ' '}] Constrained holdout drawdown better than baseline ({dd_delta:+.4%})",
            f"- [{'x' if _to_float(covariance_summary.get('max_abs_correlation', 0.0)) < 0.3 else ' '}] Top contributor pairs not excessively correlated (max abs corr: {_to_float(covariance_summary.get('max_abs_correlation', 0.0)):.4f})",
            f"- [{'x' if _to_float(top_domain_summary.get('contribution_share_of_total_return', 0.0)) < 0.5 else ' '}] No single domain dominates returns (top domain share: {_to_float(top_domain_summary.get('contribution_share_of_total_return', 0.0)):.1%})",
        ]
    )

    report_path = docs_dir / f"{artifact_prefix}_diagnostics_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _build_dataset_with_history_backoff(
    project_root: pathlib.Path,
    base_config: BuildConfig,
    history_day_candidates: tuple[float, ...] = (60.0, 45.0, 36.0, 30.0, 24.0),
) -> tuple[dict[str, pathlib.Path], float]:
    """Try stricter-to-looser history windows until markets survive."""
    last_error: RuntimeError | None = None
    for min_days in history_day_candidates:
        cfg = BuildConfig(**{**base_config.__dict__, "min_history_days": float(min_days)})
        try:
            artifacts = build_dataset(project_root=project_root, config=cfg)
            return artifacts, float(min_days)
        except NoMarketsAfterHistoryFilterError as exc:
            last_error = exc
            print(
                f"No markets at min_history_days={min_days}; "
                "retrying with a less strict history filter..."
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Dataset build failed without a captured error.")


def _clone_processed_artifacts(processed: pathlib.Path, src_prefix: str, dst_prefix: str) -> int:
    """Copy ``{src_prefix}_*`` files in processed/ to ``{dst_prefix}_*`` (same suffix)."""
    n = 0
    for path in sorted(processed.glob(f"{src_prefix}_*")):
        if not path.is_file():
            continue
        suffix = path.name[len(f"{src_prefix}_") :]
        dest = processed / f"{dst_prefix}_{suffix}"
        shutil.copy2(path, dest)
        n += 1
    return n


def _make_figures(project_root: pathlib.Path, *, artifact_prefix: str) -> dict[str, pathlib.Path]:
    processed = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    title_banner = figure_title_banner(artifact_prefix)

    baseline_series = _read_series(processed / f"{artifact_prefix}_baseline_timeseries.csv")
    constrained_series = _read_series(processed / f"{artifact_prefix}_constrained_best_timeseries.csv")
    if constrained_series:
        # Align baseline and equity-hedge portfolio plots to the same holdout horizon.
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
    ax_equity.plot(constrained_cum, label="Equity hedge portfolio")
    ax_equity.set_title(f"{title_banner}Portfolio cumulative return vs baseline")
    ax_equity.set_xlabel("Step")
    ax_equity.set_ylabel("Growth of $1")
    ax_equity.legend()
    equity_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_equity_curve_comparison.png"
    fig_equity.savefig(equity_path, dpi=120)
    plt.close(fig_equity)

    fig_dd, ax_dd = plt.subplots()
    ax_dd.plot(baseline_dd, label="Equal-weight baseline")
    ax_dd.plot(constrained_dd, label="Equity hedge portfolio")
    ax_dd.set_title(f"{title_banner}Portfolio drawdown vs baseline")
    ax_dd.set_xlabel("Step")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.legend()
    dd_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_drawdown_comparison.png"
    fig_dd.savefig(dd_path, dpi=120)
    plt.close(fig_dd)

    with (processed / f"{artifact_prefix}_baseline_metrics.json").open("r", encoding="utf-8") as handle:
        baseline_metrics = json.load(handle)
    with (processed / f"{artifact_prefix}_constrained_best_metrics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        constrained_metrics = json.load(handle)

    baseline_exp = baseline_metrics.get("exposure_by_domain", {})
    constrained_exp = constrained_metrics.get("domain_exposure", {})
    all_domains = sorted(set(baseline_exp.keys()) | set(constrained_exp.keys()))
    exposure_rows = [
        {
            "category": domain,
            "baseline_exposure": float(baseline_exp.get(domain, 0.0)),
            "equity_hedge_portfolio_exposure": float(constrained_exp.get(domain, 0.0)),
            "delta_exposure": float(constrained_exp.get(domain, 0.0)) - float(baseline_exp.get(domain, 0.0)),
        }
        for domain in all_domains
    ]
    exposure_table_path = processed / f"{artifact_prefix}_category_exposure_table.csv"
    with exposure_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "baseline_exposure",
                "equity_hedge_portfolio_exposure",
                "delta_exposure",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(exposure_rows, key=lambda row: row["equity_hedge_portfolio_exposure"], reverse=True)
        )

    # Use top categories so the chart remains readable.
    top_n = 25
    ranked_domains = [
        row["category"]
        for row in sorted(
            exposure_rows,
            key=lambda row: max(row["baseline_exposure"], row["equity_hedge_portfolio_exposure"]),
            reverse=True,
        )[:top_n]
    ]
    baseline_vals = [float(baseline_exp.get(domain, 0.0)) for domain in ranked_domains]
    constrained_vals = [float(constrained_exp.get(domain, 0.0)) for domain in ranked_domains]

    x = range(len(ranked_domains))
    width = 0.4
    fig_exp, ax_exp = plt.subplots(figsize=(13, 5))
    ax_exp.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline (1/K per category)")
    ax_exp.bar([i + width / 2 for i in x], constrained_vals, width=width, label="Equity hedge portfolio")
    ax_exp.set_title(
        f"{title_banner}Category exposure: baseline vs optimized equity hedge portfolio"
    )
    ax_exp.set_ylabel("Portfolio Weight")
    ax_exp.set_xticks(list(x))
    ax_exp.set_xticklabels(ranked_domains, rotation=35, ha="right")
    ax_exp.legend()
    exposure_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_category_exposure_comparison.png"
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
    ax_roll.plot(constrained_roll, label="Equity hedge portfolio", linewidth=2.0)
    ax_roll.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax_roll.set_title(f"{title_banner}Rolling mean step return (holdout)")
    ax_roll.set_xlabel("Step")
    ax_roll.set_ylabel("Mean Return")
    ax_roll.legend()
    rolling_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_rolling_mean_return_comparison.png"
    fig_roll.tight_layout()
    fig_roll.savefig(rolling_path, dpi=120)
    plt.close(fig_roll)

    # Return distribution comparison for risk shape.
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(baseline_ret, bins=60, alpha=0.5, label="Baseline")
    ax_hist.hist(constrained_ret, bins=60, alpha=0.5, label="Equity hedge portfolio")
    ax_hist.axvline(float(np.mean(baseline_ret)), linestyle="--", linewidth=1.0, color="tab:blue")
    ax_hist.axvline(float(np.mean(constrained_ret)), linestyle="--", linewidth=1.0, color="tab:orange")
    ax_hist.set_title(f"{title_banner}Step return distribution (holdout)")
    ax_hist.set_xlabel("Step Return")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    hist_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_return_distribution_comparison.png"
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
    ax_delta.set_title(f"{title_banner}Top category exposure deltas (hedge portfolio − baseline)")
    ax_delta.set_ylabel("Delta Weight")
    exposure_delta_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_top_exposure_deltas.png"
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
    ax_scatter.scatter([constrained_vol], [constrained_mean], s=130, label="Equity hedge portfolio (holdout)")
    ax_scatter.set_title(f"{title_banner}Risk–return snapshot (holdout)")
    ax_scatter.set_xlabel("Volatility")
    ax_scatter.set_ylabel("Mean Step Return")
    ax_scatter.legend()
    risk_return_path = figures_dir / f"{artifact_prefix}_equity_hedge_portfolio_risk_return_snapshot.png"
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
    run_started = time.perf_counter()

    parser = argparse.ArgumentParser(description="Polymarket constrained optimizer pipeline.")
    parser.add_argument(
        "--artifact-prefix",
        default="week11",
        help="Prefix for processed artifacts and figures (default: week11 for Week 11 run).",
    )
    parser.add_argument(
        "--clone-from-prefix",
        default=None,
        help="Copy data/processed/{src}_* to {artifact-prefix}_* then continue (reuse prior build).",
    )
    parser.add_argument(
        "--skip-data-build",
        action="store_true",
        help="Skip API/data build; requires existing {artifact-prefix}_markets_filtered.csv (or clone first).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 5 Optuna trials (sanity / fast iteration).",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        metavar="N",
        help="Optuna trial count (overrides --quick when set). Default: 5 with --quick else 100.",
    )
    parser.add_argument(
        "--skip-stock-oil-hedge",
        action="store_true",
        help="Skip PM+stock/oil hedge timeseries (use when offline or Stooq/yfinance unavailable).",
    )
    parser.add_argument(
        "--constant-hedge-blend",
        action="store_true",
        help="Use fixed hedge_allocation every bar instead of VIX regime capital split (Architecture 1).",
    )
    args = parser.parse_args()
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact_prefix = args.artifact_prefix
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.clone_from_prefix:
        n_copied = _clone_processed_artifacts(processed_dir, args.clone_from_prefix, artifact_prefix)
        print(f"Cloned {n_copied} processed files: {args.clone_from_prefix}_* → {artifact_prefix}_*")

    base_build_config = BuildConfig(
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
            "hide-from-new",
            "parent-for-derivative",
            "earn-4",
            "pre-market",
            "rewards-20-4pt5-50",
        ),
        artifact_prefix=artifact_prefix,
        history_interval="max",
        history_fidelity=10,
        use_cached_events_if_available=True,
        history_priority_enabled=True,
        history_priority_oversample_factor=5,
    )
    # ── Toggle: set QUICK_SANITY_CHECK = False for the full Optuna run ──
    QUICK_SANITY_CHECK = False
    if args.optuna_trials is not None:
        OPTUNA_N_TRIALS = max(1, int(args.optuna_trials))
    elif QUICK_SANITY_CHECK or args.quick:
        OPTUNA_N_TRIALS = 5
    else:
        OPTUNA_N_TRIALS = 100

    # With 10-min data we have ~6× more time steps; scale walk-forward so fold count stays similar
    if base_build_config.history_fidelity <= 10:
        walkforward_train_steps = 1440  # ~10 days in 10-min steps
        walkforward_test_steps = 288    # ~2 days in 10-min steps
    else:
        walkforward_train_steps = 240
        walkforward_test_steps = 48

    regime, vix_value = get_regime(EquitySignalConfig())
    regime_scales = REGIME_PENALTY_SCALES[regime]

    def _stage_banner(name: str) -> None:
        elapsed_total = time.perf_counter() - run_started
        print(f"\n{'#'*60}", flush=True)
        print(f"  PIPELINE STAGE: {name}", flush=True)
        print(f"  (total elapsed: {elapsed_total / 60:.1f}m)", flush=True)
        print(f"{'#'*60}\n", flush=True)

    used_min_history_days = 24.0
    data_artifacts: dict[str, pathlib.Path] = {}
    if args.skip_data_build:
        markets_path = processed_dir / f"{artifact_prefix}_markets_filtered.csv"
        if not markets_path.is_file():
            raise SystemExit(
                f"Missing {markets_path}. Run without --skip-data-build or use --clone-from-prefix."
            )
        manifest_try = processed_dir / f"{artifact_prefix}_run_manifest.json"
        if manifest_try.is_file():
            try:
                used_min_history_days = float(
                    _read_json(manifest_try).get("min_history_days_used", used_min_history_days)
                )
            except (TypeError, ValueError, KeyError):
                pass
        print("Skipping data build (--skip-data-build).")
        data_sec = 0.0
    else:
        _stage_banner("Data Build")
        stage_started = time.perf_counter()
        data_artifacts, used_min_history_days = _build_dataset_with_history_backoff(
            project_root=project_root,
            base_config=base_build_config,
            history_day_candidates=(24.0,),
        )
        data_sec = time.perf_counter() - stage_started
        print(f"Data build complete in {data_sec / 60:.1f}m")
        print(f"- min_history_days_used: {used_min_history_days}")
        for key, value in data_artifacts.items():
            print(f"- {key}: {value}")

    experiment_config = ExperimentConfig(
        learning_rates=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        penalties_lambda=(0.25, 0.5, 1.0, 2.0),
        rolling_windows=(24, 48, 96, 144, 288),
        steps_per_window=5,
        objective="excess_mean_downside",
        variance_penalty=1.0,
        downside_penalty=2.0,
        variance_penalties=(0.5, 1.0, 2.0),
        downside_penalties=(1.0, 2.0, 3.0),
        enable_equity_signal=True,
        equity_signal_lambda=0.0,
        equity_signal_lambdas=(),
        equity_signal_mapping_csv=None,
        optimizer_type="adam",
        evaluation_modes=("online",),
        primary_evaluation_mode="online",
        enable_two_stage_search=False,
        stage2_top_k=8,
        max_parallel_workers=1,
        early_prune_enabled=True,
        early_prune_exposure_factor=1.5,
        domain_limits=(0.08, 0.12, 0.18, 0.25),
        max_weights=(0.04, 0.06, 0.10, 0.15),
        concentration_penalty_lambdas=(2.0, 5.0, 10.0, 20.0, 50.0),
        covariance_penalty_lambdas=(0.5, 1.0, 5.0, 10.0),
        covariance_shrinkages=(0.02, 0.05, 0.10),
        entropy_lambdas=(0.0, 0.01, 0.02),
        uniform_mixes=(0.0, 0.05, 0.1, 0.2),
        max_domain_exposure_threshold=0.12,
        holdout_fraction=0.2,
        walkforward_train_steps=walkforward_train_steps,
        walkforward_test_steps=walkforward_test_steps,
        seed=7,
    )
    experiment_config = ExperimentConfig(
        **{
            **experiment_config.__dict__,
            "variance_penalty": float(regime_scales["variance_penalty"]),
            "downside_penalty": float(regime_scales["downside_penalty"]),
            "max_domain_exposure_threshold": float(regime_scales["max_domain_exposure_threshold"]),
        }
    )
    risk_score = compute_risk_regime_zscore()
    dyn_conc = get_dynamic_concentration_params(risk_score)
    merged_conc_lambdas = tuple(
        sorted(set(experiment_config.concentration_penalty_lambdas) | {dyn_conc["concentration_penalty_lambda"]})
    )
    experiment_config = ExperimentConfig(
        **{
            **experiment_config.__dict__,
            "max_domain_exposure_threshold": float(dyn_conc["max_domain_exposure_threshold"]),
            "concentration_penalty_lambdas": merged_conc_lambdas,
        }
    )
    print(
        "Regime-adaptive config:"
        f" regime={regime.value}, vix={f'{vix_value:.2f}' if vix_value is not None else 'n/a'},"
        f" variance_penalty={experiment_config.variance_penalty:.2f},"
        f" downside_penalty={experiment_config.downside_penalty:.2f}"
    )
    print(
        "ETF risk-regime concentration:"
        f" risk_score={risk_score:.4f},"
        f" max_domain_exposure_threshold={experiment_config.max_domain_exposure_threshold:.2f},"
        f" concentration_penalty_lambdas={experiment_config.concentration_penalty_lambdas}"
    )
    config_hash = _config_hash(base_build_config, experiment_config)

    equity_regime_path = processed_dir / f"{artifact_prefix}_pipeline_equity_regime.json"
    equity_regime_path.write_text(
        json.dumps(
            {
                "regime": regime.value,
                "vix_value": vix_value,
                "applied_scales": regime_scales,
                "risk_regime_zscore": risk_score,
                "dynamic_concentration": dyn_conc,
                "merged_concentration_penalty_lambdas": list(merged_conc_lambdas),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"- equity_regime_summary: {equity_regime_path}")

    _stage_banner("Equal-Weight Baseline")
    stage_started = time.perf_counter()
    baseline_result = run_equal_weight_baseline(project_root, artifact_prefix=artifact_prefix)
    baseline_artifacts = save_baseline_outputs(
        project_root, baseline_result, artifact_prefix=artifact_prefix
    )
    baseline_sec = time.perf_counter() - stage_started
    print(f"Baseline complete in {baseline_sec:.1f}s")
    print(f"- markets: {baseline_result.market_count}")
    print(f"- sortino: {pretty_float(baseline_result.sortino)}")
    print(f"- max_drawdown: {pretty_pct(baseline_result.max_drawdown)}")
    for key, value in baseline_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner(f"Optuna Bayesian Search ({OPTUNA_N_TRIALS} trials)")
    stage_started = time.perf_counter()
    constrained_artifacts = run_optuna_search(
        project_root,
        artifact_prefix=artifact_prefix,
        config=experiment_config,
        n_trials=OPTUNA_N_TRIALS,
    )
    constrained_sec = time.perf_counter() - stage_started
    print(f"\nOptuna search complete in {constrained_sec / 60:.1f}m ({constrained_sec / 3600:.1f}h)")
    for key, value in constrained_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Covariance Diagnostics")
    stage_started = time.perf_counter()
    covariance_artifacts = run_covariance_diagnostics(project_root, artifact_prefix=artifact_prefix)
    covariance_sec = time.perf_counter() - stage_started
    print(f"Covariance diagnostics complete in {covariance_sec:.1f}s")
    for key, value in covariance_artifacts.items():
        print(f"- {key}: {value}")

    stock_oil_artifacts: dict[str, pathlib.Path] = {}
    stock_oil_sec = 0.0
    if not args.skip_stock_oil_hedge:
        _stage_banner("Architecture 1: VIX regime PM vs stock/oil hedge split")
        stage_started = time.perf_counter()
        hedge_cfg = StockOilHedgeConfig(
            regime_switching_pm_hedge_split=not args.constant_hedge_blend,
        )
        stock_oil_artifacts = run_stock_oil_hedge_experiment(
            project_root,
            artifact_prefix=artifact_prefix,
            config=hedge_cfg,
        )
        stock_oil_sec = time.perf_counter() - stage_started
        print(f"Stock/oil hedge series complete in {stock_oil_sec:.1f}s")
        for key, value in stock_oil_artifacts.items():
            print(f"- {key}: {value}")

    _stage_banner("Figure Generation")
    stage_started = time.perf_counter()
    figure_artifacts = _make_figures(project_root, artifact_prefix=artifact_prefix)
    figures_sec = time.perf_counter() - stage_started
    print(f"Figures complete in {figures_sec:.1f}s")
    for key, value in figure_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Week 9 Diagnostics Report")
    stage_started = time.perf_counter()
    week9_report_path = _make_week9_diagnostics_report(
        project_root=project_root,
        artifact_prefix=artifact_prefix,
        min_history_days_used=used_min_history_days,
    )
    report_sec = time.perf_counter() - stage_started
    print(f"Report complete in {report_sec:.1f}s")
    print(f"- diagnostics_report: {week9_report_path}")

    manifest_path = _write_run_manifest(
        project_root=project_root,
        artifact_prefix=artifact_prefix,
        config_hash=config_hash,
        stage_durations_sec={
            "data_build": data_sec,
            "baseline": baseline_sec,
            "constrained_grid_and_holdout": constrained_sec,
            "covariance_diagnostics": covariance_sec,
            "stock_oil_hedge": stock_oil_sec,
            "figure_generation": figures_sec,
            "week9_report_generation": report_sec,
            "total": time.perf_counter() - run_started,
        },
        used_min_history_days=used_min_history_days,
        artifact_groups={
            "data": {k: str(v) for k, v in data_artifacts.items()},
            "baseline": {k: str(v) for k, v in baseline_artifacts.items()},
            "constrained": {k: str(v) for k, v in constrained_artifacts.items()},
            "covariance": {k: str(v) for k, v in covariance_artifacts.items()},
            "stock_oil_hedge": {k: str(v) for k, v in stock_oil_artifacts.items()},
            "figures": {k: str(v) for k, v in figure_artifacts.items()},
            "reports": {"week9_diagnostics_report": str(week9_report_path)},
            "equity_regime": {"pipeline_equity_regime": str(equity_regime_path)},
        },
    )
    total_sec = time.perf_counter() - run_started
    print(f"\n{'#'*60}", flush=True)
    print(f"  PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {total_sec / 60:.1f}m ({total_sec / 3600:.1f}h)", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"- run_manifest: {manifest_path}")


if __name__ == "__main__":
    main()

