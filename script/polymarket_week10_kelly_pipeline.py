"""Week 10 Polymarket pipeline: Dynamic-Copula End-to-End Kelly OGD.

This is the binary-resolution Kelly variant of the Week 8 pipeline. It
replaces the mean-downside / Sortino objective and Pearson-covariance
risk model with an expected log-wealth objective evaluated by Monte-Carlo
sampling from a dynamic Gaussian copula whose correlation matrix is
generated at every time step by a small MLP over exogenous macro
features (SPY / QQQ / BTC log returns). Portfolio weights AND MLP
parameters are optimized jointly by Adam-OGD on the projected simplex
with an L1 turnover penalty.

By default the script REUSES cached ``week8_*`` inputs (markets, prices,
exogenous features) and writes all new outputs under
``week10_kelly_*``, so a running Week 8 cloud job is never disturbed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import replace
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
    _build_price_matrix,
    _read_csv,
    pretty_float,
    pretty_pct,
    run_equal_weight_baseline,
    save_baseline_outputs,
)
from src.exogenous_features import build_and_save_exogenous
from src.kelly_copula_optimizer import (
    KellyExperimentConfig,
    run_kelly_optuna_search,
    split_index_for_returns,
)
from src.polymarket_data import BuildConfig, NoMarketsAfterHistoryFilterError, build_dataset


# ---------------------------------------------------------------------------
# I/O helpers (mirrors of the week8 helpers; intentionally duplicated so this
# pipeline has zero hidden coupling to the in-flight week8 module).
# ---------------------------------------------------------------------------


def _read_series(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _read_json(path: pathlib.Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _features_to_strings(value: Any) -> list[str]:
    """Coerce a JSON-loaded value into a list of feature-name strings."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _config_hash(build_cfg: BuildConfig, experiment_cfg: KellyExperimentConfig) -> str:
    payload = {"build": build_cfg.__dict__, "experiment": experiment_cfg.__dict__}
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
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / f"{artifact_prefix}_run_manifest.json"
    payload: dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_prefix": artifact_prefix,
        "config_hash": config_hash,
        "min_history_days_used": used_min_history_days,
        "stage_durations_sec": stage_durations_sec,
        "artifacts": artifact_groups,
    }
    if extra:
        payload.update(extra)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _git_commit_and_push(
    project_root: pathlib.Path,
    *,
    remote: str,
    branch: str,
    message: str,
) -> None:
    """Stage processed outputs, commit, pull --rebase, push HEAD to remote branch."""
    if not (project_root / ".git").is_dir():
        print("Skipping git publish: no .git directory.", flush=True)
        return

    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    paths_to_add: list[str] = []
    for rel in ("data/processed", "figures", "docs"):
        if (project_root / rel).exists():
            paths_to_add.append(rel)
    if not paths_to_add:
        print("Skipping git publish: no output directories present.", flush=True)
        return

    add_r = _run(["add", *paths_to_add])
    if add_r.returncode != 0:
        raise RuntimeError(f"git add failed:\n{add_r.stderr or add_r.stdout}")
    if _run(["diff", "--cached", "--quiet"]).returncode == 0:
        print("Git publish: nothing to commit.", flush=True)
        return
    commit_r = _run(["commit", "-m", message])
    if commit_r.returncode != 0:
        raise RuntimeError(f"git commit failed:\n{commit_r.stderr or commit_r.stdout}")
    pull_r = _run(["pull", "--rebase", remote, branch])
    if pull_r.returncode != 0:
        err = (pull_r.stderr or "") + (pull_r.stdout or "")
        if "couldn't find remote ref" not in err and "could not find remote ref" not in err:
            raise RuntimeError(f"git pull --rebase failed:\n{err}")
    push_r = _run(["push", remote, f"HEAD:{branch}"])
    if push_r.returncode != 0:
        raise RuntimeError(f"git push failed:\n{push_r.stderr or push_r.stdout}")
    head_r = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    head_name = (head_r.stdout or "").strip() or "?"
    print(f"Git publish: pushed {head_name!r} to {remote}/{branch}.", flush=True)


# ---------------------------------------------------------------------------
# Data build / reuse
# ---------------------------------------------------------------------------


def _build_dataset_with_history_backoff(
    project_root: pathlib.Path,
    base_config: BuildConfig,
    history_day_candidates: tuple[float, ...] = (24.0, 18.0, 12.0, 7.0),
) -> tuple[dict[str, pathlib.Path], float]:
    last_error: RuntimeError | None = None
    for min_days in history_day_candidates:
        cfg = BuildConfig(**{**base_config.__dict__, "min_history_days": float(min_days)})
        try:
            artifacts = build_dataset(project_root=project_root, config=cfg)
            return artifacts, float(min_days)
        except NoMarketsAfterHistoryFilterError as exc:
            last_error = exc
            print(f"No markets at min_history_days={min_days}; retrying with looser filter...")
    if last_error is not None:
        raise last_error
    raise RuntimeError("Dataset build failed without a captured error.")


def _load_cached_dataset_artifacts(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
) -> dict[str, pathlib.Path]:
    processed = project_root / "data" / "processed"
    raw = project_root / "data" / "raw"
    artifacts = {
        "events_raw": raw / f"{artifact_prefix}_events_raw.json",
        "markets_filtered": processed / f"{artifact_prefix}_markets_filtered.csv",
        "price_history": processed / f"{artifact_prefix}_price_history.csv",
        "data_quality": processed / f"{artifact_prefix}_data_quality.json",
        "category_liquidity": processed / f"{artifact_prefix}_category_liquidity.csv",
        "considered_domains": processed / f"{artifact_prefix}_considered_domains.json",
    }
    missing = [str(p) for p in artifacts.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Cached dataset for prefix {artifact_prefix!r} is incomplete (missing: "
            + ", ".join(missing)
            + "). Re-run with --rebuild-data to fetch a fresh dataset."
        )
    if not _read_series(artifacts["markets_filtered"]) or not _read_series(artifacts["price_history"]):
        raise RuntimeError(
            f"Cached dataset for prefix {artifact_prefix!r} exists but is empty."
        )
    return artifacts


def _copy_input_artifacts_to_output(
    project_root: pathlib.Path,
    *,
    input_prefix: str,
    output_prefix: str,
) -> dict[str, pathlib.Path]:
    """Hard-link / copy cached input artifacts to the output prefix.

    This lets every downstream tool (covariance/figures/report) that hard-codes
    the artifact prefix find files under the new prefix. We use shallow file
    copies (not symlinks) for cross-platform safety.
    """
    import shutil

    processed = project_root / "data" / "processed"
    raw = project_root / "data" / "raw"
    sources = {
        raw / f"{input_prefix}_events_raw.json": raw / f"{output_prefix}_events_raw.json",
        processed / f"{input_prefix}_markets_filtered.csv": processed / f"{output_prefix}_markets_filtered.csv",
        processed / f"{input_prefix}_price_history.csv": processed / f"{output_prefix}_price_history.csv",
        processed / f"{input_prefix}_data_quality.json": processed / f"{output_prefix}_data_quality.json",
        processed / f"{input_prefix}_category_liquidity.csv": processed / f"{output_prefix}_category_liquidity.csv",
        processed / f"{input_prefix}_considered_domains.json": processed / f"{output_prefix}_considered_domains.json",
    }
    written: dict[str, pathlib.Path] = {}
    for src, dst in sources.items():
        if not src.exists():
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            written[dst.stem.split("_", 1)[-1]] = dst
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written[dst.stem.split("_", 1)[-1]] = dst
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_kelly_figures(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
) -> dict[str, pathlib.Path]:
    processed = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_series = _read_series(processed / f"{artifact_prefix}_baseline_timeseries.csv")
    kelly_series = _read_series(processed / f"{artifact_prefix}_kelly_best_timeseries.csv")
    if kelly_series:
        baseline_series = baseline_series[-len(kelly_series) :]

    baseline_ret = np.array(
        [float(row["portfolio_return"]) for row in baseline_series], dtype=float
    )
    kelly_ret = np.array(
        [float(row["portfolio_return"]) for row in kelly_series], dtype=float
    )
    kelly_logwealth = np.array(
        [float(row["log_wealth_increment"]) for row in kelly_series], dtype=float
    )
    kelly_cum_logwealth = np.array(
        [float(row["cumulative_log_wealth"]) for row in kelly_series], dtype=float
    )
    kelly_dd = np.array([float(row["drawdown"]) for row in kelly_series], dtype=float)
    kelly_turnover = np.array(
        [float(row["turnover_l1"]) for row in kelly_series], dtype=float
    )

    baseline_logwealth = np.log(np.clip(1.0 + baseline_ret, 1e-9, None))
    baseline_cum_logwealth = np.cumsum(baseline_logwealth)
    baseline_cum = np.cumprod(1.0 + baseline_ret) if baseline_ret.size else np.array([])
    if baseline_cum.size:
        baseline_peak = np.maximum.accumulate(baseline_cum)
        baseline_dd = baseline_cum / np.clip(baseline_peak, 1e-8, None) - 1.0
    else:
        baseline_dd = np.array([])

    # Cumulative log-wealth curve.
    fig_lw, ax_lw = plt.subplots()
    ax_lw.plot(baseline_cum_logwealth, label="Equal-weight baseline")
    ax_lw.plot(kelly_cum_logwealth, label="Dynamic-Copula Kelly OGD")
    ax_lw.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax_lw.set_title("Cumulative Log-Wealth (Holdout)")
    ax_lw.set_xlabel("Step")
    ax_lw.set_ylabel(r"$\sum_t \log(1 + r_t)$")
    ax_lw.legend()
    log_wealth_path = figures_dir / f"{artifact_prefix}_iteration_log_wealth_curve.png"
    fig_lw.tight_layout()
    fig_lw.savefig(log_wealth_path, dpi=120)
    plt.close(fig_lw)

    # Turnover per step.
    fig_to, ax_to = plt.subplots()
    ax_to.plot(kelly_turnover, label="L1 turnover", linewidth=1.0)
    if kelly_turnover.size >= 16:
        kernel = np.ones(16, dtype=float) / 16.0
        rolled = np.convolve(kelly_turnover, kernel, mode="valid")
        prefix = [float(rolled[0])] * (16 - 1)
        roll_arr = np.array(prefix + rolled.tolist())
        ax_to.plot(roll_arr, label="16-step rolling mean", linewidth=2.0)
    ax_to.set_title("L1 Turnover per Step (Holdout)")
    ax_to.set_xlabel("Step")
    ax_to.set_ylabel(r"$\|w_t - w_{t-1}\|_1$")
    ax_to.legend()
    turnover_path = figures_dir / f"{artifact_prefix}_iteration_turnover_per_step.png"
    fig_to.tight_layout()
    fig_to.savefig(turnover_path, dpi=120)
    plt.close(fig_to)

    # Drawdown comparison.
    fig_dd, ax_dd = plt.subplots()
    if baseline_dd.size:
        ax_dd.plot(baseline_dd, label="Equal-weight baseline")
    ax_dd.plot(kelly_dd, label="Dynamic-Copula Kelly OGD")
    ax_dd.set_title("Portfolio Drawdown (Holdout)")
    ax_dd.set_xlabel("Step")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.legend()
    drawdown_path = figures_dir / f"{artifact_prefix}_iteration_drawdown_comparison.png"
    fig_dd.tight_layout()
    fig_dd.savefig(drawdown_path, dpi=120)
    plt.close(fig_dd)

    # Side-by-side: cumulative log-wealth (re-emphasized as the key figure).
    fig_kvb, ax_kvb = plt.subplots(figsize=(8, 4.5))
    ax_kvb.plot(baseline_cum_logwealth, label="Equal-weight baseline", linewidth=2.0)
    ax_kvb.plot(kelly_cum_logwealth, label="Dynamic-Copula Kelly OGD", linewidth=2.0)
    ax_kvb.fill_between(
        np.arange(kelly_cum_logwealth.size),
        baseline_cum_logwealth[: kelly_cum_logwealth.size]
        if baseline_cum_logwealth.size >= kelly_cum_logwealth.size
        else np.zeros_like(kelly_cum_logwealth),
        kelly_cum_logwealth,
        alpha=0.15,
    )
    ax_kvb.set_title("Kelly OGD vs Equal-Weight: Cumulative Log-Wealth")
    ax_kvb.set_xlabel("Step")
    ax_kvb.set_ylabel(r"$\sum_t \log(1 + r_t)$")
    ax_kvb.legend()
    kvb_path = figures_dir / f"{artifact_prefix}_iteration_kelly_vs_baseline_log_wealth.png"
    fig_kvb.tight_layout()
    fig_kvb.savefig(kvb_path, dpi=120)
    plt.close(fig_kvb)

    # Copula correlation heatmap (from per-step holdout weights * implied
    # post-hoc correlation summary in the diagnostics JSON; if not available we
    # fall back to a heatmap of empirical Pearson correlations of holdout
    # asset returns implied by `_kelly_best_weights.csv`).
    diag_path = processed / f"{artifact_prefix}_kelly_copula_diagnostics.json"
    weights_path = processed / f"{artifact_prefix}_kelly_best_weights.csv"
    heatmap_data: np.ndarray | None = None
    heatmap_title = "Average Copula Correlation (Holdout)"
    if diag_path.exists() and weights_path.exists():
        diag = _read_json(diag_path)
        n_assets = _to_int(diag.get("n_assets", 0))
        # The copula diagnostic JSON only stores summary stats. Recreate a
        # representative heatmap by computing Pearson correlations of the
        # asset-level price returns across the holdout window using the
        # cached price history (this is a reasonable visual proxy and fast
        # to compute).
        markets_path = processed / f"{artifact_prefix}_markets_filtered.csv"
        history_path = processed / f"{artifact_prefix}_price_history.csv"
        if markets_path.exists() and history_path.exists() and n_assets > 0:
            _, price_matrix, _ = _build_price_matrix(
                _read_csv(markets_path), _read_csv(history_path)
            )
            if price_matrix.shape[0] > 1:
                returns = np.diff(np.log(np.clip(price_matrix, 1e-6, None)), axis=0)
                # Use the holdout slice only.
                holdout_steps = _to_int(diag.get("n_holdout_steps", 0))
                if holdout_steps > 1 and holdout_steps <= returns.shape[0]:
                    returns = returns[-holdout_steps:]
                returns = np.nan_to_num(returns, nan=0.0)
                if returns.shape[0] > 2 and returns.shape[1] > 1:
                    heatmap_data = np.corrcoef(returns, rowvar=False)
                    heatmap_title = (
                        "Empirical Pearson Correlation of Asset Returns "
                        "(visual proxy; copula is dynamic per-step)"
                    )

    fig_hm, ax_hm = plt.subplots(figsize=(7, 6))
    if heatmap_data is None:
        ax_hm.text(
            0.5,
            0.5,
            "Correlation heatmap unavailable\n(missing diagnostics or weights)",
            ha="center",
            va="center",
            transform=ax_hm.transAxes,
        )
    else:
        im = ax_hm.imshow(heatmap_data, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    ax_hm.set_title(heatmap_title)
    ax_hm.set_xlabel("Asset index")
    ax_hm.set_ylabel("Asset index")
    heatmap_path = figures_dir / f"{artifact_prefix}_iteration_copula_corr_heatmap.png"
    fig_hm.tight_layout()
    fig_hm.savefig(heatmap_path, dpi=120)
    plt.close(fig_hm)

    return {
        "log_wealth_curve": log_wealth_path,
        "turnover_per_step": turnover_path,
        "drawdown_comparison": drawdown_path,
        "kelly_vs_baseline_log_wealth": kvb_path,
        "copula_corr_heatmap": heatmap_path,
    }


# ---------------------------------------------------------------------------
# Week 10 diagnostics report
# ---------------------------------------------------------------------------


def _make_week10_diagnostics_report(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    min_history_days_used: float,
    history_fidelity_min: int,
) -> pathlib.Path:
    """Markdown report focused on log-wealth growth, turnover, and copula stats.

    Replaces every Sortino / variance metric from the week8 ``_make_week9_diagnostics_report``
    with Kelly-native quantities (log-wealth, CAGR-equivalent, turnover L1,
    copula stability).
    """
    processed = project_root / "data" / "processed"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _read_json(processed / f"{artifact_prefix}_baseline_metrics.json")
    kelly_metrics = _read_json(processed / f"{artifact_prefix}_kelly_best_metrics.json")
    diag = _read_json(processed / f"{artifact_prefix}_kelly_copula_diagnostics.json")
    baseline_ts = _read_series(processed / f"{artifact_prefix}_baseline_timeseries.csv")
    kelly_ts = _read_series(processed / f"{artifact_prefix}_kelly_best_timeseries.csv")

    bp_obj = kelly_metrics.get("best_params", {})
    bp = bp_obj if isinstance(bp_obj, dict) else {}
    km_obj = kelly_metrics.get("kelly_metrics", {})
    km = km_obj if isinstance(km_obj, dict) else {}
    split_obj = kelly_metrics.get("data_split", {})
    split_info = split_obj if isinstance(split_obj, dict) else {}
    optuna_obj = kelly_metrics.get("optuna_summary", {})
    optuna_info = optuna_obj if isinstance(optuna_obj, dict) else {}

    holdout_steps = _to_int(split_info.get("holdout_steps_total", len(kelly_ts)), default=len(kelly_ts))
    baseline_holdout_rows = baseline_ts[-holdout_steps:] if holdout_steps > 0 else []
    baseline_holdout_rets = np.array(
        [float(row["portfolio_return"]) for row in baseline_holdout_rows], dtype=float
    )
    kelly_rets = np.array([float(row["portfolio_return"]) for row in kelly_ts], dtype=float)
    kelly_logwealth = np.array(
        [float(row["log_wealth_increment"]) for row in kelly_ts], dtype=float
    )
    kelly_turnover = np.array(
        [float(row["turnover_l1"]) for row in kelly_ts], dtype=float
    )

    baseline_logwealth = (
        np.log(np.clip(1.0 + baseline_holdout_rets, 1e-9, None))
        if baseline_holdout_rets.size
        else np.array([], dtype=float)
    )
    baseline_total_logwealth = float(np.sum(baseline_logwealth))
    baseline_per_step_logwealth = (
        float(np.mean(baseline_logwealth)) if baseline_logwealth.size else 0.0
    )

    if baseline_holdout_rets.size:
        baseline_cum = np.cumprod(1.0 + baseline_holdout_rets)
        baseline_peak = np.maximum.accumulate(baseline_cum)
        baseline_dd = baseline_cum / np.clip(baseline_peak, 1e-8, None) - 1.0
        baseline_max_dd = float(np.min(baseline_dd))
    else:
        baseline_max_dd = 0.0

    # Annualized growth rate assuming step length implied by history_fidelity_min.
    # 525_600 minutes per year.
    minutes_per_step = max(int(history_fidelity_min), 1)
    steps_per_year = 525_600.0 / float(minutes_per_step)
    kelly_per_step = _to_float(km.get("holdout_log_wealth_per_step", 0.0))
    kelly_total = _to_float(km.get("holdout_log_wealth_total", 0.0))
    kelly_growth_annualized_log = kelly_per_step * steps_per_year
    kelly_cagr = float(np.expm1(kelly_growth_annualized_log))
    baseline_growth_annualized_log = baseline_per_step_logwealth * steps_per_year
    baseline_cagr = float(np.expm1(baseline_growth_annualized_log))

    log_wealth_delta = kelly_total - baseline_total_logwealth
    dd_delta = _to_float(km.get("holdout_max_drawdown", 0.0)) - baseline_max_dd

    baseline_total_turnover = (
        float(np.sum(np.abs(np.diff(np.ones_like(baseline_holdout_rets)))))
        if baseline_holdout_rets.size
        else 0.0
    )
    # Equal-weight baseline has zero turnover by construction (renormalization
    # of the same fixed weights when assets enter / leave is small but
    # non-zero in practice; we report 0 here because the baseline writer
    # does not record per-step weights).

    lines: list[str] = [
        "# Week 10 Diagnostics Report — Dynamic-Copula End-to-End Kelly OGD",
        "",
        "## Run Context",
        f"- artifact prefix: `{artifact_prefix}`",
        f"- min history days used after backoff: `{min_history_days_used}`",
        f"- market count: `{_to_int(baseline_metrics.get('market_count', 0))}`",
        f"- tuning steps: `{_to_int(split_info.get('tuning_steps', 0))}`",
        f"- holdout steps: `{_to_int(split_info.get('holdout_steps_total', 0))}`",
        f"- step duration assumed: `{minutes_per_step} min` (steps/year = `{steps_per_year:.0f}`)",
        f"- objective: **expected log-wealth via Monte-Carlo dynamic Gaussian copula (binary Kelly)**",
        f"- Optuna trials: `{_to_int(optuna_info.get('completed_trials', 0))}` completed",
        f"- selected MLP hidden dim: `{_to_int(bp.get('mlp_hidden_dim', 0))}`",
        f"- MLP parameter count: `{_to_int(km.get('mlp_param_count', 0))}`",
        "",
        "## Selected Hyperparameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Weight learning rate (`lr_w`) | `{_to_float(bp.get('lr_w', 0.0)):.5f}` |",
        f"| MLP learning rate (`lr_theta`) | `{_to_float(bp.get('lr_theta', 0.0)):.5f}` |",
        f"| Rolling window | `{_to_int(bp.get('rolling_window', 0))}` |",
        f"| Monte-Carlo samples per inner step | `{_to_int(bp.get('mc_samples', 0))}` |",
        f"| L1 turnover penalty (`lambda_turn`) | `{_to_float(bp.get('turnover_lambda', 0.0)):.4f}` |",
        f"| Copula shrinkage to identity | `{_to_float(bp.get('copula_shrinkage', 0.0)):.3f}` |",
        f"| Straight-through Bernoulli temperature | `{_to_float(bp.get('copula_temperature', 0.0)):.3f}` |",
        f"| Concentration penalty `lambda_conc` | `{_to_float(bp.get('concentration_penalty_lambda', 0.0)):.3f}` |",
        f"| Per-asset cap (`max_weight`) | `{_to_float(bp.get('max_weight', 0.0)):.3f}` |",
        f"| Transaction fee rate (`fee_rate`) | `{_to_float(bp.get('fee_rate', 0.0)):.5f}` ({_to_float(bp.get('fee_rate', 0.0)) * 10_000:.1f} bps per unit L1 turnover) |",
        f"| Downside penalty (`dd_penalty`) | `{_to_float(bp.get('dd_penalty', 0.0)):.3f}` |",
        "",
        "## Holdout Log-Wealth Growth (Kelly objective, primary metric)",
        "",
        (
            "> Kelly columns are **net of fees** (realized return has "
            "``fee_rate * turnover`` deducted per step before log-wealth is "
            "accumulated). Baseline is gross; its per-step turnover is zero so "
            "baseline net = baseline gross."
            if _to_float(bp.get("fee_rate", 0.0)) > 0.0
            else "> Kelly columns are gross of fees (``fee_rate = 0``)."
        ),
        "",
        "| Metric | Equal-Weight Baseline | Dynamic-Copula Kelly OGD | Delta |",
        "|--------|-----------------------|--------------------------|-------|",
        f"| Total log-wealth ($\\sum_t \\log(1+r_t)$) | {baseline_total_logwealth:+.6f} | {kelly_total:+.6f} | {log_wealth_delta:+.6f} |",
        f"| Per-step log-wealth | {baseline_per_step_logwealth:+.8f} | {kelly_per_step:+.8f} | {kelly_per_step - baseline_per_step_logwealth:+.8f} |",
        f"| Annualized log-growth | {baseline_growth_annualized_log:+.4f} | {kelly_growth_annualized_log:+.4f} | {kelly_growth_annualized_log - baseline_growth_annualized_log:+.4f} |",
        f"| Equivalent CAGR | {baseline_cagr:+.2%} | {kelly_cagr:+.2%} | {kelly_cagr - baseline_cagr:+.2%} |",
        f"| Max drawdown (realized) | {baseline_max_dd:+.4%} | {_to_float(km.get('holdout_max_drawdown', 0.0)):+.4%} | {dd_delta:+.4%} |",
        f"| Mean realized step return | {float(np.mean(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0:+.8f} | {_to_float(km.get('holdout_mean_realized_return', 0.0)):+.8f} | — |",
        f"| Realized return volatility | {float(np.std(baseline_holdout_rets)) if baseline_holdout_rets.size else 0.0:.8f} | {_to_float(km.get('holdout_volatility', 0.0)):.8f} | — |",
        "",
        "## Turnover Metrics (L1 weight changes, market-friction proxy)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total holdout L1 turnover | `{_to_float(km.get('holdout_total_turnover_l1', 0.0)):.4f}` |",
        f"| Average per-step L1 turnover | `{_to_float(km.get('holdout_avg_turnover_l1', 0.0)):.6f}` |",
        f"| Maximum per-step L1 turnover | `{_to_float(km.get('holdout_max_turnover_l1', 0.0)):.6f}` |",
        f"| Equal-weight baseline turnover (reference) | `{baseline_total_turnover:.4f}` |",
        "",
        "## Dynamic Copula Diagnostics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average off-diagonal correlation (mean) | `{_to_float(diag.get('average_correlation_off_diag_mean', 0.0)):+.4f}` |",
        f"| Average off-diagonal correlation (mean abs) | `{_to_float(diag.get('average_correlation_off_diag_abs_mean', 0.0)):.4f}` |",
        f"| Maximum off-diagonal correlation (abs) | `{_to_float(diag.get('average_correlation_off_diag_max_abs', 0.0)):.4f}` |",
        f"| Average correlation matrix condition number | `{_to_float(diag.get('average_correlation_condition_number', 0.0)):.2f}` |",
        f"| Holdout steps requiring PD shrinkage fallback | `{_to_int(diag.get('pd_fallback_steps', 0))}` ({_to_float(diag.get('pd_fallback_step_fraction', 0.0)) * 100:.1f}% of steps) |",
        f"| Exogenous features used | `{', '.join(_features_to_strings(diag.get('exogenous_features_used')))}` |",
        f"| Copula disabled (R = I baseline) | `{bool(diag.get('disable_copula', False))}` |",
        "",
        "## Interpretation Checklist",
        f"- [{'x' if log_wealth_delta > 0 else ' '}] Kelly OGD beats equal-weight on cumulative log-wealth (delta `{log_wealth_delta:+.6f}`)",
        f"- [{'x' if _to_float(km.get('holdout_max_drawdown', 0.0)) > baseline_max_dd else ' '}] Kelly OGD has smaller (less negative) max drawdown than baseline",
        f"- [{'x' if _to_float(km.get('holdout_avg_turnover_l1', 0.0)) <= 0.5 else ' '}] Average per-step L1 turnover stays below 0.5 (`{_to_float(km.get('holdout_avg_turnover_l1', 0.0)):.4f}`)",
        f"- [{'x' if _to_float(diag.get('pd_fallback_step_fraction', 0.0)) <= 0.05 else ' '}] PD shrinkage fallback fires on at most 5% of steps (`{_to_float(diag.get('pd_fallback_step_fraction', 0.0)) * 100:.1f}%`)",
    ]

    report_path = docs_dir / "week10_kelly_diagnostics_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Week 10 Polymarket pipeline: data reuse / build → equal-weight baseline → "
            "exogenous features → Dynamic-Copula End-to-End Kelly OGD (Optuna QMC) → "
            "figures → diagnostics report."
        )
    )
    parser.add_argument(
        "--artifact-prefix",
        default="week10_kelly",
        help="Stem for every artifact this run writes (default: week10_kelly).",
    )
    parser.add_argument(
        "--input-artifact-prefix",
        default="week8",
        help="Cached dataset prefix to consume (default: week8). The script copies the "
        "cached markets/prices CSVs to {artifact_prefix}_* and reuses them as inputs.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force a fresh build_dataset() under --artifact-prefix instead of reusing "
        "the cached --input-artifact-prefix dataset.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Override Optuna trial count (default 100, or 5 when QUICK_SANITY_CHECK).",
    )
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna trials (study.optimize n_jobs). Use 1 for sequential.",
    )
    parser.add_argument(
        "--mc-samples-override",
        type=int,
        default=None,
        help="Pin Monte-Carlo sample count S for every Optuna trial (skips MC sweep).",
    )
    parser.add_argument(
        "--lambda-turnover-override",
        type=float,
        default=None,
        help="Pin the L1 turnover penalty for every Optuna trial.",
    )
    parser.add_argument(
        "--fee-rate-values",
        type=str,
        default=None,
        metavar="F1,F2,...",
        help="Comma-separated fee_rate sweep values (decimals; 0.0010 = 10 bps). "
        "Overrides the default ``(0.0,)`` and enables a categorical Optuna "
        "dimension. Each value is applied in BOTH the training loss "
        "(+ fee_rate * ||Delta w||_1) and the reported realized return "
        "(r_net = r_gross - fee_rate * turnover). Round 7 K10E sweep: "
        "--fee-rate-values 0,0.0010,0.0050,0.0200.",
    )
    parser.add_argument(
        "--fee-rate-override",
        type=float,
        default=None,
        help="Pin a single fee_rate for every Optuna trial (decimal).",
    )
    parser.add_argument(
        "--dd-penalty-values",
        type=str,
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated dd_penalty sweep values. Adds "
        "``dd_penalty * mean(relu(-rho_mc)**2)`` to the Kelly training "
        "objective (downside-semivariance on MC-sampled per-step returns). "
        "Round 7 K10F sweep: --dd-penalty-values 0,0.5,2,5,10.",
    )
    parser.add_argument(
        "--dd-penalty-override",
        type=float,
        default=None,
        help="Pin a single dd_penalty for every Optuna trial.",
    )
    parser.add_argument(
        "--no-exogenous",
        action="store_true",
        help="Disable the dynamic copula (force R_t = I). Useful sanity baseline that "
        "isolates the Kelly + L1-turnover gain from the copula contribution.",
    )
    parser.add_argument(
        "--reduced-search",
        action="store_true",
        help="Narrow Optuna ranges for faster denser Sobol coverage on a small budget.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip the Optuna search and resume from disk-cached *_kelly_best_*.json/csv.",
    )
    parser.add_argument(
        "--git-commit-and-push",
        action="store_true",
        help="After a successful run: git add data/processed figures docs, commit, "
        "git pull --rebase <remote> <branch>, git push <remote> HEAD:<branch>.",
    )
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-push-branch", default="cloud-runs")
    parser.add_argument("--git-commit-message", default=None, metavar="MSG")
    args = parser.parse_args()

    project_root = REPO_ROOT
    run_started = time.perf_counter()

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
        artifact_prefix=args.artifact_prefix,
        history_interval="max",
        history_fidelity=60,
        use_cached_events_if_available=True,
        history_priority_enabled=True,
        history_priority_oversample_factor=5,
    )

    QUICK_SANITY_CHECK = False
    OPTUNA_N_TRIALS = 5 if QUICK_SANITY_CHECK else 100

    if base_build_config.history_fidelity <= 10:
        walkforward_train_steps = 1440
        walkforward_test_steps = 288
    else:
        walkforward_train_steps = 240
        walkforward_test_steps = 48

    def _parse_float_csv(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
        if raw is None:
            return default
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        try:
            return tuple(float(x) for x in parts) or default
        except ValueError as exc:
            raise SystemExit(f"Invalid float in {raw!r}: {exc}") from None

    fee_rates_cfg = _parse_float_csv(args.fee_rate_values, (0.0,))
    dd_penalty_cfg = _parse_float_csv(args.dd_penalty_values, (0.0,))

    experiment_config = KellyExperimentConfig(
        learning_rates_w=(0.005, 0.01, 0.02, 0.05),
        learning_rates_theta=(1e-4, 5e-4, 1e-3, 5e-3),
        rolling_windows=(24, 48, 96),
        mc_samples=(256, 512, 1024),
        turnover_lambdas=(0.0, 0.001, 0.01, 0.05, 0.1),
        copula_shrinkages=(0.05, 0.1, 0.25),
        copula_temperatures=(0.05, 0.1, 0.25),
        mlp_hidden_dims=(8, 16, 32),
        concentration_penalty_lambdas=(2.0, 10.0, 50.0),
        max_weights=(0.04, 0.06, 0.10, 0.15),
        fee_rates=fee_rates_cfg,
        dd_penalty_lambdas=dd_penalty_cfg,
        steps_per_window=3,
        weight_parameterization="projected_simplex",
        optimizer_type="adam",
        holdout_fraction=0.2,
        walkforward_train_steps=walkforward_train_steps,
        walkforward_test_steps=walkforward_test_steps,
        seed=7,
        optuna_n_jobs=args.optuna_n_jobs,
        disable_copula=bool(args.no_exogenous),
    )
    if args.reduced_search:
        experiment_config = replace(
            experiment_config,
            learning_rates_w=(0.01, 0.02, 0.05),
            learning_rates_theta=(5e-4, 1e-3),
            mc_samples=(256, 512),
            turnover_lambdas=(0.0, 0.01, 0.05),
            copula_shrinkages=(0.05, 0.1),
            copula_temperatures=(0.1,),
            mlp_hidden_dims=(8, 16),
            max_weights=(0.06, 0.10),
        )
    config_hash = _config_hash(base_build_config, experiment_config)

    def _stage_banner(name: str) -> None:
        elapsed_total = time.perf_counter() - run_started
        print(f"\n{'#' * 60}", flush=True)
        print(f"  PIPELINE STAGE: {name}", flush=True)
        print(f"  (total elapsed: {elapsed_total / 60:.1f}m)", flush=True)
        print(f"{'#' * 60}\n", flush=True)

    # ---------------- Stage 1: data build / reuse ----------------
    _stage_banner("Data Build / Reuse")
    stage_started = time.perf_counter()
    used_min_history_days = float(base_build_config.min_history_days)
    if args.rebuild_data:
        try:
            data_artifacts, used_min_history_days = _build_dataset_with_history_backoff(
                project_root=project_root,
                base_config=base_build_config,
            )
        except NoMarketsAfterHistoryFilterError as exc:
            print(f"Fresh data build failed ({exc}); attempting cached fallback for "
                  f"{args.artifact_prefix!r}.")
            data_artifacts = _load_cached_dataset_artifacts(
                project_root=project_root, artifact_prefix=args.artifact_prefix
            )
    else:
        try:
            input_artifacts = _load_cached_dataset_artifacts(
                project_root=project_root, artifact_prefix=args.input_artifact_prefix
            )
            print(
                f"Reusing cached input dataset under prefix {args.input_artifact_prefix!r}."
            )
        except RuntimeError as exc:
            print(
                f"Cached input prefix {args.input_artifact_prefix!r} unavailable: {exc}\n"
                f"Falling back to fresh build under {args.artifact_prefix!r}."
            )
            data_artifacts, used_min_history_days = _build_dataset_with_history_backoff(
                project_root=project_root,
                base_config=base_build_config,
            )
            input_artifacts = data_artifacts
        copied = _copy_input_artifacts_to_output(
            project_root,
            input_prefix=args.input_artifact_prefix,
            output_prefix=args.artifact_prefix,
        )
        data_artifacts = {
            **{k: v for k, v in input_artifacts.items()},
            **{f"copied_{k}": v for k, v in copied.items()},
        }
    data_sec = time.perf_counter() - stage_started
    print(f"Data stage complete in {data_sec:.1f}s")
    print(f"- min_history_days_used: {used_min_history_days}")
    for k, v in data_artifacts.items():
        print(f"- {k}: {v}")

    # ---------------- Stage 2: equal-weight baseline ----------------
    _stage_banner("Equal-Weight Baseline")
    stage_started = time.perf_counter()
    baseline_result = run_equal_weight_baseline(
        project_root, artifact_prefix=args.artifact_prefix
    )
    baseline_artifacts = save_baseline_outputs(
        project_root, baseline_result, artifact_prefix=args.artifact_prefix
    )
    baseline_sec = time.perf_counter() - stage_started
    print(f"Baseline complete in {baseline_sec:.1f}s")
    print(f"- markets: {baseline_result.market_count}")
    print(f"- sortino (reference only): {pretty_float(baseline_result.sortino)}")
    print(f"- max_drawdown: {pretty_pct(baseline_result.max_drawdown)}")

    # ---------------- Stage 3: exogenous features ----------------
    _stage_banner("Exogenous Yahoo Features")
    stage_started = time.perf_counter()
    ap = args.artifact_prefix
    exog_csv = project_root / "data" / "processed" / f"{ap}_exogenous_features.csv"
    exog_quality = project_root / "data" / "processed" / f"{ap}_exogenous_quality.json"
    exogenous_artifacts: dict[str, pathlib.Path] = {}
    if not exog_csv.exists() or not exog_quality.exists():
        markets_p = project_root / "data" / "processed" / f"{ap}_markets_filtered.csv"
        history_p = project_root / "data" / "processed" / f"{ap}_price_history.csv"
        baseline_ts_p = project_root / "data" / "processed" / f"{ap}_baseline_timeseries.csv"
        ts_values, _, _ = _build_price_matrix(_read_csv(markets_p), _read_csv(history_p))
        if len(ts_values) >= 2:
            step_ts = ts_values[1:]
            baseline_rows = _read_series(baseline_ts_p)
            baseline_ts_list = [_to_int(row.get("timestamp"), default=-1) for row in baseline_rows]
            if baseline_ts_list != step_ts:
                raise RuntimeError(
                    "Exogenous step timestamps must match baseline_timeseries exactly."
                )
            split_idx_exog = split_index_for_returns(len(step_ts), experiment_config)
            csv_p, json_p = build_and_save_exogenous(
                project_root,
                ap,
                step_ts,
                split_idx_exog,
                baseline_timestamps=baseline_ts_list,
            )
            exogenous_artifacts = {"exogenous_features": csv_p, "exogenous_quality": json_p}
    else:
        # Reuse the cached exogenous CSV from the input prefix (already aligned
        # to baseline timestamps).
        in_exog = project_root / "data" / "processed" / f"{args.input_artifact_prefix}_exogenous_features.csv"
        in_exog_q = project_root / "data" / "processed" / f"{args.input_artifact_prefix}_exogenous_quality.json"
        import shutil

        if in_exog.exists() and not exog_csv.exists():
            shutil.copy2(in_exog, exog_csv)
        if in_exog_q.exists() and not exog_quality.exists():
            shutil.copy2(in_exog_q, exog_quality)
        exogenous_artifacts = {
            "exogenous_features": exog_csv,
            "exogenous_quality": exog_quality,
        }
    # If still missing (no cached weekly inputs and build skipped), generate.
    if "exogenous_features" not in exogenous_artifacts or not exogenous_artifacts["exogenous_features"].exists():
        markets_p = project_root / "data" / "processed" / f"{ap}_markets_filtered.csv"
        history_p = project_root / "data" / "processed" / f"{ap}_price_history.csv"
        baseline_ts_p = project_root / "data" / "processed" / f"{ap}_baseline_timeseries.csv"
        ts_values, _, _ = _build_price_matrix(_read_csv(markets_p), _read_csv(history_p))
        step_ts = ts_values[1:]
        baseline_rows = _read_series(baseline_ts_p)
        baseline_ts_list = [_to_int(row.get("timestamp"), default=-1) for row in baseline_rows]
        split_idx_exog = split_index_for_returns(len(step_ts), experiment_config)
        csv_p, json_p = build_and_save_exogenous(
            project_root,
            ap,
            step_ts,
            split_idx_exog,
            baseline_timestamps=baseline_ts_list,
        )
        exogenous_artifacts = {"exogenous_features": csv_p, "exogenous_quality": json_p}
    exogenous_sec = time.perf_counter() - stage_started
    print(f"Exogenous features complete in {exogenous_sec:.1f}s")
    for k, v in exogenous_artifacts.items():
        print(f"- {k}: {v}")

    # ---------------- Stage 4: Kelly Optuna search ----------------
    n_trials = args.optuna_trials if args.optuna_trials is not None else OPTUNA_N_TRIALS
    if args.skip_optuna:
        _stage_banner("Kelly Optuna search (SKIPPED — resuming from disk)")
        kelly_artifacts: dict[str, pathlib.Path] = {}
        for key in (
            "kelly_grid",
            "kelly_best_metrics",
            "kelly_best_timeseries",
            "kelly_best_weights",
            "kelly_copula_diagnostics",
        ):
            stem_part = key.replace("kelly_", "")
            ext = "csv" if "timeseries" in key or "weights" in key or "grid" in key else "json"
            ext = "csv" if "grid" in key or "timeseries" in key or "weights" in key else "json"
            candidate = (
                project_root
                / "data"
                / "processed"
                / f"{ap}_kelly_{stem_part}.{ext}"
            )
            if candidate.exists():
                kelly_artifacts[key] = candidate
        if "kelly_best_metrics" not in kelly_artifacts:
            raise SystemExit(
                "--skip-optuna requested but cached kelly_best_metrics file is missing."
            )
        kelly_sec = 0.0
    else:
        _stage_banner(f"Kelly Optuna QMC search ({n_trials} trials)")
        stage_started = time.perf_counter()
        kelly_artifacts = run_kelly_optuna_search(
            project_root,
            input_artifact_prefix=ap,  # we copied input -> output prefix already
            output_artifact_prefix=ap,
            config=experiment_config,
            n_trials=n_trials,
            mc_samples_override=args.mc_samples_override,
            turnover_lambda_override=args.lambda_turnover_override,
            fee_rate_override=args.fee_rate_override,
            dd_penalty_override=args.dd_penalty_override,
            disable_copula=bool(args.no_exogenous),
        )
        kelly_sec = time.perf_counter() - stage_started
        print(f"Kelly search complete in {kelly_sec / 60:.1f}m")
    for k, v in kelly_artifacts.items():
        print(f"- {k}: {v}")

    # ---------------- Stage 5: figures ----------------
    _stage_banner("Figure Generation")
    stage_started = time.perf_counter()
    figure_artifacts = _make_kelly_figures(project_root, artifact_prefix=ap)
    figures_sec = time.perf_counter() - stage_started
    print(f"Figures complete in {figures_sec:.1f}s")
    for k, v in figure_artifacts.items():
        print(f"- {k}: {v}")

    # ---------------- Stage 6: diagnostics report ----------------
    _stage_banner("Week 10 Diagnostics Report")
    stage_started = time.perf_counter()
    week10_report_path = _make_week10_diagnostics_report(
        project_root,
        artifact_prefix=ap,
        min_history_days_used=used_min_history_days,
        history_fidelity_min=int(base_build_config.history_fidelity),
    )
    report_sec = time.perf_counter() - stage_started
    print(f"Report complete in {report_sec:.1f}s")
    print(f"- diagnostics_report: {week10_report_path}")

    # ---------------- Manifest ----------------
    manifest_path = _write_run_manifest(
        project_root=project_root,
        artifact_prefix=ap,
        config_hash=config_hash,
        stage_durations_sec={
            "data_build_or_reuse": data_sec,
            "baseline": baseline_sec,
            "exogenous_features": exogenous_sec,
            "kelly_search": kelly_sec,
            "figure_generation": figures_sec,
            "week10_report_generation": report_sec,
            "total": time.perf_counter() - run_started,
        },
        used_min_history_days=used_min_history_days,
        artifact_groups={
            "data": {k: str(v) for k, v in data_artifacts.items()},
            "baseline": {k: str(v) for k, v in baseline_artifacts.items()},
            "exogenous": {k: str(v) for k, v in exogenous_artifacts.items()},
            "kelly": {k: str(v) for k, v in kelly_artifacts.items()},
            "figures": {k: str(v) for k, v in figure_artifacts.items()},
            "reports": {"week10_kelly_diagnostics_report": str(week10_report_path)},
        },
        extra={
            "input_artifact_prefix": args.input_artifact_prefix,
            "rebuild_data": bool(args.rebuild_data),
            "no_exogenous": bool(args.no_exogenous),
            "mc_samples_override": args.mc_samples_override,
            "lambda_turnover_override": args.lambda_turnover_override,
            "fee_rate_override": args.fee_rate_override,
            "fee_rate_values": list(fee_rates_cfg),
            "dd_penalty_override": args.dd_penalty_override,
            "dd_penalty_values": list(dd_penalty_cfg),
            "git_commit_and_push_requested": bool(args.git_commit_and_push),
        },
    )
    total_sec = time.perf_counter() - run_started
    print(f"\n{'#' * 60}", flush=True)
    print(f"  PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {total_sec / 60:.1f}m ({total_sec / 3600:.1f}h)", flush=True)
    print(f"{'#' * 60}", flush=True)
    print(f"- run_manifest: {manifest_path}")

    if args.git_commit_and_push:
        tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
        commit_msg = args.git_commit_message or f"Week10 Kelly pipeline cloud run {tag}"
        _stage_banner("Git commit and push")
        _git_commit_and_push(
            project_root,
            remote=args.git_remote,
            branch=args.git_push_branch,
            message=commit_msg,
        )


if __name__ == "__main__":
    main()
