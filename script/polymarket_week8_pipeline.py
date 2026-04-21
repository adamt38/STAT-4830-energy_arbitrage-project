"""End-to-end Week 8 pipeline after Week 4 prototype."""

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
from src.constrained_optimizer import ExperimentConfig, run_experiment_grid, run_optuna_search, split_index_for_returns
from src.exogenous_features import build_and_save_exogenous
from src.covariance_diagnostics import run_covariance_diagnostics
from src.polymarket_data import BuildConfig, NoMarketsAfterHistoryFilterError, build_dataset


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
    """Stage processed outputs, commit if needed, pull --rebase, push HEAD to remote branch."""
    if not (project_root / ".git").is_dir():
        print("Skipping git publish: no .git directory (not a git checkout).", flush=True)
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
        p = project_root / rel
        if p.exists():
            paths_to_add.append(rel)
    if not paths_to_add:
        print("Skipping git publish: data/processed, figures, and docs are all missing.", flush=True)
        return

    add_r = _run(["add", *paths_to_add])
    if add_r.returncode != 0:
        raise RuntimeError(
            f"git add failed ({add_r.returncode}):\n{add_r.stderr or add_r.stdout}"
        )

    diff_r = _run(["diff", "--cached", "--quiet"])
    if diff_r.returncode == 0:
        print("Git publish: nothing new to commit; skipping pull/push.", flush=True)
        return

    commit_r = _run(["commit", "-m", message])
    if commit_r.returncode != 0:
        raise RuntimeError(
            f"git commit failed ({commit_r.returncode}):\n{commit_r.stderr or commit_r.stdout}"
        )

    pull_r = _run(["pull", "--rebase", remote, branch])
    if pull_r.returncode != 0:
        err = (pull_r.stderr or "") + (pull_r.stdout or "")
        if "couldn't find remote ref" in err or "could not find remote ref" in err:
            print(
                f"Git publish: pull --rebase skipped (remote branch {remote}/{branch!r} missing); "
                "attempting push only.",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"git pull --rebase {remote} {branch} failed ({pull_r.returncode}):\n{err}"
            )

    push_r = _run(["push", remote, f"HEAD:{branch}"])
    if push_r.returncode != 0:
        raise RuntimeError(
            f"git push {remote} HEAD:{branch} failed ({push_r.returncode}):\n"
            f"{push_r.stderr or push_r.stdout}"
        )

    head_r = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    head_name = (head_r.stdout or "").strip() or "?"
    print(
        f"Git publish: committed and pushed branch {head_name!r} to {remote}/{branch}.",
        flush=True,
    )


def _make_week9_diagnostics_report(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    min_history_days_used: float,
    constrained_artifact_stem: str | None = None,
) -> pathlib.Path:
    """Create a compact week 9 markdown report from latest artifacts.

    ``artifact_prefix`` selects baseline, exogenous, and covariance files (e.g. ``week8``).
    ``constrained_artifact_stem`` overrides paths for Optuna holdout outputs when using
    ``output_artifact_suffix`` (e.g. ``week8_macro_explicit``); defaults to ``artifact_prefix``.
    """
    processed = project_root / "data" / "processed"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    cstem = constrained_artifact_stem if constrained_artifact_stem is not None else artifact_prefix

    baseline_metrics = _read_json(processed / f"{artifact_prefix}_baseline_metrics.json")
    constrained_metrics = _read_json(processed / f"{cstem}_constrained_best_metrics.json")
    covariance_summary = _read_json(processed / f"{artifact_prefix}_covariance_summary.json")
    attribution_summary = _read_json(
        processed / f"{cstem}_constrained_best_attribution_summary.json"
    )
    market_contrib = _read_series(
        processed / f"{cstem}_constrained_best_market_return_contributions.csv"
    )
    domain_contrib = _read_series(
        processed / f"{cstem}_constrained_best_domain_return_contributions.csv"
    )
    corr_pairs = _read_series(
        processed / f"{cstem}_constrained_best_top_market_correlation_pairs.csv"
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

    def _max_dd_masked(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        c = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(c)
        dd = c / np.clip(peak, 1e-8, None) - 1.0
        return float(np.min(dd))

    exog_path = processed / f"{artifact_prefix}_exogenous_features.csv"
    open_closed_section: list[str] = []
    if exog_path.exists() and holdout_steps_total > 0:
        exog_rows = _read_series(exog_path)
        exog_holdout = exog_rows[-holdout_steps_total:]
        constrained_ts_path = processed / f"{cstem}_constrained_best_timeseries.csv"
        constrained_hold = _read_series(constrained_ts_path) if constrained_ts_path.exists() else []
        ch_rets = np.array([float(r["portfolio_return"]) for r in constrained_hold], dtype=float)
        if (
            len(exog_holdout) == len(baseline_holdout_rows)
            and ch_rets.size == baseline_holdout_rets.size
            and baseline_holdout_rets.size > 0
        ):
            open_flags = np.array(
                [_to_int(r.get("is_equity_open", 0), default=0) for r in exog_holdout],
                dtype=int,
            )
            stale_flags = np.array(
                [_to_int(r.get("exog_is_stale", 1), default=1) for r in exog_holdout],
                dtype=int,
            )
            open_m = open_flags == 1
            closed_m = open_flags == 0
            pct_open = 100.0 * float(np.mean(open_flags))
            pct_stale = 100.0 * float(np.mean(stale_flags))
            bl_open = baseline_holdout_rets[open_m]
            bl_closed = baseline_holdout_rets[closed_m]
            ch_open = ch_rets[open_m]
            ch_closed = ch_rets[closed_m]
            open_closed_section = [
                "",
                "## Holdout — US equity session vs closed (exogenous mask)",
                "",
                "Subset metrics use chronological holdout steps where `is_equity_open` is 1 "
                "(NYSE regular hours, Mon–Fri 09:30–16:00 ET; exchange holidays are not excluded). "
                "Max drawdown on each subset uses cumulative wealth `cumprod(1+r)` over **only** those steps "
                "(gapped timeline, not calendar-interpolated).",
                "",
                f"- Holdout steps with equity open: `{pct_open:.1f}%`",
                f"- Holdout steps marked exog-stale: `{pct_stale:.1f}%`",
                "",
                "| Subset | Metric | Baseline | Constrained |",
                "|--------|--------|----------|-------------|",
                f"| Open | Sortino | {_sortino_np(bl_open):.4f} | {_sortino_np(ch_open):.4f} |",
                f"| Open | Mean return | {float(np.mean(bl_open)) if bl_open.size else 0.0:.8f} | "
                f"{float(np.mean(ch_open)) if ch_open.size else 0.0:.8f} |",
                f"| Open | Volatility | {float(np.std(bl_open)) if bl_open.size else 0.0:.8f} | "
                f"{float(np.std(ch_open)) if ch_open.size else 0.0:.8f} |",
                f"| Open | Max drawdown (subset) | {_max_dd_masked(bl_open):.4%} | {_max_dd_masked(ch_open):.4%} |",
                f"| Closed | Sortino | {_sortino_np(bl_closed):.4f} | {_sortino_np(ch_closed):.4f} |",
                f"| Closed | Mean return | {float(np.mean(bl_closed)) if bl_closed.size else 0.0:.8f} | "
                f"{float(np.mean(ch_closed)) if ch_closed.size else 0.0:.8f} |",
                f"| Closed | Volatility | {float(np.std(bl_closed)) if bl_closed.size else 0.0:.8f} | "
                f"{float(np.std(ch_closed)) if ch_closed.size else 0.0:.8f} |",
                f"| Closed | Max drawdown (subset) | {_max_dd_masked(bl_closed):.4%} | {_max_dd_masked(ch_closed):.4%} |",
            ]
        else:
            open_closed_section = [
                "",
                "## Holdout — US equity session vs closed",
                "",
                "*Skipped: exogenous holdout row count, baseline holdout, or constrained timeseries length mismatch.*",
            ]

    lines = [
        "# Week 9 Diagnostics Report",
        "",
        "## Run Context",
        f"- artifact prefix: `{artifact_prefix}`",
        f"- constrained artifact stem: `{cstem}`",
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
        *open_closed_section,
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

    report_path = docs_dir / "week9_diagnostics_report.md"
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


def _load_cached_dataset_artifacts(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
) -> dict[str, pathlib.Path]:
    """Return required cached dataset artifacts if they all exist."""
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
    missing = [str(path) for path in artifacts.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cached dataset fallback requested, but required artifacts are missing: "
            + ", ".join(missing)
        )
    markets_rows = _read_series(artifacts["markets_filtered"])
    price_rows = _read_series(artifacts["price_history"])
    if not markets_rows or not price_rows:
        raise RuntimeError(
            "Cached dataset artifacts exist but are empty. "
            "A previous failed fetch likely overwrote the processed files for this artifact prefix."
        )
    return artifacts


def _make_figures(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str = "week8",
    constrained_suffix: str = "",
) -> dict[str, pathlib.Path]:
    processed = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cstem = f"{artifact_prefix}{constrained_suffix}"
    baseline_series = _read_series(processed / f"{artifact_prefix}_baseline_timeseries.csv")
    constrained_series = _read_series(processed / f"{cstem}_constrained_best_timeseries.csv")
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
    equity_path = figures_dir / f"{artifact_prefix}_iteration_equity_curve_comparison.png"
    fig_equity.savefig(equity_path, dpi=120)
    plt.close(fig_equity)

    fig_dd, ax_dd = plt.subplots()
    ax_dd.plot(baseline_dd, label="Equal-weight baseline")
    ax_dd.plot(constrained_dd, label="Constrained OGD/SGD")
    ax_dd.set_title("Portfolio Drawdown")
    ax_dd.set_xlabel("Step")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.legend()
    dd_path = figures_dir / f"{artifact_prefix}_iteration_drawdown_comparison.png"
    fig_dd.savefig(dd_path, dpi=120)
    plt.close(fig_dd)

    with (processed / f"{artifact_prefix}_baseline_metrics.json").open("r", encoding="utf-8") as handle:
        baseline_metrics = json.load(handle)
    with (processed / f"{cstem}_constrained_best_metrics.json").open("r", encoding="utf-8") as handle:
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
    exposure_table_path = processed / f"{artifact_prefix}_category_exposure_table.csv"
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
    exposure_path = figures_dir / f"{artifact_prefix}_iteration_category_exposure_comparison.png"
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
    rolling_path = figures_dir / f"{artifact_prefix}_iteration_rolling_mean_return_comparison.png"
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
    hist_path = figures_dir / f"{artifact_prefix}_iteration_return_distribution_comparison.png"
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
    exposure_delta_path = figures_dir / f"{artifact_prefix}_iteration_top_exposure_deltas.png"
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
    risk_return_path = figures_dir / f"{artifact_prefix}_iteration_risk_return_snapshot.png"
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
    parser = argparse.ArgumentParser(description="Week 8 Polymarket pipeline (data → baseline → exog → Optuna → report).")
    parser.add_argument(
        "--macro-integration",
        choices=("rescale", "explicit", "both"),
        default="rescale",
        help="Macro objective path for Optuna (default rescale = legacy m_risk + uniform_mix bumps).",
    )
    parser.add_argument(
        "--macro-modes",
        default=None,
        help="Comma-separated macro modes (rescale,explicit,both). Runs Optuna once per mode with "
        "output_artifact_suffix _macro_<mode> (none for rescale) so files do not overwrite.",
    )
    parser.add_argument(
        "--optuna-artifact-suffix",
        default=None,
        help="Suffix appended to week8 for constrained outputs (default: empty for rescale, else _macro_<mode>).",
    )
    parser.add_argument(
        "--joint-macro-mode-search",
        action="store_true",
        help="Single Optuna study with categorical macro_mode and conditional regime_k / lambda_macro_explicit.",
    )
    parser.add_argument(
        "--etf-tracking",
        action="store_true",
        help="Penalize deviation of rolling-window portfolio returns from equal-weight SPY/QQQ/XLE/TLT/BTC log returns "
        "(from exogenous_features CSV). Optuna searches lambda_etf_tracking.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Override Optuna trial count (default 100, or 5 when QUICK_SANITY_CHECK is True).",
    )
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna trials (passed to study.optimize n_jobs). Use 1 for sequential (default). "
        "Use -1 for auto (cpu_count). On many-core pods, try 16–32 with OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip the Optuna search stage entirely and resume the pipeline from covariance "
        "diagnostics onward, reusing previously-written *_constrained_best_*.json/csv artifacts. "
        "Use this when an earlier run finished Optuna but crashed in a downstream stage.",
    )
    parser.add_argument(
        "--git-commit-and-push",
        action="store_true",
        help="After a successful run: git add data/processed figures docs, commit, "
        "git pull --rebase <remote> <branch>, git push <remote> HEAD:<branch>. "
        "Requires git credentials (SSH key or token) and a normal checkout on the machine.",
    )
    parser.add_argument(
        "--git-remote",
        default="origin",
        help="Remote for --git-commit-and-push (default: origin).",
    )
    parser.add_argument(
        "--git-push-branch",
        default="cloud-runs",
        help="Branch for pull --rebase and push HEAD:<branch> (default: cloud-runs).",
    )
    parser.add_argument(
        "--git-commit-message",
        default=None,
        metavar="MSG",
        help="Commit message for --git-commit-and-push. Default: timestamped auto message.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="week8",
        help="Stem for every artifact this run writes (raw cache, processed CSV/JSON, "
        "figures, manifest, week 9 report). Use unique values per pod when running "
        "experiments in parallel (e.g. week8_A, week8_B, ...) so branches merge cleanly.",
    )
    parser.add_argument(
        "--top-k-bagging",
        type=int,
        default=1,
        help="Average the holdout portfolio returns of the top-K Optuna trials (by walk-forward "
        "objective) instead of using a single best trial. K=1 (default) preserves legacy behavior. "
        "Reduces selection bias from many-trial Sobol search; recommended K=5–10.",
    )
    parser.add_argument(
        "--baseline-shrinkage",
        action="store_true",
        help="Add baseline_alpha ∈ [0, 1] as an Optuna hyperparameter. Final per-step return is "
        "alpha * r_constrained + (1 - alpha) * r_equalweight; Optuna picks alpha. Acts as "
        "shrinkage-to-uniform at the portfolio level.",
    )
    parser.add_argument(
        "--beat-baseline-objective",
        action="store_true",
        help="Change the Optuna objective from Sortino(constrained) to "
        "Sortino(constrained) - Sortino(equal-weight) on the same walk-forward folds. "
        "Pushes the search toward configurations that *beat* equal-weight rather than just "
        "maximizing absolute Sortino.",
    )
    parser.add_argument(
        "--reduced-search",
        action="store_true",
        help="Narrow the Optuna search space to ranges around the values that converged across "
        "Round 1 pods (drops outer extremes for lr, domain_limit, max_weight, variance/downside "
        "penalties; fixes entropy_lambda at 0). Same trial count → denser Sobol coverage.",
    )
    parser.add_argument(
        "--momentum-screening",
        action="store_true",
        help="Pre-screen markets by absolute recent price momentum before the optimizer sees them. "
        "Ranks candidate markets by |return over the last N days| and keeps only the top-K movers. "
        "Addresses the 'weak signal / dead-weight markets' problem diagnosed in the Week 9 synthesis: "
        "reduces parameter count so constraint thresholds become binding and per-market tilts are "
        "large enough to rise above gradient noise. Combines cleanly with --reduced-search, "
        "--top-k-bagging, and the macro modes (orthogonal lever).",
    )
    parser.add_argument(
        "--momentum-top-n",
        type=int,
        default=20,
        help="Number of markets to retain after momentum screening (default 20). "
        "Only used when --momentum-screening is set. Must be <= --max-markets; if not set, "
        "defaults to 20.",
    )
    parser.add_argument(
        "--momentum-lookback-days",
        type=float,
        default=5.0,
        help="Window (in days) for computing per-market momentum (default 5.0). "
        "Only used when --momentum-screening is set. Shorter windows (e.g. 3.0) capture "
        "very recent price action; longer windows (e.g. 7.0) smooth out noise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed passed to ExperimentConfig.seed (default 7). Use a different value "
        "for replication runs to verify that a positive holdout delta is not seed-specific.",
    )
    parser.add_argument(
        "--learnable-inclusion",
        action="store_true",
        help="Direction A (meta-selection): add a second parameter vector (inclusion logits, "
        "one per market) to the online optimizer. Effective weight is sigmoid(s_i) * softmax(w)_i, "
        "renormalized. Both vectors are jointly optimized by Adam/SGD. Two regularizers keep the "
        "gates well-behaved: a soft cardinality penalty targeting --inclusion-target-k active "
        "markets, and a binary-entropy penalty pushing each gate to 0 or 1. Initial gates are "
        "seeded from a momentum proxy (cumulative return over the warmup window). Generalizes "
        "--momentum-screening from a hand-picked top-K to a learned online selection. Use with "
        "the FULL universe (do NOT pass --momentum-screening).",
    )
    parser.add_argument(
        "--inclusion-target-k",
        type=int,
        default=15,
        help="Target number of active markets for learnable inclusion (default 15). "
        "The cardinality regularizer pushes Σ sigmoid(inclusion_logits) toward this value.",
    )
    parser.add_argument(
        "--inclusion-init-gain",
        type=float,
        default=2.0,
        help="Scale factor for the momentum-proxy initialization of inclusion logits (default 2.0). "
        "Larger values start high-|momentum| markets with gates closer to 1 and low-|momentum| "
        "markets with gates closer to 0. Optuna searches over this range when --learnable-inclusion.",
    )
    parser.add_argument(
        "--lambda-inclusion-cardinality",
        type=float,
        default=1.0,
        help="Strength of the cardinality penalty for learnable inclusion (default 1.0). Optuna "
        "searches this range when --learnable-inclusion is set.",
    )
    parser.add_argument(
        "--lambda-inclusion-commitment",
        type=float,
        default=0.05,
        help="Strength of the binary-entropy commitment penalty for learnable inclusion "
        "(default 0.05). Larger values push gates harder toward 0 or 1.",
    )
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
        momentum_screening_enabled=bool(args.momentum_screening),
        momentum_lookback_days=float(args.momentum_lookback_days),
        momentum_top_n=(
            int(args.momentum_top_n) if args.momentum_screening else None
        ),
    )
    # ── Toggle: set QUICK_SANITY_CHECK = False for the full Optuna run ──
    QUICK_SANITY_CHECK = False
    OPTUNA_N_TRIALS = 5 if QUICK_SANITY_CHECK else 100

    # With 10-min data we have ~6× more time steps; scale walk-forward so fold count stays similar
    if base_build_config.history_fidelity <= 10:
        walkforward_train_steps = 1440  # ~10 days in 10-min steps
        walkforward_test_steps = 288    # ~2 days in 10-min steps
    else:
        walkforward_train_steps = 240
        walkforward_test_steps = 48

    experiment_config = ExperimentConfig(
        learning_rates=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        penalties_lambda=(0.25, 0.5, 1.0, 2.0),
        rolling_windows=(24, 48, 96),
        steps_per_window=3,
        objective="mean_downside",
        variance_penalty=1.0,
        downside_penalty=2.0,
        variance_penalties=(0.5, 1.0, 2.0),
        downside_penalties=(1.0, 2.0, 3.0),
        optimizer_type="adam",
        weight_parameterization="projected_simplex",  # switch to "projected_simplex" to enable PGD-style updates
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
        seed=int(args.seed),
        optuna_n_jobs=args.optuna_n_jobs,
        learnable_inclusion_enabled=bool(args.learnable_inclusion),
        inclusion_target_k=int(args.inclusion_target_k),
        inclusion_init_gain=float(args.inclusion_init_gain),
        lambda_inclusion_cardinality=float(args.lambda_inclusion_cardinality),
        lambda_inclusion_commitment=float(args.lambda_inclusion_commitment),
    )
    macro_modes_list = (
        [x.strip() for x in args.macro_modes.split(",") if x.strip()] if args.macro_modes else []
    )
    for m in macro_modes_list:
        if m not in ("rescale", "explicit", "both"):
            raise SystemExit(f"Invalid --macro-modes entry {m!r}; expected rescale, explicit, or both.")
    if args.joint_macro_mode_search and macro_modes_list:
        raise SystemExit("Use either --joint-macro-mode-search or --macro-modes, not both.")
    if not macro_modes_list and not args.joint_macro_mode_search:
        experiment_config = replace(experiment_config, macro_integration=args.macro_integration)
    if args.etf_tracking:
        experiment_config = replace(experiment_config, use_etf_tracking=True)
    if args.reduced_search:
        experiment_config = replace(
            experiment_config,
            learning_rates=(0.005, 0.01, 0.02, 0.05),
            penalties_lambda=(0.25, 0.5, 1.0),
            domain_limits=(0.08, 0.12),
            max_weights=(0.04, 0.06, 0.10),
            entropy_lambdas=(0.0,),
            variance_penalties=(1.0, 2.0),
            downside_penalties=(2.0, 3.0),
            uniform_mixes=(0.0, 0.05, 0.1),
        )
    config_hash = _config_hash(base_build_config, experiment_config)

    def _stage_banner(name: str) -> None:
        elapsed_total = time.perf_counter() - run_started
        print(f"\n{'#'*60}", flush=True)
        print(f"  PIPELINE STAGE: {name}", flush=True)
        print(f"  (total elapsed: {elapsed_total / 60:.1f}m)", flush=True)
        print(f"{'#'*60}\n", flush=True)

    if args.momentum_screening:
        print(
            f"\n[CONFIG] momentum screening ENABLED: "
            f"top_n={args.momentum_top_n}, lookback_days={args.momentum_lookback_days}",
            flush=True,
        )
    if args.learnable_inclusion:
        print(
            f"\n[CONFIG] learnable inclusion ENABLED (Direction A): "
            f"target_k={args.inclusion_target_k}, init_gain={args.inclusion_init_gain}, "
            f"lambda_card={args.lambda_inclusion_cardinality}, "
            f"lambda_commit={args.lambda_inclusion_commitment}",
            flush=True,
        )
        if args.momentum_screening:
            print(
                "  WARNING: --momentum-screening and --learnable-inclusion are both set. "
                "Learnable inclusion is designed for the FULL universe; momentum pre-filter "
                "reduces the universe before the gates see it. Expected behavior may degrade.",
                flush=True,
            )

    _stage_banner("Data Build")
    stage_started = time.perf_counter()
    try:
        data_artifacts, used_min_history_days = _build_dataset_with_history_backoff(
            project_root=project_root,
            base_config=base_build_config,
            history_day_candidates=(24.0, 18.0, 12.0, 7.0),
        )
    except NoMarketsAfterHistoryFilterError as exc:
        print(f"Data build failed ({exc}); falling back to cached processed dataset.")
        data_artifacts = _load_cached_dataset_artifacts(
            project_root=project_root,
            artifact_prefix=base_build_config.artifact_prefix,
        )
        used_min_history_days = base_build_config.min_history_days
    data_sec = time.perf_counter() - stage_started
    print(f"Data build complete in {data_sec / 60:.1f}m")
    print(f"- min_history_days_used: {used_min_history_days}")
    for key, value in data_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Equal-Weight Baseline")
    stage_started = time.perf_counter()
    baseline_result = run_equal_weight_baseline(project_root, artifact_prefix=args.artifact_prefix)
    baseline_artifacts = save_baseline_outputs(project_root, baseline_result, artifact_prefix=args.artifact_prefix)
    baseline_sec = time.perf_counter() - stage_started
    print(f"Baseline complete in {baseline_sec:.1f}s")
    print(f"- markets: {baseline_result.market_count}")
    print(f"- sortino: {pretty_float(baseline_result.sortino)}")
    print(f"- max_drawdown: {pretty_pct(baseline_result.max_drawdown)}")
    for key, value in baseline_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Exogenous Yahoo features")
    stage_started = time.perf_counter()
    exogenous_artifacts: dict[str, pathlib.Path] = {}
    ap = args.artifact_prefix
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
                "Exogenous step timestamps must match baseline_timeseries exactly (order and values)."
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
    exogenous_sec = time.perf_counter() - stage_started
    print(f"Exogenous features complete in {exogenous_sec:.1f}s")
    for key, value in exogenous_artifacts.items():
        print(f"- {key}: {value}")

    n_trials_opt = args.optuna_trials if args.optuna_trials is not None else OPTUNA_N_TRIALS
    manifest_constrained_flat: dict[str, str] = {}
    last_optuna_suffix = ""
    constrained_artifacts: dict[str, pathlib.Path] = {}
    constrained_sec = 0.0

    def _resume_suffix_from_args() -> str:
        if macro_modes_list:
            last_mode = macro_modes_list[-1]
            return "" if last_mode == "rescale" else f"_macro_{last_mode}"
        suf = args.optuna_artifact_suffix
        if suf is None:
            suf = (
                ""
                if (args.joint_macro_mode_search or args.macro_integration == "rescale")
                else f"_macro_{args.macro_integration}"
            )
        return suf

    def _reconstruct_artifacts_from_disk(suffix: str) -> dict[str, pathlib.Path]:
        processed_dir = project_root / "data" / "processed"
        stem = f"{args.artifact_prefix}{suffix}"
        candidates = {
            "constrained_grid": processed_dir / f"{args.artifact_prefix}_constrained_experiment_grid.csv",
            "constrained_best_metrics": processed_dir / f"{stem}_constrained_best_metrics.json",
            "constrained_best_timeseries": processed_dir / f"{stem}_constrained_best_timeseries.csv",
        }
        return {k: v for k, v in candidates.items() if v.exists()}

    if args.skip_optuna:
        _stage_banner("Optuna quasi-random search (SKIPPED — resuming from disk)")
        last_optuna_suffix = _resume_suffix_from_args()
        constrained_artifacts = _reconstruct_artifacts_from_disk(last_optuna_suffix)
        manifest_constrained_flat = {k: str(v) for k, v in constrained_artifacts.items()}
        required = (
            project_root
            / "data"
            / "processed"
            / f"{args.artifact_prefix}{last_optuna_suffix}_constrained_best_metrics.json"
        )
        if not required.exists():
            raise SystemExit(
                f"--skip-optuna requested but {required} is missing. "
                "Cannot resume; rerun without --skip-optuna or check --macro-integration / --macro-modes."
            )
        print(f"Reusing constrained artifacts with suffix {last_optuna_suffix!r}:")
    else:
        _stage_banner(f"Optuna quasi-random search ({n_trials_opt} trials, QMCSampler)")
        stage_started = time.perf_counter()
        if macro_modes_list:
            constrained_sec_total = 0.0
            for m in macro_modes_list:
                suf = "" if m == "rescale" else f"_macro_{m}"
                cfg_m = replace(experiment_config, macro_integration=m)
                t0 = time.perf_counter()
                arts = run_optuna_search(
                    project_root,
                    artifact_prefix=args.artifact_prefix,
                    config=cfg_m,
                    n_trials=n_trials_opt,
                    joint_macro_mode_search=False,
                    output_artifact_suffix=suf,
                    top_k_bagging=int(args.top_k_bagging),
                    baseline_shrinkage=bool(args.baseline_shrinkage),
                    beat_baseline_objective=bool(args.beat_baseline_objective),
                )
                constrained_sec_total += time.perf_counter() - t0
                for k, v in arts.items():
                    manifest_constrained_flat[f"{m}__{k}"] = str(v)
                constrained_artifacts = arts
                last_optuna_suffix = suf
            constrained_sec = constrained_sec_total
        else:
            suf_single = args.optuna_artifact_suffix
            if suf_single is None:
                suf_single = (
                    ""
                    if (args.joint_macro_mode_search or args.macro_integration == "rescale")
                    else f"_macro_{args.macro_integration}"
                )
            constrained_artifacts = run_optuna_search(
                project_root,
                artifact_prefix=args.artifact_prefix,
                config=experiment_config,
                n_trials=n_trials_opt,
                joint_macro_mode_search=args.joint_macro_mode_search,
                output_artifact_suffix=suf_single,
                top_k_bagging=int(args.top_k_bagging),
                baseline_shrinkage=bool(args.baseline_shrinkage),
                beat_baseline_objective=bool(args.beat_baseline_objective),
            )
            constrained_sec = time.perf_counter() - stage_started
            manifest_constrained_flat = {k: str(v) for k, v in constrained_artifacts.items()}
            last_optuna_suffix = suf_single
        print(f"\nOptuna search complete in {constrained_sec / 60:.1f}m ({constrained_sec / 3600:.1f}h)")
    for key, value in constrained_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Covariance Diagnostics")
    stage_started = time.perf_counter()
    covariance_artifacts = run_covariance_diagnostics(
        project_root,
        artifact_prefix=args.artifact_prefix,
        constrained_suffix=last_optuna_suffix,
    )
    covariance_sec = time.perf_counter() - stage_started
    print(f"Covariance diagnostics complete in {covariance_sec:.1f}s")
    for key, value in covariance_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Figure Generation")
    stage_started = time.perf_counter()
    figure_artifacts = _make_figures(
        project_root,
        artifact_prefix=args.artifact_prefix,
        constrained_suffix=last_optuna_suffix,
    )
    figures_sec = time.perf_counter() - stage_started
    print(f"Figures complete in {figures_sec:.1f}s")
    for key, value in figure_artifacts.items():
        print(f"- {key}: {value}")

    _stage_banner("Week 9 Diagnostics Report")
    stage_started = time.perf_counter()
    week9_cstem = f"{args.artifact_prefix}{last_optuna_suffix}" if last_optuna_suffix else args.artifact_prefix
    week9_report_path = _make_week9_diagnostics_report(
        project_root=project_root,
        artifact_prefix=args.artifact_prefix,
        min_history_days_used=used_min_history_days,
        constrained_artifact_stem=week9_cstem if week9_cstem != args.artifact_prefix else None,
    )
    report_sec = time.perf_counter() - stage_started
    print(f"Report complete in {report_sec:.1f}s")
    print(f"- diagnostics_report: {week9_report_path}")

    manifest_path = _write_run_manifest(
        project_root=project_root,
        artifact_prefix=args.artifact_prefix,
        config_hash=config_hash,
        stage_durations_sec={
            "data_build": data_sec,
            "baseline": baseline_sec,
            "exogenous_features": exogenous_sec,
            "constrained_grid_and_holdout": constrained_sec,
            "covariance_diagnostics": covariance_sec,
            "figure_generation": figures_sec,
            "week9_report_generation": report_sec,
            "total": time.perf_counter() - run_started,
        },
        used_min_history_days=used_min_history_days,
        artifact_groups={
            "data": {k: str(v) for k, v in data_artifacts.items()},
            "baseline": {k: str(v) for k, v in baseline_artifacts.items()},
            "exogenous": {k: str(v) for k, v in exogenous_artifacts.items()},
            "constrained": manifest_constrained_flat,
            "covariance": {k: str(v) for k, v in covariance_artifacts.items()},
            "figures": {k: str(v) for k, v in figure_artifacts.items()},
            "reports": {"week9_diagnostics_report": str(week9_report_path)},
        },
        extra={
            "macro_modes_ran": macro_modes_list if macro_modes_list else None,
            "single_macro_integration": (
                None
                if macro_modes_list or args.joint_macro_mode_search
                else args.macro_integration
            ),
            "joint_macro_mode_search": args.joint_macro_mode_search,
            "etf_tracking": args.etf_tracking,
            "momentum_screening": {
                "enabled": bool(args.momentum_screening),
                "top_n": int(args.momentum_top_n) if args.momentum_screening else None,
                "lookback_days": float(args.momentum_lookback_days) if args.momentum_screening else None,
            },
            "learnable_inclusion": {
                "enabled": bool(args.learnable_inclusion),
                "target_k": int(args.inclusion_target_k) if args.learnable_inclusion else None,
                "init_gain": float(args.inclusion_init_gain) if args.learnable_inclusion else None,
                "lambda_cardinality": (
                    float(args.lambda_inclusion_cardinality) if args.learnable_inclusion else None
                ),
                "lambda_commitment": (
                    float(args.lambda_inclusion_commitment) if args.learnable_inclusion else None
                ),
            },
            "optuna_constrained_artifact_suffix": last_optuna_suffix,
            "week9_constrained_artifact_stem": week9_cstem,
            "git_commit_and_push_requested": bool(args.git_commit_and_push),
        },
    )
    total_sec = time.perf_counter() - run_started
    print(f"\n{'#'*60}", flush=True)
    print(f"  PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {total_sec / 60:.1f}m ({total_sec / 3600:.1f}h)", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"- run_manifest: {manifest_path}")

    if args.git_commit_and_push:
        tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
        commit_msg = args.git_commit_message or f"Week8 pipeline cloud run {tag}"
        _stage_banner("Git commit and push")
        _git_commit_and_push(
            project_root,
            remote=args.git_remote,
            branch=args.git_push_branch,
            message=commit_msg,
        )


if __name__ == "__main__":
    main()

