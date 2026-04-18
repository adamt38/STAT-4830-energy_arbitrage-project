"""Unified pipeline with a mode toggle:

- polymarket_only: existing single-platform optimization workflow
- cross_platform: polymarket + kalshi + alignment + arbitrage signals
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
import time
from dataclasses import replace
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arbitrage_signals import generate_arbitrage_candidates
from src.baseline import (
    _build_price_matrix,
    _read_csv,
    pretty_float,
    pretty_pct,
    run_equal_weight_baseline,
    save_baseline_outputs,
)
from src.constrained_optimizer import ExperimentConfig, run_optuna_search, split_index_for_returns
from src.exogenous_features import build_and_save_exogenous
from src.covariance_diagnostics import run_covariance_diagnostics
from src.event_alignment import align_polymarket_kalshi_events
from src.kalshi_data import KalshiBuildConfig, build_kalshi_dataset
from src.market_schema import (
    CanonicalMarketRow,
    CanonicalPriceRow,
    write_canonical_markets_csv,
    write_canonical_prices_csv,
)
from src.polymarket_data import BuildConfig, NoMarketsAfterHistoryFilterError, build_dataset


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_run_manifest(
    *,
    artifact_prefix: str,
    mode: str,
    stage_durations_sec: dict[str, float],
    artifacts: dict[str, str],
) -> pathlib.Path:
    processed = REPO_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / f"{artifact_prefix}_{mode}_run_manifest.json"
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_prefix": artifact_prefix,
        "mode": mode,
        "stage_durations_sec": stage_durations_sec,
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _polymarket_input_prefix(
    processed: pathlib.Path,
    artifact_prefix: str,
    override: str | None,
) -> str:
    """Prefix for *_markets_filtered.csv / *_price_history.csv (may differ from artifact_prefix)."""
    if override:
        return override
    if (processed / f"{artifact_prefix}_markets_filtered.csv").exists():
        return artifact_prefix
    if artifact_prefix.endswith("_cross"):
        base = artifact_prefix[: -len("_cross")]
        if base and (processed / f"{base}_markets_filtered.csv").exists():
            return base
    return artifact_prefix


def _canonicalize_polymarket_outputs(
    *,
    artifact_prefix: str,
    polymarket_input_prefix: str | None = None,
) -> dict[str, pathlib.Path]:
    """Convert existing Polymarket processed artifacts into canonical schema."""
    processed = REPO_ROOT / "data" / "processed"
    in_prefix = _polymarket_input_prefix(processed, artifact_prefix, polymarket_input_prefix)
    if in_prefix != artifact_prefix:
        print(
            f"Polymarket source CSVs use prefix '{in_prefix}'; "
            f"writing '{artifact_prefix}_poly_canonical_*.csv'."
        )
    markets_path = processed / f"{in_prefix}_markets_filtered.csv"
    prices_path = processed / f"{in_prefix}_price_history.csv"
    poly_markets_out = processed / f"{artifact_prefix}_poly_canonical_markets.csv"
    poly_prices_out = processed / f"{artifact_prefix}_poly_canonical_prices.csv"

    markets_rows = _read_csv(markets_path) if markets_path.exists() else []
    price_rows = _read_csv(prices_path) if prices_path.exists() else []
    token_to_market: dict[str, dict[str, str]] = {}
    canonical_markets: list[CanonicalMarketRow] = []
    for row in markets_rows:
        token = str(row.get("yes_token_id", ""))
        token_to_market[token] = row
        canonical_markets.append(
            CanonicalMarketRow(
                exchange="polymarket",
                exchange_event_id=str(row.get("event_id", "")),
                exchange_market_id=str(row.get("market_id", "")),
                exchange_symbol=token,
                question=str(row.get("question", "")),
                domain=str(row.get("domain", "other")),
                end_time_utc=str(row.get("event_end", "")),
                liquidity=float(row.get("market_liquidity", 0.0) or 0.0),
                fee_bps=0.0,
                event_title=str(row.get("event_title", "") or ""),
                event_slug=str(row.get("event_slug", "") or ""),
                market_slug=str(row.get("market_slug", "") or ""),
            )
        )

    canonical_prices: list[CanonicalPriceRow] = []
    for row in price_rows:
        token = str(row.get("token_id", ""))
        meta = token_to_market.get(token)
        if meta is None:
            continue
        yes_price = float(row.get("price", 0.0) or 0.0)
        canonical_prices.append(
            CanonicalPriceRow(
                exchange="polymarket",
                exchange_market_id=str(meta.get("market_id", "")),
                exchange_symbol=token,
                timestamp=int(row.get("timestamp", 0) or 0),
                datetime_utc=str(row.get("datetime_utc", "")),
                yes_price=yes_price,
                no_price=max(0.0, 1.0 - yes_price),
                bid_yes=yes_price,
                ask_yes=yes_price,
            )
        )

    write_canonical_markets_csv(poly_markets_out, canonical_markets)
    write_canonical_prices_csv(poly_prices_out, canonical_prices)
    return {
        "poly_canonical_markets": poly_markets_out,
        "poly_canonical_prices": poly_prices_out,
    }


def _run_polymarket_optimization(
    artifact_prefix: str,
    optuna_trials: int,
    *,
    macro_integration: str = "rescale",
    joint_macro_mode_search: bool = False,
    etf_tracking: bool = False,
    optuna_artifact_suffix: str | None = None,
    optuna_n_jobs: int = 1,
) -> dict[str, pathlib.Path]:
    """Run the existing single-platform optimization flow."""
    build_cfg = BuildConfig(
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
        artifact_prefix=artifact_prefix,
        history_interval="max",
        history_fidelity=60,
        use_cached_events_if_available=True,
        history_priority_enabled=True,
        history_priority_oversample_factor=5,
    )
    if build_cfg.history_fidelity <= 10:
        walkforward_train_steps = 1440
        walkforward_test_steps = 288
    else:
        walkforward_train_steps = 240
        walkforward_test_steps = 48

    experiment_cfg = ExperimentConfig(
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
        weight_parameterization="projected_simplex",
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
        optuna_n_jobs=optuna_n_jobs,
    )
    if etf_tracking:
        experiment_cfg = replace(experiment_cfg, use_etf_tracking=True)

    try:
        data_artifacts = build_dataset(project_root=REPO_ROOT, config=build_cfg)
    except NoMarketsAfterHistoryFilterError:
        raise RuntimeError(
            "No markets survived Polymarket filters. Try lowering min_history_days or increasing max_closed_events."
        )

    baseline_result = run_equal_weight_baseline(REPO_ROOT, artifact_prefix=artifact_prefix)
    baseline_artifacts = save_baseline_outputs(REPO_ROOT, baseline_result, artifact_prefix=artifact_prefix)
    markets_p = REPO_ROOT / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv"
    history_p = REPO_ROOT / "data" / "processed" / f"{artifact_prefix}_price_history.csv"
    baseline_ts_p = REPO_ROOT / "data" / "processed" / f"{artifact_prefix}_baseline_timeseries.csv"
    if markets_p.exists() and history_p.exists() and baseline_ts_p.exists():
        ts_values, _, _ = _build_price_matrix(_read_csv(markets_p), _read_csv(history_p))
        if len(ts_values) >= 2:
            step_ts = ts_values[1:]
            with baseline_ts_p.open("r", newline="", encoding="utf-8") as handle:
                baseline_ts_list = [int(row["timestamp"]) for row in csv.DictReader(handle)]
            if baseline_ts_list == step_ts:
                split_idx_exog = split_index_for_returns(len(step_ts), experiment_cfg)
                build_and_save_exogenous(
                    REPO_ROOT,
                    artifact_prefix,
                    step_ts,
                    split_idx_exog,
                    baseline_timestamps=baseline_ts_list,
                )
    optuna_cfg = (
        experiment_cfg
        if joint_macro_mode_search
        else replace(experiment_cfg, macro_integration=macro_integration)
    )
    suf = optuna_artifact_suffix
    if suf is None:
        suf = (
            ""
            if (joint_macro_mode_search or macro_integration == "rescale")
            else f"_macro_{macro_integration}"
        )
    constrained_artifacts = run_optuna_search(
        REPO_ROOT,
        artifact_prefix=artifact_prefix,
        config=optuna_cfg,
        n_trials=optuna_trials,
        joint_macro_mode_search=joint_macro_mode_search,
        output_artifact_suffix=suf,
    )
    covariance_artifacts = run_covariance_diagnostics(REPO_ROOT, artifact_prefix=artifact_prefix)

    print(f"Baseline markets: {baseline_result.market_count}")
    print(f"Baseline sortino: {pretty_float(baseline_result.sortino)}")
    print(f"Baseline max_drawdown: {pretty_pct(baseline_result.max_drawdown)}")

    combined: dict[str, pathlib.Path] = {}
    combined.update(data_artifacts)
    combined.update(baseline_artifacts)
    combined.update(constrained_artifacts)
    combined.update(covariance_artifacts)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Polymarket/Kalshi pipeline with mode toggle.")
    parser.add_argument(
        "--mode",
        choices=("polymarket_only", "cross_platform"),
        default="polymarket_only",
        help="polymarket_only runs only the existing optimization workflow. "
        "cross_platform additionally runs Kalshi ingest, alignment, and arbitrage signal generation.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Optional artifact prefix. Default uses mode-specific prefixes.",
    )
    parser.add_argument(
        "--polymarket-input-prefix",
        default=None,
        help="Read markets_filtered.csv and price_history.csv under this prefix; "
        "canonical poly outputs still use --artifact-prefix. "
        "If omitted and those files are missing, a trailing '_cross' on the artifact prefix is stripped "
        "once (e.g. week8_cross -> week8) when the base prefix has data.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for optimization stage.",
    )
    parser.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna trials (study.optimize n_jobs). Default 1; use -1 for auto or 16–32 on large pods "
        "with OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1.",
    )
    parser.add_argument(
        "--macro-integration",
        choices=("rescale", "explicit", "both"),
        default="rescale",
        help="Macro objective path for constrained Optuna (ignored if --joint-macro-mode-search).",
    )
    parser.add_argument(
        "--joint-macro-mode-search",
        action="store_true",
        help="Single Optuna study with categorical macro_mode.",
    )
    parser.add_argument(
        "--etf-tracking",
        action="store_true",
        help="Enable ETF blend tracking penalty in constrained Optuna (see polymarket_week8_pipeline --etf-tracking).",
    )
    parser.add_argument(
        "--optuna-artifact-suffix",
        default=None,
        help="Optional suffix for constrained Optuna output filenames (default by macro mode).",
    )
    parser.add_argument(
        "--min-alignment-similarity",
        type=float,
        default=0.25,
        help="Minimum text similarity for polymarket/kalshi market matching (Jaccard on tokens). "
        "With typical cross-platform wording, scores above ~0.55 are rare; try 0.22–0.30.",
    )
    parser.add_argument(
        "--min-event-cluster-similarity",
        type=float,
        default=0.12,
        help="Min score to lock a Polymarket event cluster to one Kalshi event before market-level matching.",
    )
    parser.add_argument(
        "--alignment-max-end-days-gap",
        type=float,
        default=-1.0,
        help="When both sides have parseable end_time_utc, skip pairs farther apart than this many days. "
        "Default -1 disables (cross-platform resolution dates often differ by years). "
        "Use e.g. 30 for same-resolution-window matching.",
    )
    parser.add_argument(
        "--min-net-edge",
        type=float,
        default=0.01,
        help="Minimum post-cost arbitrage edge to keep candidate.",
    )
    parser.add_argument(
        "--slippage-buffer",
        type=float,
        default=0.01,
        help="Estimated round-trip slippage/spread cost (probability points).",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip polymarket optimization stages and only run cross-platform ingest/alignment/arbitrage.",
    )
    parser.add_argument(
        "--kalshi-broad-only",
        action="store_true",
        help="Skip Polymarket-targeted Kalshi scan; use open /markets pagination only.",
    )
    parser.add_argument(
        "--kalshi-target-min-similarity",
        type=float,
        default=0.12,
        help="Min Jaccard overlap between Kalshi event title/sub_title and Poly text for targeted ingest.",
    )
    parser.add_argument(
        "--kalshi-max-event-pages",
        type=int,
        default=25,
        help="Max GET /events pages when building targeted Kalshi universe.",
    )
    parser.add_argument(
        "--no-alignment-strict-same-event",
        dest="alignment_strict_same_event",
        action="store_false",
        help="Allow cross-event matches: unmapped Poly events search all Kalshi; global fallback when a mapped "
        "event's markets score below threshold.",
    )
    parser.set_defaults(alignment_strict_same_event=True)
    args = parser.parse_args()

    run_started = time.perf_counter()
    mode = args.mode
    artifact_prefix = args.artifact_prefix or ("week8_poly" if mode == "polymarket_only" else "week8_cross")

    print(f"Running mode: {mode}")
    print(f"Artifact prefix: {artifact_prefix}")
    stage_durations: dict[str, float] = {}
    artifact_map: dict[str, str] = {}

    if mode == "polymarket_only" and args.skip_optimization:
        raise RuntimeError("--skip-optimization can only be used with --mode cross_platform.")

    if not args.skip_optimization:
        stage_start = time.perf_counter()
        poly_artifacts = _run_polymarket_optimization(
            artifact_prefix=artifact_prefix,
            optuna_trials=args.optuna_trials,
            macro_integration=args.macro_integration,
            joint_macro_mode_search=args.joint_macro_mode_search,
            etf_tracking=args.etf_tracking,
            optuna_artifact_suffix=args.optuna_artifact_suffix,
            optuna_n_jobs=args.optuna_n_jobs,
        )
        stage_durations["polymarket_optimization"] = time.perf_counter() - stage_start
        artifact_map.update({k: str(v) for k, v in poly_artifacts.items()})
    else:
        print("Skipping polymarket optimization stages; using existing processed artifacts.")

    stage_start = time.perf_counter()
    poly_canonical = _canonicalize_polymarket_outputs(
        artifact_prefix=artifact_prefix,
        polymarket_input_prefix=args.polymarket_input_prefix,
    )
    stage_durations["polymarket_canonicalization"] = time.perf_counter() - stage_start
    artifact_map.update({k: str(v) for k, v in poly_canonical.items()})

    if mode == "cross_platform":
        stage_start = time.perf_counter()
        kalshi_artifacts = build_kalshi_dataset(
            REPO_ROOT,
            config=KalshiBuildConfig(
                artifact_prefix=artifact_prefix,
                targeted_to_poly=not args.kalshi_broad_only,
                target_min_event_similarity=args.kalshi_target_min_similarity,
                max_event_pages=args.kalshi_max_event_pages,
            ),
        )
        stage_durations["kalshi_ingestion"] = time.perf_counter() - stage_start
        artifact_map.update({k: str(v) for k, v in kalshi_artifacts.items()})

        stage_start = time.perf_counter()
        end_gap = (
            None if args.alignment_max_end_days_gap < 0 else args.alignment_max_end_days_gap
        )
        alignment_artifacts = align_polymarket_kalshi_events(
            REPO_ROOT,
            artifact_prefix=artifact_prefix,
            min_similarity=args.min_alignment_similarity,
            min_event_cluster_similarity=args.min_event_cluster_similarity,
            max_end_days_gap=end_gap,
            strict_same_event=args.alignment_strict_same_event,
        )
        stage_durations["event_alignment"] = time.perf_counter() - stage_start
        artifact_map.update({k: str(v) for k, v in alignment_artifacts.items()})

        stage_start = time.perf_counter()
        arbitrage_artifacts = generate_arbitrage_candidates(
            REPO_ROOT,
            artifact_prefix=artifact_prefix,
            slippage_buffer=args.slippage_buffer,
            min_net_edge=args.min_net_edge,
        )
        stage_durations["arbitrage_signal_generation"] = time.perf_counter() - stage_start
        artifact_map.update({k: str(v) for k, v in arbitrage_artifacts.items()})

    stage_durations["total"] = time.perf_counter() - run_started
    manifest = _write_run_manifest(
        artifact_prefix=artifact_prefix,
        mode=mode,
        stage_durations_sec=stage_durations,
        artifacts=artifact_map,
    )
    print(f"Run complete in {stage_durations['total'] / 60:.1f}m")
    print(f"Run manifest: {manifest}")


if __name__ == "__main__":
    main()
