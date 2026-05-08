"""Faster Optuna search with ETF risk-regime concentration (for large PM panels).

Uses the same regime + risk_score + dynamic concentration merge as
``polymarket_week8_pipeline.py``, but coarser walk-forward (fewer folds) so
``n_trials`` completes in reasonable time on ~60k-step histories.

Example:
  python script/run_risk_regime_optuna_beat_baseline.py \\
    --input-prefix week11_v2 --output-prefix week16_rr_fast --trials 20
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import run_equal_weight_baseline, save_baseline_outputs
from src.constrained_optimizer import ExperimentConfig, run_optuna_search
from src.equity_signal import (
    EquitySignalConfig,
    REGIME_PENALTY_SCALES,
    compute_risk_regime_zscore,
    get_dynamic_concentration_params,
    get_regime,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk-regime Optuna with coarser walk-forward.")
    ap.add_argument("--input-prefix", default="week11_v2", help="Existing processed artifact prefix.")
    ap.add_argument("--output-prefix", default="week16_rr_fast", help="Prefix for outputs (overwrites).")
    ap.add_argument("--trials", type=int, default=20, help="Optuna trials.")
    ap.add_argument("--skip-baseline", action="store_true", help="Skip baseline recompute if already present.")
    args = ap.parse_args()
    src = args.input_prefix
    dst = args.output_prefix
    processed = REPO_ROOT / "data" / "processed"
    for suffix in (
        "_markets_filtered.csv",
        "_price_history.csv",
    ):
        src_p = processed / f"{src}{suffix}"
        dst_p = processed / f"{dst}{suffix}"
        if not src_p.is_file():
            raise SystemExit(f"Missing {src_p}")
        dst_p.write_bytes(src_p.read_bytes())

    regime, vix_value = get_regime(EquitySignalConfig())
    regime_scales = REGIME_PENALTY_SCALES[regime]
    risk_score = compute_risk_regime_zscore()
    dyn = get_dynamic_concentration_params(risk_score)
    base_cfg = ExperimentConfig(
        learning_rates=(0.005, 0.01, 0.02, 0.05, 0.1),
        penalties_lambda=(0.25, 0.5, 1.0, 2.0),
        rolling_windows=(24, 48, 72, 96),
        steps_per_window=3,
        objective="excess_mean_downside",
        variance_penalty=float(regime_scales["variance_penalty"]),
        downside_penalty=float(regime_scales["downside_penalty"]),
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
        domain_limits=(0.08, 0.12, 0.18, 0.25),
        max_weights=(0.04, 0.06, 0.10, 0.15),
        concentration_penalty_lambdas=(2.0, 5.0, 10.0, 20.0, 50.0),
        covariance_penalty_lambdas=(0.5, 1.0, 5.0, 10.0),
        covariance_shrinkages=(0.02, 0.05, 0.10),
        entropy_lambdas=(0.0, 0.01, 0.02),
        uniform_mixes=(0.0, 0.05, 0.1, 0.2),
        max_domain_exposure_threshold=float(regime_scales["max_domain_exposure_threshold"]),
        holdout_fraction=0.2,
        walkforward_train_steps=480,
        walkforward_test_steps=2880,
        seed=7,
    )
    merged_conc = tuple(
        sorted(set(base_cfg.concentration_penalty_lambdas) | {dyn["concentration_penalty_lambda"]})
    )
    cfg = ExperimentConfig(
        **{
            **base_cfg.__dict__,
            "max_domain_exposure_threshold": float(dyn["max_domain_exposure_threshold"]),
            "concentration_penalty_lambdas": merged_conc,
        }
    )
    summary = {
        "input_prefix": src,
        "output_prefix": dst,
        "regime": regime.value,
        "vix_value": vix_value,
        "risk_regime_zscore": risk_score,
        "dynamic_concentration": dyn,
        "walkforward_train_steps": cfg.walkforward_train_steps,
        "walkforward_test_steps": cfg.walkforward_test_steps,
    }
    (processed / f"{dst}_risk_regime_optuna_config.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))

    if not args.skip_baseline:
        br = run_equal_weight_baseline(REPO_ROOT, artifact_prefix=dst)
        save_baseline_outputs(REPO_ROOT, br, artifact_prefix=dst)
        print("baseline_sortino", br.sortino)

    paths = run_optuna_search(
        REPO_ROOT,
        config=cfg,
        artifact_prefix=dst,
        n_trials=max(1, int(args.trials)),
    )
    print("artifacts", {k: str(v) for k, v in paths.items()})

    bm = json.loads((processed / f"{dst}_baseline_metrics.json").read_text(encoding="utf-8"))
    cm_path = processed / f"{dst}_constrained_best_metrics.json"
    if cm_path.is_file():
        cm = json.loads(cm_path.read_text(encoding="utf-8"))
        bp = cm.get("best_params") or {}
        h_sort = float(bp.get("holdout_sortino_ratio", 0.0) or 0.0)
        b_sort = float(bm.get("sortino_ratio", 0.0) or 0.0)
        print("\n--- vs baseline (holdout Sortino) ---")
        print("baseline_sortino", b_sort)
        print("constrained_holdout_sortino", h_sort)
        print("delta_sortino", h_sort - b_sort)
        print("baseline_max_dd", bm.get("max_drawdown"))
        print("constrained_holdout_max_dd", bp.get("holdout_max_drawdown"))


if __name__ == "__main__":
    main()
