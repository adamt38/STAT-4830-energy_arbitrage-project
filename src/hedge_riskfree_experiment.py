"""Compare a near–risk-free bond proxy (SHY) to a hedged PM + equity/oil sleeve.

Both legs use the same Polymarket bar timestamps (UTC). SHY daily returns are
mapped with the same *lagged* convention as in ``stock_oil_hedge`` so the
experiment is apples-to-apples on timing.
"""

from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.stock_oil_hedge import (
    StockOilHedgeConfig,
    _lagged_daily_returns_for_timestamps,
    _utc_date_int,
    compute_stock_oil_hedge_series,
    fetch_stooq_daily_closes,
)


def total_capital_gain(r: np.ndarray) -> float:
    """Compound growth over the sample: prod(1+r) - 1."""
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


@dataclass(frozen=True)
class HedgeRiskFreeExperimentConfig:
    """Risk-free leg and IO paths."""

    rf_stooq: str = "shy.us"
    rf_annual_fallback: float = 0.04
    request_timeout_sec: float = 30.0


def _aligned_risk_free_returns(
    ts_ret: np.ndarray,
    cfg: HedgeRiskFreeExperimentConfig,
) -> tuple[np.ndarray, str]:
    min_d = _utc_date_int(int(ts_ret.min())) - 10_000
    max_d = _utc_date_int(int(ts_ret.max())) + 10
    try:
        d_rf, c_rf = fetch_stooq_daily_closes(
            cfg.rf_stooq,
            start_date_int=min_d,
            end_date_int=max_d,
            timeout_sec=cfg.request_timeout_sec,
        )
        if d_rf.size < 5:
            raise RuntimeError("Too few risk-free proxy rows from Stooq.")
        r_rf = _lagged_daily_returns_for_timestamps(ts_ret, d_rf, c_rf)
        if not np.all(np.isfinite(r_rf)):
            raise RuntimeError("Non-finite risk-free returns after alignment.")
        return r_rf, cfg.rf_stooq
    except (OSError, RuntimeError, ValueError):
        daily = (1.0 + cfg.rf_annual_fallback) ** (1.0 / 252.0) - 1.0
        return np.full(ts_ret.shape[0], daily, dtype=float), f"constant_{cfg.rf_annual_fallback:.4f}_annual"


def compute_hedge_vs_riskfree(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    hedge_config: StockOilHedgeConfig | None = None,
    rf_config: HedgeRiskFreeExperimentConfig | None = None,
) -> dict[str, Any]:
    """Return metrics comparing hedged PM book vs risk-free proxy (no file IO)."""
    hcfg = hedge_config or StockOilHedgeConfig()
    rcfg = rf_config or HedgeRiskFreeExperimentConfig()
    hedge = compute_stock_oil_hedge_series(project_root, artifact_prefix, hcfg)
    ts_ret = hedge["ts_ret"]
    r_rf, rf_label = _aligned_risk_free_returns(ts_ret, rcfg)
    cum_rf = np.cumprod(1.0 + r_rf)

    r_poly = hedge["r_poly"]
    r_combined = hedge["r_combined"]
    capital_gain_rf = total_capital_gain(r_rf)
    capital_gain_poly = total_capital_gain(r_poly)
    capital_gain_hedged = total_capital_gain(r_combined)

    hedge_metrics = dict(hedge["metrics"])

    comparison: dict[str, Any] = {
        "artifact_prefix": artifact_prefix,
        "risk_free_source": rf_label,
        "capital_gain_risk_free_proxy": capital_gain_rf,
        "capital_gain_polymarket_only": capital_gain_poly,
        "capital_gain_hedged_pm_stocks_oil": capital_gain_hedged,
        "excess_capital_gain_hedged_vs_rf": capital_gain_hedged - capital_gain_rf,
        "excess_capital_gain_poly_vs_rf": capital_gain_poly - capital_gain_rf,
        "n_periods": int(ts_ret.shape[0]),
        "hedge_config": {
            "spy_stooq": hcfg.spy_stooq,
            "oil_stooq": hcfg.oil_stooq,
            "hedge_allocation": hcfg.hedge_allocation,
            "hedge_mode": hcfg.hedge_mode,
        },
        "hedge_sleeve_metrics": hedge_metrics,
    }

    return {
        "comparison": comparison,
        "ts_ret": ts_ret,
        "r_risk_free": r_rf,
        "cum_risk_free": cum_rf,
        "hedge_series": hedge,
    }


def run_hedge_vs_riskfree_experiment(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    hedge_config: StockOilHedgeConfig | None = None,
    rf_config: HedgeRiskFreeExperimentConfig | None = None,
) -> dict[str, pathlib.Path]:
    """Write comparison JSON + aligned timeseries CSV under data/processed/."""
    result = compute_hedge_vs_riskfree(
        project_root, artifact_prefix, hedge_config, rf_config
    )
    comp = result["comparison"]
    ts_ret = result["ts_ret"]
    r_rf = result["r_risk_free"]
    cum_rf = result["cum_risk_free"]
    hedge = result["hedge_series"]

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{artifact_prefix}_hedge_vs_riskfree_comparison.json"
    json_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{artifact_prefix}_hedge_vs_riskfree_timeseries.csv"
    r_combined = hedge["r_combined"]
    r_poly = hedge["r_poly"]
    cum_poly = hedge["cum_poly"]
    cum_combo = hedge["cum_combo"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "risk_free_daily_return",
                "cumulative_risk_free",
                "polymarket_return",
                "cumulative_polymarket",
                "hedged_combined_return",
                "cumulative_hedged",
            ],
        )
        w.writeheader()
        for i in range(len(ts_ret)):
            w.writerow(
                {
                    "timestamp": int(ts_ret[i]),
                    "risk_free_daily_return": float(r_rf[i]),
                    "cumulative_risk_free": float(cum_rf[i]),
                    "polymarket_return": float(r_poly[i]),
                    "cumulative_polymarket": float(cum_poly[i]),
                    "hedged_combined_return": float(r_combined[i]),
                    "cumulative_hedged": float(cum_combo[i]),
                },
            )

    return {
        "hedge_vs_riskfree_comparison": json_path,
        "hedge_vs_riskfree_timeseries": csv_path,
    }
