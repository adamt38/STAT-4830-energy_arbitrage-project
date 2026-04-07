"""Run Polymarket + oil / inverse-equity hedge sleeve experiment.

Example:
  python script/run_stock_oil_hedge.py --artifact-prefix week8
  python script/run_stock_oil_hedge.py --hedge-mode rolling_mv --hedge-allocation 0.25
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import cast

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stock_oil_hedge import HedgeMode, StockOilHedgeConfig, run_stock_oil_hedge_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock/oil hedge layer for Polymarket baseline.")
    parser.add_argument("--artifact-prefix", default="week8", help="Prefix of processed artifacts (default week8).")
    parser.add_argument("--hedge-allocation", type=float, default=0.2, help="Weight on hedge sleeve [0,1].")
    parser.add_argument("--oil-weight", type=float, default=0.5, help="Relative weight on oil leg vs short equity.")
    parser.add_argument("--inverse-equity-weight", type=float, default=0.5, help="Relative weight on -SPY leg.")
    parser.add_argument(
        "--hedge-mode",
        choices=("static", "rolling_mv"),
        default="static",
        help="static = fixed recipe; rolling_mv = scale hedge vs Polymarket for min variance.",
    )
    parser.add_argument("--rolling-window", type=int, default=48, help="Bars for rolling MV (Polymarket steps).")
    parser.add_argument(
        "--analyst-csv",
        default=None,
        help="Optional CSV with date_utc and equity_hedge_tilt columns (see data/external/).",
    )
    args = parser.parse_args()
    analyst_path = pathlib.Path(args.analyst_csv).resolve() if args.analyst_csv else None
    cfg = StockOilHedgeConfig(
        hedge_allocation=args.hedge_allocation,
        oil_leg_weight=args.oil_weight,
        inverse_equity_leg_weight=args.inverse_equity_weight,
        hedge_mode=cast(HedgeMode, args.hedge_mode),
        rolling_window=args.rolling_window,
        analyst_csv=analyst_path,
    )
    paths = run_stock_oil_hedge_experiment(REPO_ROOT, artifact_prefix=args.artifact_prefix, config=cfg)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
