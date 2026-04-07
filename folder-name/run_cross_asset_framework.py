"""Run cross-asset framework ideas for PM + equities.

Example:
  python script/run_cross_asset_framework.py --artifact-prefix week8
  python script/run_cross_asset_framework.py --options-prior-csv data/external/options_prior_template.csv
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cross_asset_framework import CrossAssetConfig, run_cross_asset_framework


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-asset PM framework with stock signals.")
    parser.add_argument("--artifact-prefix", default="week8")
    parser.add_argument("--topic-mapping-csv", default=None)
    parser.add_argument("--options-prior-csv", default=None)
    parser.add_argument("--pm-notional", type=float, default=10_000.0)
    parser.add_argument("--divergence-threshold", type=float, default=0.10)
    parser.add_argument("--momentum-window", type=int, default=96)
    parser.add_argument("--momentum-lag", type=int, default=1)
    parser.add_argument("--regime-low-vix", type=float, default=18.0)
    parser.add_argument("--regime-high-vix", type=float, default=25.0)
    parser.add_argument("--vix-symbol", default="vix.us")
    args = parser.parse_args()

    cfg = CrossAssetConfig(
        options_prior_csv=pathlib.Path(args.options_prior_csv).resolve() if args.options_prior_csv else None,
        topic_mapping_csv=pathlib.Path(args.topic_mapping_csv).resolve() if args.topic_mapping_csv else None,
        pm_notional=args.pm_notional,
        divergence_threshold=args.divergence_threshold,
        momentum_window=args.momentum_window,
        momentum_lag=args.momentum_lag,
        regime_low_vix=args.regime_low_vix,
        regime_high_vix=args.regime_high_vix,
        vix_symbol=args.vix_symbol,
    )
    outputs = run_cross_asset_framework(REPO_ROOT, artifact_prefix=args.artifact_prefix, config=cfg)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
