"""Refresh processed Polymarket CSVs + baseline series for an artifact prefix.

Re-runs ``build_dataset`` (markets + price history) and ``save_baseline_outputs``.
Does **not** rerun Optuna or wide sweep, so existing ``*_constrained_best_metrics.json``
and ``*_recent_wide_sweep_summary.json`` stay valid for ``run_seven_day_hedge_optimal.py``.

Use ``--force-fetch`` to bypass cached ``data/raw/{prefix}_events_raw.json`` and pull
fresh events from the API (requires network).

Example:
  python script/refresh_pm_data_prefix.py --prefix week11_v2 --force-fetch
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import run_equal_weight_baseline, save_baseline_outputs
from src.polymarket_data import BuildConfig, build_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh PM processed data + baseline for a prefix.")
    ap.add_argument("--prefix", default="week11_v2", help="Artifact prefix (e.g. week11_v2).")
    ap.add_argument(
        "--force-fetch",
        action="store_true",
        help="Ignore cached raw events JSON; refetch from API.",
    )
    args = ap.parse_args()
    pfx = str(args.prefix)

    cfg = BuildConfig(
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
        artifact_prefix=pfx,
        history_interval="max",
        history_fidelity=10,
        use_cached_events_if_available=not bool(args.force_fetch),
        history_priority_enabled=True,
        history_priority_oversample_factor=5,
    )
    paths = build_dataset(REPO_ROOT, config=cfg)
    for k, v in paths.items():
        print(f"build_dataset: {k}: {v}")

    br = run_equal_weight_baseline(REPO_ROOT, artifact_prefix=pfx)
    out = save_baseline_outputs(REPO_ROOT, br, artifact_prefix=pfx)
    for k, v in out.items():
        print(f"baseline: {k}: {v}")


if __name__ == "__main__":
    main()
