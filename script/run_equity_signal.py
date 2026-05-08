"""Build aligned PM-domain vs equity-sector signals and pair diagnostics."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.equity_signal import EquitySignalConfig, build_asset_equity_signal_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PM-domain/equity-sector pair features.")
    parser.add_argument("--artifact-prefix", default="week8")
    parser.add_argument("--mapping-csv", default="data/external/domain_ticker_map_template.csv")
    args = parser.parse_args()

    mapping_path = pathlib.Path(args.mapping_csv)
    if not mapping_path.is_absolute():
        mapping_path = (REPO_ROOT / mapping_path).resolve()

    markets_rows = []
    import csv

    with (REPO_ROOT / "data" / "processed" / f"{args.artifact_prefix}_markets_filtered.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        markets_rows = list(csv.DictReader(handle))
    asset_domains = [row.get("domain", "other") for row in markets_rows]

    _, artifacts = build_asset_equity_signal_matrix(
        project_root=REPO_ROOT,
        artifact_prefix=args.artifact_prefix,
        asset_domains=asset_domains,
        config=EquitySignalConfig(mapping_csv=mapping_path),
    )
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
