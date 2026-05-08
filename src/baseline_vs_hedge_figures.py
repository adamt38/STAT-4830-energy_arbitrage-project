"""Helpers for aligning baseline vs hedge time series (figure pipeline support)."""

from __future__ import annotations

import pathlib


def align_baseline_to_hedge_rows(
    baseline: list[dict[str, str]],
    hedge: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep only baseline rows whose timestamps appear in hedge, ordered like hedge."""
    hedge_ts = [row["timestamp"] for row in hedge]
    b_map = {row["timestamp"]: row for row in baseline}
    baseline_out = []
    hedge_out = []
    for hrow in hedge:
        ts = hrow["timestamp"]
        if ts in b_map:
            baseline_out.append(b_map[ts])
            hedge_out.append(hrow)
    return baseline_out, hedge_out


def make_baseline_vs_hedge_figures(project_root: pathlib.Path, artifact_prefix: str) -> None:
    """Raise FileNotFoundError if expected processed artifacts for the prefix are missing."""
    base = project_root / "data" / "processed"
    for suffix in ("_baseline_timeseries.csv", "_constrained_best_timeseries.csv"):
        path = base / f"{artifact_prefix}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(str(path))
