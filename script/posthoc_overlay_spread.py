"""Post-hoc PM-internal negative-correlation spread evaluator (Round 6 Pod S3).

Takes a finished constrained pod's `{prefix}_category_correlation.csv` and
`{prefix}_price_history.csv`, identifies the top-N negatively correlated domain
pairs via ``src.pm_risk_overlay.top_negative_correlation_pairs``, and computes
the zero-investment spread return series (``pm_category_spread_returns``).

We then evaluate the blended series

    r_combined(t; lam) = r_baseline(t) + lam * r_spread(t)

at a grid of ``spread_lambda`` values (dollar exposure to the spread on top of
the equal-weight domain-baseline PM allocation). The purpose is to quantify
whether zero-investment PM-internal pairs trading on negatively correlated
categories would have improved the baseline's Sortino / drawdown profile. If
it does, it becomes a Round 7 candidate to fold into the optimizer objective.

Usage:

    python script/posthoc_overlay_spread.py \\
        --artifact-prefix week11_I \\
        --max-pairs 0,2,5,10 \\
        --spread-lambdas 0.0,0.1,0.25,0.5 \\
        --corr-threshold -0.002 \\
        --output-stem week11_I_pm_spread

No network access required; reads only existing pipeline artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from src.baseline import _build_price_matrix, _compute_returns, _read_csv
from src.pm_risk_overlay import (
    pm_category_spread_returns,
    read_category_correlation_csv,
    top_negative_correlation_pairs,
)

DEFAULT_MAX_PAIRS: tuple[int, ...] = (0, 2, 5, 10)
DEFAULT_SPREAD_LAMBDAS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)


def _sortino(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    d = np.minimum(r, 0.0)
    dev = float(np.sqrt(np.mean(np.square(d))))
    return float(np.mean(r) / (dev + 1e-8))


def _max_drawdown(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / np.clip(peak, 1e-8, None) - 1.0))


def _total_return(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def _parse_int_list(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        vals = tuple(int(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(f"must be comma-separated ints, got {raw!r}: {exc}")
    if any(v < 0 for v in vals):
        raise SystemExit(f"ints must be >= 0, got {vals}")
    return vals or default


def _parse_float_list(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        return default
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        vals = tuple(float(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(f"must be comma-separated floats, got {raw!r}: {exc}")
    return vals or default


def _load_returns_and_domains(
    markets_path: Path, prices_path: Path
) -> tuple[np.ndarray, list[str]]:
    markets_rows = _read_csv(markets_path)
    history_rows = _read_csv(prices_path)
    _ts, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    if price_matrix.size == 0:
        raise SystemExit("no price history loaded")
    returns = _compute_returns(price_matrix)
    returns = np.nan_to_num(returns, nan=0.0)
    token_to_domain = {
        str(row.get("yes_token_id", "")).strip(): str(row.get("domain", "other")).strip().lower()
        or "other"
        for row in markets_rows
    }
    domains = [token_to_domain.get(t, "other") for t in kept_tokens]
    return returns, domains


def _baseline_domain_equal_returns(returns: np.ndarray, domains: list[str]) -> np.ndarray:
    """Equal-weight per domain then equal-weight within domain, rebalanced each step."""
    uniq = sorted(set(domains))
    n_dom = len(uniq)
    by_dom: dict[str, list[int]] = {d: [] for d in uniq}
    for j, d in enumerate(domains):
        by_dom[d].append(j)
    T, N = returns.shape
    base_w = np.zeros(N, dtype=np.float64)
    per_dom = 1.0 / max(1, n_dom)
    for d, ixs in by_dom.items():
        if ixs:
            share = per_dom / len(ixs)
            for j in ixs:
                base_w[j] = share
    return returns @ base_w


def _metrics_row(
    max_pairs: int, spread_lambda: float, returns: np.ndarray
) -> dict[str, float]:
    return {
        "max_pairs": int(max_pairs),
        "spread_lambda": float(spread_lambda),
        "n_steps": int(returns.size),
        "mean_return": float(np.mean(returns)) if returns.size else 0.0,
        "volatility": float(np.std(returns)) if returns.size else 0.0,
        "sortino": _sortino(returns),
        "max_drawdown": _max_drawdown(returns),
        "total_return": _total_return(returns),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(
    path: Path,
    rows: Sequence[dict[str, float]],
    prefix: str,
    pairs: list[tuple[str, str, float]],
    n_steps: int,
) -> None:
    lines = [
        f"# Post-hoc PM-category spread sweep: `{prefix}`",
        "",
        f"- **returns/domains**: `{prefix}_price_history.csv` + `{prefix}_markets_filtered.csv` (steps: {n_steps})",
        f"- **correlation matrix**: `{prefix}_category_correlation.csv`",
        f"- **candidate negative-correlation pairs (top-{len(pairs)}):**",
    ]
    for a, b, c in pairs:
        lines.append(f"  - `{a}` ↔ `{b}`  (ρ = {c:+.4f})")
    lines += [
        "",
        "Combined return: `r(t; max_pairs, lambda) = r_baseline(t) + lambda * r_spread(t)`"
        " where `r_spread` is the mean of `r(domain_A) - r(domain_B)` across the selected pairs.",
        "",
        "| max_pairs | spread_lambda | Sortino | mean_ret | volatility | max_dd | total_ret |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['max_pairs']:d} "
            f"| {r['spread_lambda']:.2f} "
            f"| {r['sortino']:+.4f} "
            f"| {r['mean_return']:+.6f} "
            f"| {r['volatility']:.6f} "
            f"| {r['max_drawdown']:+.4%} "
            f"| {r['total_return']:+.4%} |"
        )
    best = max(rows, key=lambda r: r["sortino"])
    lines += [
        "",
        "## Argmax",
        "",
        f"- **best Sortino** at max_pairs={best['max_pairs']}, lambda={best['spread_lambda']:.2f} "
        f"(Sortino = {best['sortino']:+.4f})",
        "",
        "If the Sortino argmax is strictly interior (spread_lambda > 0) and beats the "
        "spread_lambda=0 row (pure baseline), zero-investment PM-internal pairs trading "
        "on negatively correlated categories adds risk-adjusted value and is worth folding "
        "into Round 7.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc PM-category spread sweep (Round 6 S3).")
    ap.add_argument("--artifact-prefix", required=True)
    ap.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )
    ap.add_argument("--max-pairs", default=None, metavar="N1,N2,...")
    ap.add_argument("--spread-lambdas", default=None, metavar="L1,L2,...")
    ap.add_argument("--corr-threshold", type=float, default=-0.002)
    ap.add_argument("--output-stem", default=None)
    args = ap.parse_args()

    max_pairs_list = _parse_int_list(args.max_pairs, DEFAULT_MAX_PAIRS)
    lambdas = _parse_float_list(args.spread_lambdas, DEFAULT_SPREAD_LAMBDAS)
    threshold = float(args.corr_threshold)
    processed = Path(args.processed_dir)

    markets_path = processed / f"{args.artifact_prefix}_markets_filtered.csv"
    prices_path = processed / f"{args.artifact_prefix}_price_history.csv"
    corr_path = processed / f"{args.artifact_prefix}_category_correlation.csv"
    for p in (markets_path, prices_path, corr_path):
        if not p.exists():
            raise SystemExit(f"missing file: {p}")

    returns, domains = _load_returns_and_domains(markets_path, prices_path)
    baseline_ret = _baseline_domain_equal_returns(returns, domains)
    labels, corr = read_category_correlation_csv(corr_path)

    max_N = max(max_pairs_list) if max_pairs_list else 0
    all_pairs = top_negative_correlation_pairs(
        labels, corr, max_pairs=max(max_N, 1), max_correlation=threshold
    )

    rows: list[dict[str, float]] = []
    for N in max_pairs_list:
        pairs = all_pairs[:N]
        spread = pm_category_spread_returns(returns, domains, pairs)
        for lam in lambdas:
            if N == 0 and lam != 0.0:
                continue  # trivial row: no spread applied when lambda=0 or N=0
            combined = baseline_ret + lam * spread
            rows.append(_metrics_row(N, lam, combined))

    output_stem = args.output_stem or f"{args.artifact_prefix}_pm_spread"
    out_csv = processed / f"{output_stem}.csv"
    out_md = processed / f"{output_stem}_summary.md"
    _write_csv(out_csv, rows)
    _write_markdown(out_md, rows, args.artifact_prefix, all_pairs[:max_N], baseline_ret.size)

    summary = {
        "artifact_prefix": args.artifact_prefix,
        "n_steps": int(baseline_ret.size),
        "corr_threshold": threshold,
        "max_pairs_grid": list(max_pairs_list),
        "spread_lambdas_grid": list(lambdas),
        "candidate_pairs": [
            {"a": a, "b": b, "corr": float(c)} for (a, b, c) in all_pairs[:max_N]
        ],
        "best_row": max(rows, key=lambda r: r["sortino"]),
        "baseline_sortino": _sortino(baseline_ret),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
