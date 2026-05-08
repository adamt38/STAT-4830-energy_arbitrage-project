"""Post-hoc equity-domain tilt evaluator (Round 6 Pod S2).

Takes an already-run constrained pod's artifacts and applies the equity-domain
tilt overlay from ``src.pm_risk_overlay.build_equity_domain_tilt_multiplier``
at a sweep of ``tilt_strength`` values. For each strength, it produces a
tilted domain-equal portfolio (the multiplicatively re-weighted domain basket,
re-normalized) and measures Sortino / total return / max drawdown vs the pure
equal-weight baseline.

Why this evaluates the **tilt overlay in isolation** rather than tilting the
optimizer's best weights: the constrained pipeline exports ``{stem}_constrained
_best_timeseries.csv`` as a per-step scalar (``portfolio_return``), not the full
(T, N) weights matrix, so we can't multiply tilt(t, d) * weight(t, n) directly.
Instead we reconstruct per-asset returns from ``{prefix}_price_history.csv`` and
apply the tilt to a domain-equal allocation. If this beats the equal-weight
baseline by a meaningful Sortino delta, the tilt signal has value and becomes
a candidate for Round 7 integration inside the optimizer.

Usage:

    python script/posthoc_overlay_tilt.py \\
        --artifact-prefix week11_I \\
        --ticker-map data/external/domain_ticker_map_template.csv \\
        --tilt-strengths 0,5,10,20,33.3 \\
        --max-multiplier 2.0 \\
        --output-stem week11_I_equity_tilt

Requires ``yfinance`` (pulled lazily by ``build_equity_domain_tilt_multiplier``)
and network access for the daily-return fetch.
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
    build_equity_domain_tilt_multiplier,
    load_domain_ticker_tilt_map,
)

DEFAULT_STRENGTHS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 33.3)


def _sortino(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    dev = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(returns) / (dev + 1e-8))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / np.clip(peak, 1e-8, None) - 1.0))


def _total_return(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def _parse_strengths(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_STRENGTHS
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise SystemExit("--tilt-strengths requires at least one value.")
    try:
        values = tuple(float(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(f"--tilt-strengths must be floats, got {raw!r}: {exc}")
    if any(v < 0.0 for v in values):
        raise SystemExit(f"--tilt-strengths must be >= 0, got {values}")
    return values


def _load_markets_and_prices(
    markets_path: Path, prices_path: Path
) -> tuple[list[int], np.ndarray, list[str], list[str]]:
    """Return (timestamps, price_matrix, token_ids, domains) all aligned.

    Matches the conventions in ``src.baseline._build_price_matrix``: markets
    are keyed by ``yes_token_id``, history rows are keyed by ``token_id``.
    """
    markets_rows = _read_csv(markets_path)
    history_rows = _read_csv(prices_path)
    ts_values, matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    token_to_domain = {
        str(row.get("yes_token_id", "")).strip(): str(row.get("domain", "other")).strip().lower()
        or "other"
        for row in markets_rows
    }
    domains = [token_to_domain.get(t, "other") for t in kept_tokens]
    return ts_values, matrix, kept_tokens, domains


def _domain_equal_weights(domains: list[str]) -> np.ndarray:
    """Equal-weight per domain, each domain split equally across its members.

    Matches ``baseline._dynamic_portfolio_returns`` semantics for the rebalance-to-
    domain-equal baseline used in the pipeline.
    """
    uniq = sorted(set(domains))
    n_dom = len(uniq)
    n_ast = len(domains)
    w = np.zeros(n_ast, dtype=np.float64)
    per_dom = 1.0 / max(1, n_dom)
    counts = {d: domains.count(d) for d in uniq}
    for i, d in enumerate(domains):
        w[i] = per_dom / max(1, counts[d])
    return w


def _apply_tilt_to_weights(
    base_weights: np.ndarray,
    domains: list[str],
    tilt: np.ndarray,
    uniq_domains: list[str],
) -> np.ndarray:
    """tilted[t, n] = base[n] * tilt[t, domain_idx(n)]; renormalize per row."""
    T = tilt.shape[0]
    N = base_weights.shape[0]
    d_idx = {d: i for i, d in enumerate(uniq_domains)}
    out = np.empty((T, N), dtype=np.float64)
    for n in range(N):
        k = d_idx.get(domains[n], None)
        if k is None:
            out[:, n] = base_weights[n]
        else:
            out[:, n] = base_weights[n] * tilt[:, k]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 1e-12, 1.0, row_sum)
    return out / row_sum


def _returns_from_weights(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    """W shape (T, N), R shape (T, N). Per-step portfolio return = sum_n W[t-1, n] * R[t, n]."""
    if W.shape[0] != R.shape[0] or W.shape[1] != R.shape[1]:
        raise ValueError(f"shape mismatch: W={W.shape} R={R.shape}")
    lagged = np.vstack([W[0:1, :], W[:-1, :]])
    port = np.sum(lagged * R, axis=1)
    return port


def _metrics_row(strength: float, returns: np.ndarray) -> dict[str, float]:
    return {
        "tilt_strength": float(strength),
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
    path: Path, rows: Sequence[dict[str, float]], prefix: str, map_csv: Path, cov_rows: int
) -> None:
    lines = [
        f"# Post-hoc equity-domain tilt sweep: `{prefix}`",
        "",
        f"- **prices**: `{prefix}_price_history.csv` (rows: {cov_rows})",
        f"- **domain→ticker map**: `{map_csv}`",
        "",
        "Portfolio definition: `w_tilted(t, n) = base_domain_equal(n) * tilt(t, domain(n))`"
        " then renormalized. `tilt_strength = 0` reproduces the domain-equal baseline.",
        "",
        "| strength | Sortino | mean_ret | volatility | max_dd | total_ret |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tilt_strength']:.2f} "
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
        f"- **best Sortino** at strength = {best['tilt_strength']:.2f} "
        f"(Sortino = {best['sortino']:+.4f})",
        "",
        "If the Sortino argmax is strictly interior (strength > 0) and beats the strength=0 "
        "row, the equity-domain tilt signal carries real information and is worth adding to "
        "the optimizer objective in Round 7.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc equity-domain tilt sweep (Round 6 S2).")
    ap.add_argument("--artifact-prefix", required=True)
    ap.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )
    ap.add_argument(
        "--ticker-map",
        default=str(REPO_ROOT / "data" / "external" / "domain_ticker_map_template.csv"),
    )
    ap.add_argument("--tilt-strengths", default=None, metavar="S1,S2,...")
    ap.add_argument("--max-multiplier", type=float, default=2.0)
    ap.add_argument("--output-stem", default=None)
    args = ap.parse_args()

    strengths = _parse_strengths(args.tilt_strengths)
    max_mult = float(args.max_multiplier)
    processed = Path(args.processed_dir)
    map_path = Path(args.ticker_map)
    if not map_path.is_absolute():
        map_path = (REPO_ROOT / map_path).resolve()
    if not map_path.is_file():
        raise SystemExit(f"--ticker-map not found: {map_path}")

    domain_to_ticker = load_domain_ticker_tilt_map(map_path)
    if not domain_to_ticker:
        raise SystemExit(
            f"ticker map {map_path} is empty; populate domain,ticker rows before running."
        )

    markets_path = processed / f"{args.artifact_prefix}_markets_filtered.csv"
    prices_path = processed / f"{args.artifact_prefix}_price_history.csv"
    for p in (markets_path, prices_path):
        if not p.exists():
            raise SystemExit(f"missing file: {p}")

    timestamps, price_matrix, _kept_tokens, domains = _load_markets_and_prices(
        markets_path, prices_path
    )
    if price_matrix.size == 0:
        raise SystemExit(f"no price history loaded for prefix {args.artifact_prefix}")
    returns = _compute_returns(price_matrix)
    returns = np.nan_to_num(returns, nan=0.0)
    aligned_ts = np.asarray(timestamps[1:], dtype=np.int64)
    T = returns.shape[0]

    base_w = _domain_equal_weights(domains)
    uniq_domains = sorted(set(domains))

    tilt, _used_map = build_equity_domain_tilt_multiplier(
        pm_timestamps=aligned_ts,
        domains=domains,
        domain_to_ticker=domain_to_ticker,
        tilt_strength=1.0,
        max_multiplier=max_mult,
    )
    if tilt.shape[0] != T:
        raise SystemExit(f"tilt shape {tilt.shape} incompatible with returns {returns.shape}")

    rows: list[dict[str, float]] = []
    for strength in strengths:
        scaled_tilt = np.clip(
            1.0 + strength * np.clip(tilt - 1.0, 0.0, None), 1.0, max_mult
        )
        W = _apply_tilt_to_weights(base_w, domains, scaled_tilt, uniq_domains)
        port_ret = _returns_from_weights(W, returns)
        rows.append(_metrics_row(strength, port_ret))

    output_stem = args.output_stem or f"{args.artifact_prefix}_equity_tilt"
    out_csv = processed / f"{output_stem}.csv"
    out_md = processed / f"{output_stem}_summary.md"
    _write_csv(out_csv, rows)
    _write_markdown(out_md, rows, args.artifact_prefix, map_path, T)

    summary = {
        "artifact_prefix": args.artifact_prefix,
        "ticker_map": str(map_path),
        "tilt_strengths": list(strengths),
        "max_multiplier": max_mult,
        "n_steps": int(T),
        "n_domains_with_tickers": len(_used_map),
        "best_strength_by_sortino": max(rows, key=lambda r: r["sortino"])["tilt_strength"],
        "best_sortino": max(rows, key=lambda r: r["sortino"])["sortino"],
        "strength0_sortino": rows[0]["sortino"] if rows and rows[0]["tilt_strength"] == 0.0 else None,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
