"""Baseline portfolio metrics for Polymarket cross-domain allocation."""

from __future__ import annotations

import csv
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    """Container for baseline outputs."""

    timestamps: list[int]
    returns: np.ndarray
    cumulative: np.ndarray
    drawdown: np.ndarray
    sortino: float
    max_drawdown: float
    exposure_by_domain: dict[str, float]
    market_count: int


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _build_price_matrix(
    markets_rows: list[dict[str, str]],
    history_rows: list[dict[str, str]],
) -> tuple[list[int], np.ndarray, list[str]]:
    token_to_domain = {row["yes_token_id"]: row["domain"] for row in markets_rows}
    tokens = sorted(token_to_domain.keys())
    ts_values = sorted({int(row["timestamp"]) for row in history_rows})
    ts_index = {ts: idx for idx, ts in enumerate(ts_values)}
    token_index = {token: idx for idx, token in enumerate(tokens)}

    matrix = np.full((len(ts_values), len(tokens)), np.nan, dtype=float)
    for row in history_rows:
        token = row["token_id"]
        if token not in token_index:
            continue
        i = ts_index[int(row["timestamp"])]
        j = token_index[token]
        matrix[i, j] = float(row["price"])

    # Forward-fill only after each token's first observation.
    # Keep pre-listing points as NaN so assets can enter over time.
    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        valid = np.where(~np.isnan(col_vals))[0]
        if len(valid) == 0:
            continue
        first = valid[0]
        for idx in range(first + 1, len(col_vals)):
            if np.isnan(col_vals[idx]):
                col_vals[idx] = col_vals[idx - 1]
        matrix[:, col] = col_vals

    if not np.any(~np.isnan(matrix)):
        return [], np.array([], dtype=float), []

    # Keep tokens with at least one valid observation; dynamic-universe logic
    # downstream will renormalize weights among currently available assets.
    keep_cols = np.any(~np.isnan(matrix), axis=0)
    kept_tokens = [token for token, keep in zip(tokens, keep_cols, strict=False) if keep]
    filtered = matrix[:, keep_cols]
    return ts_values, filtered, kept_tokens


def _compute_returns(price_matrix: np.ndarray) -> np.ndarray:
    if price_matrix.shape[0] < 2:
        return np.array([], dtype=float)
    prev = price_matrix[:-1]
    nxt = price_matrix[1:]
    returns = (nxt - prev) / np.clip(prev, 1e-6, None)
    return returns


def _dynamic_portfolio_returns(returns_matrix: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    """Compute returns with time-varying available asset set."""
    if returns_matrix.size == 0 or base_weights.size == 0:
        return np.array([], dtype=float)
    out: list[float] = []
    for t in range(returns_matrix.shape[0]):
        step = returns_matrix[t]
        valid = np.isfinite(step)
        if not np.any(valid):
            out.append(0.0)
            continue
        masked_weights = np.where(valid, base_weights, 0.0)
        weight_sum = float(np.sum(masked_weights))
        if weight_sum <= 0.0:
            masked_weights = np.where(valid, 1.0, 0.0)
            weight_sum = float(np.sum(masked_weights))
        step_weights = masked_weights / max(weight_sum, 1e-12)
        step_returns = np.nan_to_num(step, nan=0.0)
        out.append(float(step_returns @ step_weights))
    return np.array(out, dtype=float)


def _sortino_ratio(returns: np.ndarray, eps: float = 1e-8) -> float:
    if returns.size == 0:
        return 0.0
    mean_return = float(np.mean(returns))
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
    if downside_std < eps:
        return 0.0
    return mean_return / downside_std


def _max_drawdown(cumulative: np.ndarray) -> tuple[np.ndarray, float]:
    if cumulative.size == 0:
        return np.array([], dtype=float), 0.0
    running_peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / np.clip(running_peak, 1e-8, None) - 1.0
    return drawdown, float(np.min(drawdown))


def run_equal_weight_baseline(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
) -> BaselineResult:
    """Run equal-weight baseline and return metrics."""
    markets_path = project_root / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv"
    prices_path = project_root / "data" / "processed" / f"{artifact_prefix}_price_history.csv"

    markets_rows = _read_csv(markets_path)
    history_rows = _read_csv(prices_path)
    timestamps, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)

    if returns_matrix.size == 0 or len(kept_tokens) == 0:
        empty = np.array([], dtype=float)
        return BaselineResult(
            timestamps=[],
            returns=empty,
            cumulative=empty,
            drawdown=empty,
            sortino=0.0,
            max_drawdown=0.0,
            exposure_by_domain={},
            market_count=0,
        )

    # Equal category weight baseline: each category gets 1/K,
    # and markets inside category split that category weight equally.
    token_to_domain = {
        row["yes_token_id"]: row["domain"] for row in markets_rows if row["yes_token_id"] in set(kept_tokens)
    }
    unique_domains = sorted({token_to_domain[token] for token in kept_tokens})
    domain_count = max(len(unique_domains), 1)
    domain_share = 1.0 / float(domain_count)
    domain_to_tokens: dict[str, list[str]] = {}
    for token in kept_tokens:
        domain_to_tokens.setdefault(token_to_domain[token], []).append(token)

    weights = np.zeros(len(kept_tokens), dtype=float)
    token_index = {token: idx for idx, token in enumerate(kept_tokens)}
    for domain, tokens_in_domain in domain_to_tokens.items():
        if not tokens_in_domain:
            continue
        per_market_weight = domain_share / float(len(tokens_in_domain))
        for token in tokens_in_domain:
            weights[token_index[token]] = per_market_weight
    portfolio_returns = _dynamic_portfolio_returns(returns_matrix, weights)
    cumulative = np.cumprod(1.0 + portfolio_returns)
    drawdown, max_dd = _max_drawdown(cumulative)

    domain_counts: dict[str, int] = {}
    for token in kept_tokens:
        domain = token_to_domain.get(token, "other")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    exposure = {domain: domain_share for domain in domain_counts}

    return BaselineResult(
        timestamps=timestamps[1:],
        returns=portfolio_returns,
        cumulative=cumulative,
        drawdown=drawdown,
        sortino=_sortino_ratio(portfolio_returns),
        max_drawdown=max_dd,
        exposure_by_domain=exposure,
        market_count=len(kept_tokens),
    )


def save_baseline_outputs(
    project_root: pathlib.Path,
    result: BaselineResult,
    artifact_prefix: str = "week8",
) -> dict[str, pathlib.Path]:
    """Persist baseline metrics and time-series outputs."""
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = out_dir / f"{artifact_prefix}_baseline_timeseries.csv"
    with series_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "portfolio_return", "cumulative_return", "drawdown"],
        )
        writer.writeheader()
        for ts, ret, cum, dd in zip(
            result.timestamps,
            result.returns.tolist(),
            result.cumulative.tolist(),
            result.drawdown.tolist(),
            strict=False,
        ):
            writer.writerow(
                {
                    "timestamp": int(ts),
                    "portfolio_return": float(ret),
                    "cumulative_return": float(cum),
                    "drawdown": float(dd),
                }
            )

    metrics = {
        "strategy": "equal_weight_baseline",
        "sortino_ratio": result.sortino,
        "max_drawdown": result.max_drawdown,
        "market_count": result.market_count,
        "exposure_by_domain": result.exposure_by_domain,
        "mean_return": float(np.mean(result.returns)) if result.returns.size else 0.0,
        "volatility": float(np.std(result.returns)) if result.returns.size else 0.0,
    }
    metrics_path = out_dir / f"{artifact_prefix}_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "baseline_timeseries": series_path,
        "baseline_metrics": metrics_path,
    }


def pretty_pct(x: float) -> str:
    """Format metric as percent string."""
    return f"{x * 100.0:.2f}%"


def pretty_float(x: float) -> str:
    """Format metric as fixed-point string."""
    if math.isfinite(x):
        return f"{x:.4f}"
    return "nan"

