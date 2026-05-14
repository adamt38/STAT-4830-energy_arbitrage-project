"""Fee-aware majority-side strategy for Polymarket binary markets."""

from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.baseline import (
    BaselineResult,
    _build_price_matrix,
    _max_drawdown,
    _read_csv,
    _sortino_ratio,
    run_equal_weight_baseline,
)


@dataclass(frozen=True)
class MajorityStrategyResult:
    """Container for Majority strategy outputs."""

    timestamps: list[int]
    returns: np.ndarray
    cumulative: np.ndarray
    drawdown: np.ndarray
    sortino: float
    max_drawdown: float
    market_count: int
    gross_returns: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    turnover_l1: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    transaction_costs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    majority_yes_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    majority_no_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    fee_rate: float = 0.0
    threshold: float = 0.5


def _equal_domain_token_weights(
    markets_rows: list[dict[str, str]],
    kept_tokens: list[str],
) -> np.ndarray:
    """Return equal-domain, equal-within-domain token weights matching the baseline."""
    token_to_domain = {
        row["yes_token_id"]: row.get("domain", "other")
        for row in markets_rows
        if row.get("yes_token_id") in set(kept_tokens)
    }
    unique_domains = sorted({token_to_domain[token] for token in kept_tokens})
    domain_share = 1.0 / float(max(len(unique_domains), 1))
    domain_to_tokens: dict[str, list[str]] = {}
    for token in kept_tokens:
        domain_to_tokens.setdefault(token_to_domain[token], []).append(token)

    weights = np.zeros(len(kept_tokens), dtype=float)
    token_index = {token: idx for idx, token in enumerate(kept_tokens)}
    for tokens_in_domain in domain_to_tokens.values():
        if not tokens_in_domain:
            continue
        per_market_weight = domain_share / float(len(tokens_in_domain))
        for token in tokens_in_domain:
            weights[token_index[token]] = per_market_weight
    return weights


def _compute_majority_side_path(
    yes_price_matrix: np.ndarray,
    base_weights: np.ndarray,
    *,
    fee_rate: float = 0.0,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute returns for the side whose start-of-step probability is the majority.

    The strategy uses the current YES price as the implied probability. If
    ``YES > threshold`` it holds the YES side for that token during the next
    step; if ``NO = 1 - YES > threshold`` it holds the synthetic NO side.
    Transaction costs are charged on the full YES/NO position vector.
    """
    if yes_price_matrix.shape[0] < 2 or base_weights.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty

    yes_prev = yes_price_matrix[:-1]
    yes_next = yes_price_matrix[1:]
    no_prev = 1.0 - yes_prev
    no_next = 1.0 - yes_next
    yes_returns = (yes_next - yes_prev) / np.clip(yes_prev, 1e-6, None)
    no_returns = (no_next - no_prev) / np.clip(no_prev, 1e-6, None)

    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    yes_weight_totals: list[float] = []
    no_weight_totals: list[float] = []
    prev_side_weights = np.zeros(base_weights.size * 2, dtype=float)

    for t in range(yes_returns.shape[0]):
        current_yes = yes_prev[t]
        step_yes = yes_returns[t]
        step_no = no_returns[t]
        valid_yes = np.isfinite(current_yes) & np.isfinite(step_yes)
        valid_no = np.isfinite(current_yes) & np.isfinite(step_no)
        yes_majority = valid_yes & (current_yes > float(threshold))
        no_majority = valid_no & ((1.0 - current_yes) > float(threshold))

        yes_weights = np.where(yes_majority, base_weights, 0.0)
        no_weights = np.where(no_majority, base_weights, 0.0)
        weight_sum = float(np.sum(yes_weights) + np.sum(no_weights))
        if weight_sum <= 0.0:
            side_weights = np.zeros(base_weights.size * 2, dtype=float)
            gross_return = 0.0
        else:
            yes_weights = yes_weights / weight_sum
            no_weights = no_weights / weight_sum
            side_weights = np.concatenate([yes_weights, no_weights])
            gross_return = float(
                np.nan_to_num(step_yes, nan=0.0) @ yes_weights
                + np.nan_to_num(step_no, nan=0.0) @ no_weights
            )

        turnover = float(np.sum(np.abs(side_weights - prev_side_weights)))
        transaction_cost = float(fee_rate) * turnover
        gross_returns.append(gross_return)
        net_returns.append(gross_return - transaction_cost)
        turnovers.append(turnover)
        yes_weight_totals.append(float(np.sum(side_weights[: base_weights.size])))
        no_weight_totals.append(float(np.sum(side_weights[base_weights.size :])))
        prev_side_weights = side_weights

    return (
        np.array(net_returns, dtype=float),
        np.array(gross_returns, dtype=float),
        np.array(turnovers, dtype=float),
        np.array(yes_weight_totals, dtype=float),
        np.array(no_weight_totals, dtype=float),
    )


def run_majority_strategy(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    *,
    fee_rate: float = 0.0,
    threshold: float = 0.5,
) -> MajorityStrategyResult:
    """Run the fee-aware Majority strategy for a processed artifact prefix."""
    markets_path = project_root / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv"
    prices_path = project_root / "data" / "processed" / f"{artifact_prefix}_price_history.csv"
    markets_rows = _read_csv(markets_path)
    history_rows = _read_csv(prices_path)
    timestamps, yes_price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)

    if yes_price_matrix.size == 0 or len(kept_tokens) == 0:
        empty = np.array([], dtype=float)
        return MajorityStrategyResult(
            timestamps=[],
            returns=empty,
            cumulative=empty,
            drawdown=empty,
            sortino=0.0,
            max_drawdown=0.0,
            market_count=0,
            gross_returns=empty,
            turnover_l1=empty,
            transaction_costs=empty,
            majority_yes_weight=empty,
            majority_no_weight=empty,
            fee_rate=float(fee_rate),
            threshold=float(threshold),
        )

    base_weights = _equal_domain_token_weights(markets_rows, kept_tokens)
    returns, gross_returns, turnover_l1, yes_w, no_w = _compute_majority_side_path(
        yes_price_matrix,
        base_weights,
        fee_rate=fee_rate,
        threshold=threshold,
    )
    cumulative = np.cumprod(1.0 + returns)
    drawdown, max_dd = _max_drawdown(cumulative)
    return MajorityStrategyResult(
        timestamps=timestamps[1:],
        returns=returns,
        cumulative=cumulative,
        drawdown=drawdown,
        sortino=_sortino_ratio(returns),
        max_drawdown=max_dd,
        market_count=len(kept_tokens),
        gross_returns=gross_returns,
        turnover_l1=turnover_l1,
        transaction_costs=float(fee_rate) * turnover_l1,
        majority_yes_weight=yes_w,
        majority_no_weight=no_w,
        fee_rate=float(fee_rate),
        threshold=float(threshold),
    )


def _compound_gain(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0) if returns.size else 0.0


def save_majority_strategy_outputs(
    project_root: pathlib.Path,
    result: MajorityStrategyResult,
    baseline: BaselineResult,
    *,
    output_prefix: str,
    input_prefix: str,
    strategy_name: str = "Majority strategy",
    window_days: float | None = None,
    max_steps: int | None = None,
) -> dict[str, pathlib.Path]:
    """Persist Majority strategy time series, summary, and baseline comparison."""
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(result.timestamps), len(baseline.timestamps), result.returns.size, baseline.returns.size)
    timestamps_full = np.array(result.timestamps[-n:], dtype=np.int64)
    selected = np.arange(n, dtype=np.int64)
    if window_days is not None and n:
        start_ts = int(timestamps_full[-1] - float(window_days) * 86400.0)
        selected = selected[timestamps_full[selected] >= start_ts]
    if max_steps is not None and int(max_steps) > 0 and selected.size > int(max_steps):
        sampled = np.linspace(0, selected.size - 1, int(max_steps), dtype=np.int64)
        selected = selected[sampled]

    timestamps = timestamps_full[selected].tolist()
    strat_returns = result.returns[-n:][selected]
    base_returns = baseline.returns[-n:][selected]
    strat_cum = np.cumprod(1.0 + strat_returns)
    base_cum = np.cumprod(1.0 + base_returns)
    strat_window_drawdown, strat_window_max_dd = _max_drawdown(strat_cum)
    base_window_drawdown, base_window_max_dd = _max_drawdown(base_cum)

    series_path = out_dir / f"{output_prefix}_timeseries.csv"
    with series_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "majority_return",
                "majority_gross_return",
                "majority_turnover_l1",
                "majority_transaction_cost",
                "majority_cumulative_return",
                "majority_drawdown",
                "majority_yes_weight",
                "majority_no_weight",
                "baseline_return",
                "baseline_cumulative_return",
            ],
        )
        writer.writeheader()
        for i, ts in enumerate(timestamps):
            source_i = int(selected[i])
            writer.writerow(
                {
                    "timestamp": int(ts),
                    "majority_return": float(strat_returns[i]),
                    "majority_gross_return": float(result.gross_returns[-n:][source_i]),
                    "majority_turnover_l1": float(result.turnover_l1[-n:][source_i]),
                    "majority_transaction_cost": float(result.transaction_costs[-n:][source_i]),
                    "majority_cumulative_return": float(strat_cum[i]),
                    "majority_drawdown": float(strat_window_drawdown[i]),
                    "majority_yes_weight": float(result.majority_yes_weight[-n:][source_i]),
                    "majority_no_weight": float(result.majority_no_weight[-n:][source_i]),
                    "baseline_return": float(base_returns[i]),
                    "baseline_cumulative_return": float(base_cum[i]),
                }
            )

    majority_gain = _compound_gain(strat_returns)
    baseline_gain = _compound_gain(base_returns)
    selected_yes_weight = result.majority_yes_weight[-n:][selected]
    selected_no_weight = result.majority_no_weight[-n:][selected]
    selected_invested_weight = selected_yes_weight + selected_no_weight
    summary: dict[str, Any] = {
        "strategy": strategy_name,
        "description": (
            "Equal-domain allocation that holds only sides above the configured implied-probability "
            "threshold at the start of each step: YES if YES > threshold, synthetic NO if "
            "1 - YES > threshold. If no side qualifies, the portfolio stays in cash for that step."
        ),
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "threshold": result.threshold,
        "fee_rate": result.fee_rate,
        "market_count": result.market_count,
        "window": {
            "days_requested": float(window_days) if window_days is not None else None,
            "raw_aligned_steps": int(n),
            "sampled_steps": int(selected.size),
            "first_row_timestamp": int(timestamps[0]) if timestamps else None,
            "last_row_timestamp": int(timestamps[-1]) if timestamps else None,
        },
        "n_aligned_steps": int(selected.size),
        "majority_gain": majority_gain,
        "baseline_gain": baseline_gain,
        "excess_gain_vs_baseline": majority_gain - baseline_gain,
        "growth_of_1_majority": float(strat_cum[-1]) if strat_cum.size else 1.0,
        "growth_of_1_baseline": float(base_cum[-1]) if base_cum.size else 1.0,
        "sortino_majority": _sortino_ratio(strat_returns),
        "sortino_baseline": _sortino_ratio(base_returns),
        "max_drawdown_majority": strat_window_max_dd,
        "max_drawdown_baseline": base_window_max_dd,
        "mean_majority_return": float(np.mean(strat_returns)) if strat_returns.size else 0.0,
        "mean_baseline_return": float(np.mean(base_returns)) if base_returns.size else 0.0,
        "volatility_majority": float(np.std(strat_returns)) if strat_returns.size else 0.0,
        "volatility_baseline": float(np.std(base_returns)) if base_returns.size else 0.0,
        "avg_majority_yes_weight": (
            float(np.mean(selected_yes_weight)) if selected.size else 0.0
        ),
        "avg_majority_no_weight": (
            float(np.mean(selected_no_weight)) if selected.size else 0.0
        ),
        "avg_invested_weight": (
            float(np.mean(selected_invested_weight)) if selected.size else 0.0
        ),
        "active_step_fraction": (
            float(np.mean(selected_invested_weight > 0.0)) if selected.size else 0.0
        ),
        "total_turnover_l1": float(np.sum(result.turnover_l1[-n:][selected])),
        "avg_turnover_l1": (
            float(np.mean(result.turnover_l1[-n:][selected])) if selected.size else 0.0
        ),
        "total_transaction_cost": float(np.sum(result.transaction_costs[-n:][selected])),
    }
    summary_path = out_dir / f"{output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "majority_timeseries": series_path,
        "majority_summary": summary_path,
    }


def run_majority_vs_baseline(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    *,
    output_prefix: str | None = None,
    strategy_name: str = "Majority strategy",
    fee_rate: float = 0.0,
    threshold: float = 0.5,
    window_days: float | None = None,
    max_steps: int | None = None,
) -> tuple[MajorityStrategyResult, BaselineResult, dict[str, pathlib.Path]]:
    """Run Majority strategy and an aligned fee-aware baseline, then save comparison outputs."""
    out_prefix = output_prefix or f"{artifact_prefix}_majority_strategy"
    majority = run_majority_strategy(
        project_root,
        artifact_prefix,
        fee_rate=fee_rate,
        threshold=threshold,
    )
    baseline = run_equal_weight_baseline(
        project_root,
        artifact_prefix,
        fee_rate=fee_rate,
    )
    paths = save_majority_strategy_outputs(
        project_root,
        majority,
        baseline,
        output_prefix=out_prefix,
        input_prefix=artifact_prefix,
        strategy_name=strategy_name,
        window_days=window_days,
        max_steps=max_steps,
    )
    return majority, baseline, paths
