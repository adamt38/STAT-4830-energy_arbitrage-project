"""Arbitrage signal generation from aligned cross-platform markets."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_arbitrage_candidates(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    fee_bps_poly: float = 0.0,
    fee_bps_kalshi: float = 0.0,
    slippage_buffer: float = 0.01,
    min_net_edge: float = 0.01,
) -> dict[str, pathlib.Path]:
    """Create ranked cross-exchange arbitrage candidate artifacts."""
    processed = project_root / "data" / "processed"
    aligned_path = processed / f"{artifact_prefix}_aligned_market_pairs.csv"
    poly_prices_path = processed / f"{artifact_prefix}_poly_canonical_prices.csv"
    kalshi_prices_path = processed / f"{artifact_prefix}_kalshi_canonical_prices.csv"

    aligned_rows = _read_csv(aligned_path) if aligned_path.exists() else []
    poly_prices = _read_csv(poly_prices_path) if poly_prices_path.exists() else []
    kalshi_prices = _read_csv(kalshi_prices_path) if kalshi_prices_path.exists() else []

    poly_by_market = {row.get("exchange_market_id", ""): row for row in poly_prices}
    kalshi_by_market = {row.get("exchange_market_id", ""): row for row in kalshi_prices}

    results: list[dict[str, Any]] = []
    for row in aligned_rows:
        poly_market = str(row.get("poly_market_id", ""))
        kalshi_market = str(row.get("kalshi_market_id", ""))
        poly_quote = poly_by_market.get(poly_market)
        kalshi_quote = kalshi_by_market.get(kalshi_market)
        if poly_quote is None or kalshi_quote is None:
            continue

        p_poly = _safe_float(poly_quote.get("yes_price"), default=0.0)
        p_kalshi = _safe_float(kalshi_quote.get("yes_price"), default=0.0)
        raw_spread = abs(p_poly - p_kalshi)
        fee_cost = (fee_bps_poly + fee_bps_kalshi) / 10000.0
        total_cost = fee_cost + slippage_buffer
        net_edge = raw_spread - total_cost
        if net_edge < min_net_edge:
            continue

        cheap_exchange = "polymarket" if p_poly < p_kalshi else "kalshi"
        rich_exchange = "kalshi" if cheap_exchange == "polymarket" else "polymarket"
        strategy_note = (
            "buy cheap YES and hedge rich side via opposite exposure "
            "(sell YES or buy NO where executable)"
        )
        results.append(
            {
                "canonical_event_key": row.get("canonical_event_key", ""),
                "poly_market_id": poly_market,
                "kalshi_market_id": kalshi_market,
                "poly_yes_price": p_poly,
                "kalshi_yes_price": p_kalshi,
                "raw_spread": raw_spread,
                "estimated_total_cost": total_cost,
                "net_edge": net_edge,
                "cheap_exchange": cheap_exchange,
                "rich_exchange": rich_exchange,
                "strategy_note": strategy_note,
                "alignment_score": _safe_float(row.get("score"), default=0.0),
            }
        )

    results_sorted = sorted(results, key=lambda item: float(item["net_edge"]), reverse=True)

    candidates_path = processed / f"{artifact_prefix}_arbitrage_candidates.csv"
    _write_csv(
        candidates_path,
        rows=results_sorted,
        fieldnames=[
            "canonical_event_key",
            "poly_market_id",
            "kalshi_market_id",
            "poly_yes_price",
            "kalshi_yes_price",
            "raw_spread",
            "estimated_total_cost",
            "net_edge",
            "cheap_exchange",
            "rich_exchange",
            "strategy_note",
            "alignment_score",
        ],
    )

    summary = {
        "artifact_prefix": artifact_prefix,
        "aligned_pairs_considered": len(aligned_rows),
        "candidates_found": len(results_sorted),
        "min_net_edge": min_net_edge,
        "slippage_buffer": slippage_buffer,
        "fee_bps_poly": fee_bps_poly,
        "fee_bps_kalshi": fee_bps_kalshi,
        "best_candidate": results_sorted[0] if results_sorted else {},
    }
    summary_path = processed / f"{artifact_prefix}_arbitrage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "arbitrage_candidates": candidates_path,
        "arbitrage_summary": summary_path,
    }
