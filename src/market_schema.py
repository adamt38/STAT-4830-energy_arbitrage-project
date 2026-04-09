"""Canonical schema helpers for multi-platform prediction markets."""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalMarketRow:
    """One market metadata row in a platform-agnostic format."""

    exchange: str
    exchange_event_id: str
    exchange_market_id: str
    exchange_symbol: str
    question: str
    domain: str
    end_time_utc: str
    liquidity: float
    fee_bps: float = 0.0
    event_title: str = ""
    event_slug: str = ""
    market_slug: str = ""
    series_ticker: str = ""


@dataclass(frozen=True)
class CanonicalPriceRow:
    """One binary-outcome price observation in canonical form."""

    exchange: str
    exchange_market_id: str
    exchange_symbol: str
    timestamp: int
    datetime_utc: str
    yes_price: float
    no_price: float
    bid_yes: float = 0.0
    ask_yes: float = 0.0


def write_canonical_markets_csv(path: pathlib.Path, rows: list[CanonicalMarketRow]) -> None:
    """Persist canonical market metadata rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exchange",
                "exchange_event_id",
                "exchange_market_id",
                "exchange_symbol",
                "question",
                "event_title",
                "event_slug",
                "market_slug",
                "series_ticker",
                "domain",
                "end_time_utc",
                "liquidity",
                "fee_bps",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_canonical_prices_csv(path: pathlib.Path, rows: list[CanonicalPriceRow]) -> None:
    """Persist canonical price-history rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exchange",
                "exchange_market_id",
                "exchange_symbol",
                "timestamp",
                "datetime_utc",
                "yes_price",
                "no_price",
                "bid_yes",
                "ask_yes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
