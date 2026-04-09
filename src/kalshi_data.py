"""Kalshi ingestion utilities for cross-platform arbitrage research.

Public market data uses the production Trade API v2 host documented at
https://docs.kalshi.com/getting_started/quick_start_market_data

This module:
- Tries the current public base URL first, then a legacy host as fallback.
- Parses GetMarketsResponse ``markets`` and FixedPointDollars price strings.
- On total failure, emits empty artifacts with diagnostics (no crash).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import requests

from src.event_alignment import _jaccard, _token_set
from src.market_schema import (
    CanonicalMarketRow,
    CanonicalPriceRow,
    write_canonical_markets_csv,
    write_canonical_prices_csv,
)

# Documented production server (see Kalshi OpenAPI / quick start).
KALSHI_API_BASE_PRIMARY = "https://api.elections.kalshi.com/trade-api/v2"

# Older hostname; some environments still resolve it but it may RST or differ.
KALSHI_API_BASE_LEGACY = "https://trading-api.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiBuildConfig:
    """Configuration for building Kalshi canonical artifacts."""

    artifact_prefix: str = "week8"
    max_markets: int = 200
    request_timeout_sec: int = 20
    sleep_between_requests_sec: float = 0.05
    # When True, scan GET /events (nested markets) and keep only events whose
    # title/sub_title overlaps Polymarket canonical text (see target_min_event_similarity).
    targeted_to_poly: bool = True
    target_min_event_similarity: float = 0.12
    max_event_pages: int = 25


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fixed_point_dollars_to_prob(value: Any) -> float | None:
    """Parse Kalshi FixedPointDollars string (e.g. '0.5600') to [0, 1]."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        x = float(s)
        return min(max(x, 0.0), 1.0)
    except ValueError:
        return None


def _yes_mid_from_market(market: dict[str, Any]) -> tuple[float, float, float]:
    """Return (yes_mid, bid_yes, ask_yes) in [0,1] from API market dict."""
    bid = _fixed_point_dollars_to_prob(market.get("yes_bid_dollars"))
    ask = _fixed_point_dollars_to_prob(market.get("yes_ask_dollars"))
    last = _fixed_point_dollars_to_prob(market.get("last_price_dollars"))

    # Legacy / alternate field names (older responses)
    if bid is None:
        bid = _fixed_point_dollars_to_prob(market.get("yes_bid"))
    if ask is None:
        ask = _fixed_point_dollars_to_prob(market.get("yes_ask"))
    if last is None:
        last = _fixed_point_dollars_to_prob(market.get("last_price"))

    bid_f = float(bid) if bid is not None else 0.0
    ask_f = float(ask) if ask is not None else 0.0
    if bid is not None and ask is not None and ask_f >= bid_f and (ask_f > 0 or bid_f > 0):
        mid = (bid_f + ask_f) / 2.0
    elif last is not None:
        mid = float(last)
        if bid_f == 0.0 and ask_f == 0.0:
            bid_f, ask_f = mid, mid
        else:
            bid_f = bid_f if bid_f > 0 else mid
            ask_f = ask_f if ask_f > 0 else mid
    elif bid is not None:
        mid = bid_f
        ask_f = ask_f if ask_f > 0 else mid
    elif ask is not None:
        mid = ask_f
        bid_f = bid_f if bid_f > 0 else mid
    else:
        mid = 0.0
    mid = min(max(mid, 0.0), 1.0)
    bid_f = min(max(bid_f, 0.0), 1.0)
    ask_f = min(max(ask_f, 0.0), 1.0)
    return mid, bid_f, ask_f


def _extract_markets(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Return (markets_list, cursor) from GetMarketsResponse or legacy shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], None
    if isinstance(payload, dict):
        cursor = payload.get("cursor")
        cursor_str = str(cursor) if cursor is not None else None
        for key in ("markets", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)], cursor_str
    return [], None


def _fetch_markets_page(
    base_url: str,
    *,
    limit: int,
    status: str,
    cursor: str | None,
    timeout: int,
) -> tuple[int, Any]:
    """GET /markets; return (http_status, json_or_none)."""
    params: dict[str, Any] = {"limit": limit, "status": status}
    if cursor:
        params["cursor"] = cursor
    url = f"{base_url.rstrip('/')}/markets"
    headers = {
        "User-Agent": "STAT4830-energy-arbitrage/1.0 (+research)",
        "Accept": "application/json",
    }
    response = requests.get(url, params=params, timeout=timeout, headers=headers)
    status_code = int(response.status_code)
    if status_code != 200:
        return status_code, None
    try:
        return status_code, response.json()
    except json.JSONDecodeError:
        return status_code, None


def _ingest_from_bases(cfg: KalshiBuildConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Try primary API base, then legacy. Follow cursor until max_markets or empty cursor.

    Returns (markets, meta) where meta includes diagnostics for data_quality.json.
    """
    limit_per_page = min(max(cfg.max_markets, 1), 1000)
    meta: dict[str, Any] = {
        "ingest_mode": "broad_markets",
        "bases_tried": [],
        "http_status": None,
        "api_base_used": None,
        "pages_fetched": 0,
        "cursor_final": None,
        "request_error": None,
    }
    all_markets: list[dict[str, Any]] = []

    for base in (KALSHI_API_BASE_PRIMARY, KALSHI_API_BASE_LEGACY):
        meta["bases_tried"].append(base)
        all_markets = []
        meta["pages_fetched"] = 0
        meta["cursor_final"] = None
        cursor: str | None = None
        try:
            while len(all_markets) < cfg.max_markets:
                need = min(limit_per_page, cfg.max_markets - len(all_markets))
                if need < 1:
                    break
                status_code, payload = _fetch_markets_page(
                    base,
                    limit=need,
                    status="open",
                    cursor=cursor,
                    timeout=cfg.request_timeout_sec,
                )
                meta["http_status"] = status_code
                if status_code != 200 or not isinstance(payload, dict):
                    raise RuntimeError(f"HTTP {status_code} from {base}/markets")
                page_markets, next_cursor = _extract_markets(payload)
                meta["pages_fetched"] += 1
                meta["cursor_final"] = next_cursor
                all_markets.extend(page_markets)
                if not page_markets:
                    break
                if not next_cursor:
                    break
                cursor = next_cursor
                time.sleep(cfg.sleep_between_requests_sec)

            meta["api_base_used"] = base
            meta["request_error"] = None
            return all_markets[: cfg.max_markets], meta
        except Exception as exc:
            meta["request_error"] = f"{type(exc).__name__}: {exc}"
            all_markets = []
            continue

    return [], meta


def _read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _kalshi_event_vs_poly_score(event: dict[str, Any], poly_texts: list[str]) -> float:
    kalshi_blob = f"{event.get('title', '')} {event.get('sub_title', '')}".strip()
    if not kalshi_blob or not poly_texts:
        return 0.0
    kalshi_tokens = _token_set(kalshi_blob)
    best = 0.0
    for pt in poly_texts:
        pt = pt.strip()
        if not pt:
            continue
        best = max(best, _jaccard(kalshi_tokens, _token_set(pt)))
    return best


def _fetch_events_page(
    base_url: str,
    *,
    limit: int,
    cursor: str | None,
    status: str,
    with_nested_markets: bool,
    timeout: int,
) -> tuple[int, Any]:
    """GET /events; return (http_status, json_or_none)."""
    url = f"{base_url.rstrip('/')}/events"
    params: dict[str, Any] = {
        "limit": min(max(limit, 1), 200),
        "status": status,
        "with_nested_markets": with_nested_markets,
    }
    if cursor:
        params["cursor"] = cursor
    headers = {
        "User-Agent": "STAT4830-energy-arbitrage/1.0 (+research)",
        "Accept": "application/json",
    }
    response = requests.get(url, params=params, timeout=timeout, headers=headers)
    status_code = int(response.status_code)
    if status_code != 200:
        return status_code, None
    try:
        return status_code, response.json()
    except json.JSONDecodeError:
        return status_code, None


def _ingest_targeted_to_poly(
    poly_rows: list[dict[str, str]],
    cfg: KalshiBuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """
    Paginate GET /events?with_nested_markets=true and keep markets only from events
    whose title/sub_title has sufficient token overlap with any Polymarket row.

    Returns (markets, meta, event_ticker_to_title).
    """
    poly_texts = [
        f"{r.get('event_title', '')} {r.get('question', '')}".strip() for r in poly_rows
    ]
    poly_texts = [t for t in poly_texts if t]
    if not poly_texts:
        return [], {"ingest_mode": "targeted_poly", "error": "empty_poly_texts"}, {}

    meta: dict[str, Any] = {
        "ingest_mode": "targeted_poly",
        "target_min_event_similarity": cfg.target_min_event_similarity,
        "max_event_pages": cfg.max_event_pages,
        "pages_scanned": 0,
        "events_seen": 0,
        "events_passed_filter": 0,
        "api_base_used": None,
        "request_error": None,
    }

    for base in (KALSHI_API_BASE_PRIMARY, KALSHI_API_BASE_LEGACY):
        markets_out: list[dict[str, Any]] = []
        event_titles: dict[str, str] = {}
        meta["pages_scanned"] = 0
        meta["events_seen"] = 0
        meta["events_passed_filter"] = 0
        cursor: str | None = None
        try:
            while meta["pages_scanned"] < cfg.max_event_pages and len(markets_out) < cfg.max_markets:
                status_code, payload = _fetch_events_page(
                    base,
                    limit=200,
                    cursor=cursor,
                    status="open",
                    with_nested_markets=True,
                    timeout=cfg.request_timeout_sec,
                )
                if status_code != 200 or not isinstance(payload, dict):
                    raise RuntimeError(f"GET /events HTTP {status_code}")
                events = payload.get("events") or []
                next_cursor = payload.get("cursor")
                meta["pages_scanned"] += 1
                meta["api_base_used"] = base
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    meta["events_seen"] += 1
                    score = _kalshi_event_vs_poly_score(event, poly_texts)
                    if score < cfg.target_min_event_similarity:
                        continue
                    meta["events_passed_filter"] += 1
                    et = str(event.get("event_ticker") or "").strip()
                    if et:
                        event_titles[et] = str(event.get("title") or "").strip()
                    for m in event.get("markets") or []:
                        if not isinstance(m, dict):
                            continue
                        if m.get("market_type") is not None and str(m.get("market_type")) != "binary":
                            continue
                        m2 = dict(m)
                        if et and not m2.get("event_ticker"):
                            m2["event_ticker"] = et
                        st = event.get("series_ticker")
                        if st is not None and not m2.get("series_ticker"):
                            m2["series_ticker"] = st
                        markets_out.append(m2)
                        if len(markets_out) >= cfg.max_markets:
                            break
                    if len(markets_out) >= cfg.max_markets:
                        break
                if len(markets_out) >= cfg.max_markets:
                    break
                if not events:
                    break
                if not next_cursor:
                    break
                cursor = str(next_cursor)
                time.sleep(cfg.sleep_between_requests_sec)

            if markets_out:
                return markets_out[: cfg.max_markets], meta, event_titles
        except Exception as exc:
            meta["request_error"] = f"{type(exc).__name__}: {exc}"
            continue

    return [], meta, {}


def _normalize_domain(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return "other"
    return value.replace(" ", "-")


def _fetch_event_payload(
    base_url: str,
    event_ticker: str,
    *,
    timeout: int,
) -> dict[str, Any] | None:
    """GET /events/{event_ticker}; return parsed JSON or None."""
    url = f"{base_url.rstrip('/')}/events/{event_ticker}"
    headers = {
        "User-Agent": "STAT4830-energy-arbitrage/1.0 (+research)",
        "Accept": "application/json",
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def _event_title_from_payload(payload: dict[str, Any]) -> str:
    """Extract human-readable title from GetEventResponse."""
    event_obj = payload.get("event")
    if isinstance(event_obj, dict):
        title = event_obj.get("title") or event_obj.get("title_text")
        if title:
            return str(title).strip()
    return ""


def fetch_kalshi_event_titles(
    base_url: str,
    markets: list[dict[str, Any]],
    *,
    timeout: int,
    sleep_sec: float,
    known_titles: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    For unique event_ticker values, GET /events/{ticker} and map ticker -> title.

    ``known_titles`` (e.g. from GET /events nested responses) avoids redundant fetches.

    Returns (ticker_to_title, stats).
    """
    known = {k: v for k, v in (known_titles or {}).items() if str(v).strip()}
    tickers: set[str] = set()
    for m in markets:
        et = str(m.get("event_ticker") or "").strip()
        if et:
            tickers.add(et)
    stats: dict[str, Any] = {
        "unique_event_tickers": len(tickers),
        "events_fetched_ok": 0,
        "events_failed": 0,
        "events_from_nested_cache": 0,
    }
    out: dict[str, str] = {}
    for ticker in sorted(tickers):
        if ticker in known:
            out[ticker] = known[ticker]
            stats["events_from_nested_cache"] += 1
    for ticker in sorted(tickers):
        if ticker in out:
            continue
        payload = _fetch_event_payload(base_url, ticker, timeout=timeout)
        time.sleep(sleep_sec)
        if not isinstance(payload, dict):
            stats["events_failed"] += 1
            continue
        title = _event_title_from_payload(payload)
        if title:
            out[ticker] = title
            stats["events_fetched_ok"] += 1
        else:
            stats["events_failed"] += 1
    return out, stats


def build_kalshi_dataset(
    project_root: pathlib.Path,
    config: KalshiBuildConfig | None = None,
) -> dict[str, pathlib.Path]:
    """Build Kalshi canonical artifacts.

    Outputs:
    - data/raw/{prefix}_kalshi_markets_raw.json
    - data/processed/{prefix}_kalshi_canonical_markets.csv
    - data/processed/{prefix}_kalshi_canonical_prices.csv
    - data/processed/{prefix}_kalshi_data_quality.json
    """
    cfg = config or KalshiBuildConfig()
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{cfg.artifact_prefix}_kalshi_markets_raw.json"
    markets_path = processed_dir / f"{cfg.artifact_prefix}_kalshi_canonical_markets.csv"
    prices_path = processed_dir / f"{cfg.artifact_prefix}_kalshi_canonical_prices.csv"
    quality_path = processed_dir / f"{cfg.artifact_prefix}_kalshi_data_quality.json"

    poly_path = processed_dir / f"{cfg.artifact_prefix}_poly_canonical_markets.csv"
    poly_rows = _read_csv_rows(poly_path)
    has_poly = bool(poly_rows) and any(
        (r.get("event_title") or r.get("question") or "").strip() for r in poly_rows
    )

    fetched: list[dict[str, Any]] = []
    fetch_meta: dict[str, Any] = {}
    known_event_titles: dict[str, str] = {}
    kalshi_ingest_mode = "broad"

    if cfg.targeted_to_poly and has_poly:
        fetched, fetch_meta, known_event_titles = _ingest_targeted_to_poly(poly_rows, cfg)
        if fetched:
            kalshi_ingest_mode = "targeted"

    if not fetched:
        prev_targeted_empty = cfg.targeted_to_poly and has_poly
        fetched, fetch_meta = _ingest_from_bases(cfg)
        kalshi_ingest_mode = "broad_fallback" if prev_targeted_empty else "broad"

    event_title_map: dict[str, str] = {}
    event_fetch_stats: dict[str, Any] = {}
    api_base = fetch_meta.get("api_base_used")
    if isinstance(api_base, str) and api_base and fetched:
        event_title_map, event_fetch_stats = fetch_kalshi_event_titles(
            api_base,
            fetched,
            timeout=cfg.request_timeout_sec,
            sleep_sec=cfg.sleep_between_requests_sec,
            known_titles=known_event_titles,
        )

    raw_path.write_text(
        json.dumps(
            {
                "fetch_meta": fetch_meta,
                "kalshi_ingest_mode": kalshi_ingest_mode,
                "event_fetch_stats": event_fetch_stats,
                "markets": fetched,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    market_rows: list[CanonicalMarketRow] = []
    price_rows: list[CanonicalPriceRow] = []
    now = dt.datetime.now(dt.timezone.utc)
    now_ts = int(now.timestamp())
    now_iso = now.isoformat()

    for market in fetched:
        # API: ticker is the market id; event_ticker groups markets.
        market_id = str(market.get("ticker") or market.get("id") or "")
        if not market_id:
            continue
        if market.get("market_type") is not None and str(market.get("market_type")) != "binary":
            continue

        event_id = str(market.get("event_ticker") or market.get("event_id") or "")
        ticker = market_id
        title = str(
            market.get("title")
            or market.get("yes_sub_title")
            or market.get("subtitle")
            or ticker
        )
        kalshi_event_title = event_title_map.get(event_id, "") if event_id else ""
        series_ticker = str(market.get("series_ticker") or "").strip()
        category = _normalize_domain(str(market.get("category") or market.get("event_category") or "other"))
        end_time = str(
            market.get("latest_expiration_time")
            or market.get("close_time")
            or market.get("expected_expiration_time")
            or market.get("expiration_time")
            or ""
        )
        vol = market.get("volume_fp") or market.get("volume") or "0"
        liquidity = _safe_float(vol, default=0.0)

        yes_mid, bid_yes, ask_yes = _yes_mid_from_market(market)
        no_mid = min(max(1.0 - yes_mid, 0.0), 1.0)

        market_rows.append(
            CanonicalMarketRow(
                exchange="kalshi",
                exchange_event_id=event_id,
                exchange_market_id=market_id,
                exchange_symbol=ticker,
                question=title,
                domain=category,
                end_time_utc=end_time,
                liquidity=liquidity,
                fee_bps=0.0,
                event_title=kalshi_event_title,
                series_ticker=series_ticker,
            )
        )
        price_rows.append(
            CanonicalPriceRow(
                exchange="kalshi",
                exchange_market_id=market_id,
                exchange_symbol=ticker,
                timestamp=now_ts,
                datetime_utc=now_iso,
                yes_price=yes_mid,
                no_price=no_mid,
                bid_yes=bid_yes,
                ask_yes=ask_yes,
            )
        )

    write_canonical_markets_csv(markets_path, market_rows)
    write_canonical_prices_csv(prices_path, price_rows)

    quality = {
        "exchange": "kalshi",
        "kalshi_ingest_mode": kalshi_ingest_mode,
        "kalshi_api_base_primary": KALSHI_API_BASE_PRIMARY,
        "kalshi_api_base_legacy": KALSHI_API_BASE_LEGACY,
        "api_base_used": fetch_meta.get("api_base_used"),
        "http_status": fetch_meta.get("http_status"),
        "pages_fetched": fetch_meta.get("pages_fetched")
        or fetch_meta.get("pages_scanned"),
        "cursor_final": fetch_meta.get("cursor_final"),
        "markets_fetched_raw": len(fetched),
        "markets_kept_canonical": len(market_rows),
        "price_rows_written": len(price_rows),
        "request_error": fetch_meta.get("request_error"),
        "bases_tried": fetch_meta.get("bases_tried"),
        "kalshi_event_fetch": event_fetch_stats,
        "fetch_meta": fetch_meta,
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    return {
        "kalshi_raw_markets": raw_path,
        "kalshi_canonical_markets": markets_path,
        "kalshi_canonical_prices": prices_path,
        "kalshi_data_quality": quality_path,
    }
