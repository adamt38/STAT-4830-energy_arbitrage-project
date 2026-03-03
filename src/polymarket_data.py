"""Data ingestion and preprocessing utilities for Polymarket experiments."""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DEFAULT_CATEGORY_LIMIT = 16


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for building a local Polymarket analysis dataset."""

    max_events: int = 60
    events_page_limit: int = 50
    min_event_markets: int = 1
    min_history_points: int = 20
    min_history_days: float = 14.0
    max_markets: int = 24
    max_categories: int = DEFAULT_CATEGORY_LIMIT
    per_category_market_cap: int = 2
    min_category_liquidity: float = 0.0
    excluded_category_slugs: tuple[str, ...] = (
        "hide-from-new",
        "parent-for-derivative",
        "earn-4",
        "pre-market",
        "rewards-20-4pt5-50",
    )
    artifact_prefix: str = "week8"
    history_interval: str = "max"
    history_fidelity: int = 60
    request_timeout_sec: int = 20
    sleep_between_requests_sec: float = 0.05


def _request_json(
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int = 3,
) -> Any:
    """Fetch JSON with small retry logic for network resilience."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(0.3 * attempt)
    if last_error is None:
        raise RuntimeError("Unexpected request failure without exception")
    raise RuntimeError(f"Request failed for {url} with params={params}") from last_error


def fetch_active_events(config: BuildConfig) -> list[dict[str, Any]]:
    """Fetch paginated active events with nested markets and tags."""
    events: list[dict[str, Any]] = []
    offset = 0

    while len(events) < config.max_events:
        page = _request_json(
            f"{GAMMA_BASE_URL}/events",
            {
                "active": "true",
                "closed": "false",
                "limit": config.events_page_limit,
                "offset": offset,
            },
            timeout=config.request_timeout_sec,
        )
        if not isinstance(page, list) or not page:
            break

        events.extend(page)
        offset += config.events_page_limit
        time.sleep(config.sleep_between_requests_sec)

    return events[: config.max_events]


def _parse_json_list_field(raw_value: Any) -> list[str]:
    """Parse list-like fields that arrive as JSON strings."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return []


def _derive_fallback_domain_from_tags(event_tags: list[dict[str, Any]]) -> str:
    """Map tags to fallback coarse domain when no specific category is found."""
    values: list[str] = []
    for tag in event_tags:
        label = str(tag.get("label", "")).lower()
        slug = str(tag.get("slug", "")).lower()
        values.extend([label, slug])

    rules = {
        "politics": ["politic", "election", "government", "senate", "president"],
        "crypto": ["crypto", "bitcoin", "ethereum", "solana", "defi"],
        "sports": ["sports", "nba", "nfl", "mlb", "nhl", "soccer", "tennis"],
        "finance": ["finance", "economy", "stocks", "business", "fed", "rates"],
        "culture": ["celebrity", "movie", "music", "tv", "social-media", "media"],
        "science": ["science", "tech", "ai", "space", "health"],
    }
    for domain, needles in rules.items():
        if any(any(needle in value for needle in needles) for value in values):
            return domain
    return "other"


def _derive_category_from_tags(event_tags: list[dict[str, Any]]) -> str:
    """Return a specific market category slug from event tags when possible."""
    generic = {
        "news",
        "breaking-news",
        "trending",
        "prediction",
        "featured",
        "sports",
        "politics",
        "world",
        "elections",
        "global-elections",
        "us-presidential-election",
        "economy",
        "finance",
        "crypto",
    }
    candidate_slugs: list[str] = []
    for tag in event_tags:
        slug = str(tag.get("slug", "")).strip().lower()
        if not slug:
            continue
        if slug in generic:
            continue
        candidate_slugs.append(slug)
    if candidate_slugs:
        # Prefer the most specific category slug.
        return sorted(candidate_slugs, key=len, reverse=True)[0]
    return _derive_fallback_domain_from_tags(event_tags)


def _select_balanced_market_rows(
    market_rows: list[dict[str, Any]],
    max_categories: int,
    per_category_market_cap: int,
    max_markets: int,
    min_category_liquidity: float,
    excluded_categories: set[str],
) -> list[dict[str, Any]]:
    """Select markets with broad category coverage via capped round-robin."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    category_liquidity: dict[str, float] = {}
    for row in market_rows:
        category = row["domain"]
        by_category.setdefault(category, []).append(row)
        category_liquidity[category] = category_liquidity.get(category, 0.0) + float(
            row.get("market_liquidity", 0.0)
        )

    # Keep the highest-liquidity categories while applying quality filters.
    selected_categories = [
        category
        for category, _ in sorted(
            by_category.items(),
            key=lambda kv: category_liquidity.get(kv[0], 0.0),
            reverse=True,
        )
        if category_liquidity.get(category, 0.0) >= min_category_liquidity
        and category not in excluded_categories
    ][:max_categories]

    for category in selected_categories:
        by_category[category] = sorted(
            by_category[category],
            key=lambda row: float(row.get("market_liquidity", 0.0)),
            reverse=True,
        )[:per_category_market_cap]

    selected_rows: list[dict[str, Any]] = []
    done = False
    round_index = 0
    while not done and len(selected_rows) < max_markets:
        done = True
        for category in selected_categories:
            rows = by_category.get(category, [])
            if round_index < len(rows):
                selected_rows.append(rows[round_index])
                done = False
                if len(selected_rows) >= max_markets:
                    break
        round_index += 1

    return selected_rows


def flatten_event_markets(
    events: list[dict[str, Any]],
    min_event_markets: int,
) -> list[dict[str, Any]]:
    """Flatten nested event+market records into market-level rows."""
    rows: list[dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("id", ""))
        markets = event.get("markets") or []
        if not isinstance(markets, list) or len(markets) < min_event_markets:
            continue

        event_tags = event.get("tags") or []
        if not isinstance(event_tags, list):
            event_tags = []
        domain = _derive_category_from_tags(event_tags)

        tag_labels = [str(tag.get("label", "")) for tag in event_tags]
        tag_slugs = [str(tag.get("slug", "")) for tag in event_tags]

        for market in markets:
            question = str(market.get("question", "")).strip()
            if not question:
                continue

            outcomes = _parse_json_list_field(market.get("outcomes"))
            clob_ids = _parse_json_list_field(market.get("clobTokenIds"))
            if len(outcomes) < 1 or len(clob_ids) < 1:
                continue

            # Prefer "Yes" token for interpretable binary-return convention.
            yes_index = 0
            for index, outcome in enumerate(outcomes):
                if outcome.lower() == "yes":
                    yes_index = index
                    break
            yes_token_id = clob_ids[min(yes_index, len(clob_ids) - 1)]

            rows.append(
                {
                    "event_id": event_id,
                    "event_slug": str(event.get("slug", "")),
                    "event_title": str(event.get("title", "")),
                    "market_id": str(market.get("id", "")),
                    "market_slug": str(market.get("slug", "")),
                    "question": question,
                    "yes_token_id": yes_token_id,
                    "domain": domain,
                    "market_liquidity": float(market.get("liquidity", 0.0) or 0.0),
                    "tag_labels": "|".join(tag_labels),
                    "tag_slugs": "|".join(tag_slugs),
                    "event_start": str(event.get("startDate", "")),
                    "event_end": str(event.get("endDate", "")),
                }
            )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fetch_price_history(
    token_id: str,
    config: BuildConfig,
) -> list[dict[str, Any]]:
    """Fetch historical prices for one market token."""
    response = _request_json(
        f"{CLOB_BASE_URL}/prices-history",
        {
            "market": token_id,
            "interval": config.history_interval,
            "fidelity": config.history_fidelity,
        },
        timeout=config.request_timeout_sec,
    )
    history = response.get("history", []) if isinstance(response, dict) else []
    rows: list[dict[str, Any]] = []
    for point in history:
        if not isinstance(point, dict):
            continue
        ts = point.get("t")
        price = point.get("p")
        if ts is None or price is None:
            continue
        rows.append(
            {
                "token_id": token_id,
                "timestamp": int(ts),
                "datetime_utc": dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat(),
                "price": float(price),
            }
        )
    return rows


def _history_span_days(history_rows: list[dict[str, Any]]) -> float:
    """Compute time span covered by a token history in days."""
    if not history_rows:
        return 0.0
    ts = sorted(int(row["timestamp"]) for row in history_rows)
    if len(ts) < 2:
        return 0.0
    return float(ts[-1] - ts[0]) / 86400.0


def _compute_data_quality(
    market_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute simple quality checks for reproducibility."""
    unique_market_ids = {row["market_id"] for row in market_rows}
    unique_tokens = {row["yes_token_id"] for row in market_rows}
    domain_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    duplicate_points = 0

    seen_pairs: set[tuple[str, int]] = set()
    for row in history_rows:
        domain = next(
            (m["domain"] for m in market_rows if m["yes_token_id"] == row["token_id"]),
            "unknown",
        )
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        token = row["token_id"]
        token_counts[token] = token_counts.get(token, 0) + 1

        pair = (token, row["timestamp"])
        if pair in seen_pairs:
            duplicate_points += 1
        seen_pairs.add(pair)

    tokens_with_history = sum(1 for count in token_counts.values() if count > 0)
    market_missing_history = len(unique_tokens) - tokens_with_history

    monotonic_failures = 0
    for token in unique_tokens:
        token_ts = [
            row["timestamp"]
            for row in history_rows
            if row["token_id"] == token
        ]
        if token_ts != sorted(token_ts):
            monotonic_failures += 1

    return {
        "markets_count": len(market_rows),
        "unique_market_ids": len(unique_market_ids),
        "unique_tokens": len(unique_tokens),
        "history_points": len(history_rows),
        "market_missing_history": market_missing_history,
        "duplicate_history_points": duplicate_points,
        "non_monotonic_tokens": monotonic_failures,
        "domain_coverage": domain_counts,
    }


def build_dataset(
    project_root: pathlib.Path,
    config: BuildConfig | None = None,
) -> dict[str, pathlib.Path]:
    """Build and cache Polymarket dataset artifacts for downstream modeling."""
    cfg = config or BuildConfig()
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"

    events = fetch_active_events(cfg)
    raw_events_path = data_raw / f"{cfg.artifact_prefix}_events_raw.json"
    raw_events_path.parent.mkdir(parents=True, exist_ok=True)
    raw_events_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    market_rows = flatten_event_markets(events, min_event_markets=cfg.min_event_markets)

    filtered_market_rows = _select_balanced_market_rows(
        market_rows=market_rows,
        max_categories=cfg.max_categories,
        per_category_market_cap=cfg.per_category_market_cap,
        max_markets=cfg.max_markets,
        min_category_liquidity=cfg.min_category_liquidity,
        excluded_categories=set(cfg.excluded_category_slugs),
    )

    # Build a category liquidity summary so category choices are transparent.
    category_liquidity: dict[str, float] = {}
    category_market_count: dict[str, int] = {}
    for row in market_rows:
        category = row["domain"]
        category_liquidity[category] = category_liquidity.get(category, 0.0) + float(
            row.get("market_liquidity", 0.0)
        )
        category_market_count[category] = category_market_count.get(category, 0) + 1

    selected_categories = sorted({row["domain"] for row in filtered_market_rows})

    histories: list[dict[str, Any]] = []
    dropped_by_short_history_days = 0
    dropped_by_short_history_points = 0
    for row in filtered_market_rows:
        token_id = row["yes_token_id"]
        token_history = fetch_price_history(token_id, cfg)
        history_days = _history_span_days(token_history)
        if len(token_history) < cfg.min_history_points:
            dropped_by_short_history_points += 1
            time.sleep(cfg.sleep_between_requests_sec)
            continue
        if history_days < cfg.min_history_days:
            dropped_by_short_history_days += 1
            time.sleep(cfg.sleep_between_requests_sec)
            continue
        if len(token_history) >= cfg.min_history_points:
            histories.extend(token_history)
        time.sleep(cfg.sleep_between_requests_sec)

    # Keep only markets with sufficient history.
    valid_tokens = {row["token_id"] for row in histories}
    final_market_rows = [
        row for row in filtered_market_rows if row["yes_token_id"] in valid_tokens
    ]

    markets_path = data_processed / f"{cfg.artifact_prefix}_markets_filtered.csv"
    prices_path = data_processed / f"{cfg.artifact_prefix}_price_history.csv"
    quality_path = data_processed / f"{cfg.artifact_prefix}_data_quality.json"
    category_liquidity_path = data_processed / f"{cfg.artifact_prefix}_category_liquidity.csv"
    considered_domains_path = data_processed / f"{cfg.artifact_prefix}_considered_domains.json"

    _write_csv(
        markets_path,
        final_market_rows,
        columns=[
            "event_id",
            "event_slug",
            "event_title",
            "market_id",
            "market_slug",
            "question",
            "yes_token_id",
            "domain",
            "market_liquidity",
            "tag_labels",
            "tag_slugs",
            "event_start",
            "event_end",
        ],
    )
    _write_csv(
        prices_path,
        histories,
        columns=["token_id", "timestamp", "datetime_utc", "price"],
    )

    category_rows = [
        {
            "category": category,
            "total_liquidity": float(category_liquidity.get(category, 0.0)),
            "market_count": int(category_market_count.get(category, 0)),
            "selected_for_week8": int(category in selected_categories),
            "excluded_by_rule": int(category in set(cfg.excluded_category_slugs)),
            "passes_min_liquidity": int(
                float(category_liquidity.get(category, 0.0)) >= cfg.min_category_liquidity
            ),
        }
        for category in sorted(
            category_liquidity.keys(),
            key=lambda cat: category_liquidity.get(cat, 0.0),
            reverse=True,
        )
    ]
    _write_csv(
        category_liquidity_path,
        category_rows,
        columns=[
            "category",
            "total_liquidity",
            "market_count",
            "selected_for_week8",
            "excluded_by_rule",
            "passes_min_liquidity",
        ],
    )
    considered_domains_path.write_text(
        json.dumps(
            {
                "selected_categories": selected_categories,
                "selected_category_count": len(selected_categories),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    quality = _compute_data_quality(final_market_rows, histories)
    quality["min_history_days_filter"] = cfg.min_history_days
    quality["dropped_by_short_history_days"] = dropped_by_short_history_days
    quality["dropped_by_short_history_points"] = dropped_by_short_history_points
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    return {
        "events_raw": raw_events_path,
        "markets_filtered": markets_path,
        "price_history": prices_path,
        "data_quality": quality_path,
        "category_liquidity": category_liquidity_path,
        "considered_domains": considered_domains_path,
    }

