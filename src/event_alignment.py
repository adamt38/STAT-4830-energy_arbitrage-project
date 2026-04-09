"""Cross-platform event alignment for equivalent binary contracts."""

from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AlignedPair:
    """One matched Polymarket/Kalshi market pair."""

    canonical_event_key: str
    poly_market_id: str
    poly_question: str
    poly_event_title: str
    kalshi_market_id: str
    kalshi_question: str
    kalshi_event_title: str
    score: float
    method: str
    mapped_kalshi_exchange_event_id: str = ""


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", value.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _token_set(value: str) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if token}


def _structured_token_set(*blobs: str) -> set[str]:
    """Tokenize slugs and tickers (hyphen/underscore splits, alphanumeric runs)."""
    tokens: set[str] = set()
    for raw in blobs:
        s = str(raw or "").strip().lower()
        if not s:
            continue
        for part in re.split(r"[^a-z0-9]+", s):
            if len(part) >= 2:
                tokens.add(part)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _parse_iso_utc(value: str) -> datetime | None:
    s = (value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _end_times_compatible(poly_end: str, kalshi_end: str, max_days_gap: float | None) -> bool:
    """If both ends parse, require |Δdays| <= max_days_gap; missing dates pass."""
    if max_days_gap is None:
        return True
    a = _parse_iso_utc(poly_end)
    b = _parse_iso_utc(kalshi_end)
    if a is None or b is None:
        return True
    delta_days = abs((a - b).total_seconds()) / 86400.0
    return delta_days <= max_days_gap


def _event_group_key(row: dict[str, str]) -> str:
    eid = str(row.get("exchange_event_id") or "").strip()
    if eid:
        return eid
    return f"_solo_{row.get('exchange_market_id', '')}"


def _event_cluster_score(poly_rows: list[dict[str, str]], kalshi_rows: list[dict[str, str]]) -> tuple[float, str]:
    """Similarity between a Polymarket event cluster and a Kalshi event cluster."""
    pe = poly_rows[0]
    ke = kalshi_rows[0]
    poly_et = str(pe.get("event_title") or "").strip()
    kalshi_et = str(ke.get("event_title") or "").strip()
    s_et = _jaccard(_token_set(poly_et), _token_set(kalshi_et))

    slug_blob = " ".join(
        sorted(
            {
                str(r.get("event_slug") or "").strip()
                + " "
                + str(r.get("market_slug") or "").strip()
                for r in poly_rows
            }
        )
    )
    poly_text_blob = f"{poly_et} {slug_blob}".strip()
    kalshi_tickers = " ".join(sorted({str(r.get("exchange_market_id") or "") for r in kalshi_rows}))
    kalshi_blob = f"{kalshi_et} {ke.get('exchange_event_id', '')} {ke.get('series_ticker', '')} {kalshi_tickers}".strip()
    s_text = _jaccard(_token_set(poly_text_blob), _token_set(kalshi_blob))

    poly_struct: set[str] = set()
    for r in poly_rows:
        poly_struct |= _structured_token_set(
            str(r.get("event_slug") or ""),
            str(r.get("market_slug") or ""),
        )
    kalshi_struct: set[str] = set()
    for r in kalshi_rows:
        kalshi_struct |= _structured_token_set(
            str(r.get("exchange_event_id") or ""),
            str(r.get("exchange_market_id") or ""),
            str(r.get("series_ticker") or ""),
        )
    s_struct = _jaccard(poly_struct, kalshi_struct)

    base = max(s_et, s_text, s_struct)
    if base <= 0.0:
        return 0.0, "event_cluster_none"
    if base == s_struct:
        return base, "event_cluster_structured"
    if base == s_et:
        return base, "event_cluster_event_title"
    return base, "event_cluster_text"


def _pair_similarity(
    poly: dict[str, str],
    kalshi: dict[str, str],
    *,
    domain_bonus: float = 0.05,
) -> tuple[float, str]:
    """Text + structured (slug/ticker) similarity; domain bonus if domains match."""
    poly_et = str(poly.get("event_title") or "").strip()
    kalshi_et = str(kalshi.get("event_title") or "").strip()
    poly_q = str(poly.get("question") or "").strip()
    kalshi_q = str(kalshi.get("question") or "").strip()

    s_event = _jaccard(_token_set(poly_et), _token_set(kalshi_et))
    s_question = _jaccard(_token_set(poly_q), _token_set(kalshi_q))
    poly_blob = f"{poly_et} {poly_q}".strip()
    kalshi_blob = f"{kalshi_et} {kalshi_q}".strip()
    s_combined = _jaccard(_token_set(poly_blob), _token_set(kalshi_blob))

    s_struct = _jaccard(
        _structured_token_set(
            str(poly.get("event_slug") or ""),
            str(poly.get("market_slug") or ""),
        ),
        _structured_token_set(
            str(kalshi.get("exchange_event_id") or ""),
            str(kalshi.get("exchange_market_id") or ""),
            str(kalshi.get("series_ticker") or ""),
        ),
    )

    score_base = max(s_event, s_question, s_combined, s_struct)
    if score_base <= 0.0:
        base_method = "jaccard_max"
    elif score_base == s_struct:
        base_method = "jaccard_structured"
    elif score_base == s_combined:
        base_method = "jaccard_combined"
    elif score_base == s_event:
        base_method = "jaccard_event_title"
    else:
        base_method = "jaccard_question"

    poly_domain = str(poly.get("domain", "other"))
    kalshi_domain = str(kalshi.get("domain", "other"))
    bonus = domain_bonus if poly_domain == kalshi_domain else 0.0
    return score_base + bonus, base_method


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _index_by_market_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(r.get("exchange_market_id") or "").strip(): r for r in rows if str(r.get("exchange_market_id") or "").strip()}


def _enriched_aligned_rows(
    candidates: list[AlignedPair],
    poly_by_mid: dict[str, dict[str, str]],
    kalshi_by_mid: dict[str, dict[str, str]],
    *,
    strict_same_event: bool,
) -> list[dict[str, Any]]:
    """Join alignment pairs to canonical market rows for human-readable review."""
    out: list[dict[str, Any]] = []
    strict_s = "true" if strict_same_event else "false"
    for pair in candidates:
        pr = poly_by_mid.get(pair.poly_market_id, {})
        kr = kalshi_by_mid.get(pair.kalshi_market_id, {})
        mapped = pair.mapped_kalshi_exchange_event_id
        out.append(
            {
                "canonical_event_key": pair.canonical_event_key,
                "poly_market_id": pair.poly_market_id,
                "poly_question": pair.poly_question,
                "poly_event_title": pair.poly_event_title,
                "kalshi_market_id": pair.kalshi_market_id,
                "kalshi_question": pair.kalshi_question,
                "kalshi_event_title": pair.kalshi_event_title,
                "score": pair.score,
                "method": pair.method,
                "alignment_strict_same_event": strict_s,
                "mapped_kalshi_exchange_event_id": mapped,
                "poly_exchange_symbol": pr.get("exchange_symbol", ""),
                "poly_exchange_event_id": pr.get("exchange_event_id", ""),
                "poly_event_slug": pr.get("event_slug", ""),
                "poly_market_slug": pr.get("market_slug", ""),
                "poly_domain": pr.get("domain", ""),
                "poly_end_time_utc": pr.get("end_time_utc", ""),
                "poly_liquidity": pr.get("liquidity", ""),
                "kalshi_exchange_symbol": kr.get("exchange_symbol", ""),
                "kalshi_exchange_event_id": kr.get("exchange_event_id", ""),
                "kalshi_series_ticker": kr.get("series_ticker", ""),
                "kalshi_domain": kr.get("domain", ""),
                "kalshi_end_time_utc": kr.get("end_time_utc", ""),
                "kalshi_liquidity": kr.get("liquidity", ""),
            }
        )
    return out


ALIGNED_PAIRS_FIELDNAMES = [
    "canonical_event_key",
    "poly_market_id",
    "poly_question",
    "poly_event_title",
    "kalshi_market_id",
    "kalshi_question",
    "kalshi_event_title",
    "score",
    "method",
    "alignment_strict_same_event",
    "mapped_kalshi_exchange_event_id",
    "poly_exchange_symbol",
    "poly_exchange_event_id",
    "poly_event_slug",
    "poly_market_slug",
    "poly_domain",
    "poly_end_time_utc",
    "poly_liquidity",
    "kalshi_exchange_symbol",
    "kalshi_exchange_event_id",
    "kalshi_series_ticker",
    "kalshi_domain",
    "kalshi_end_time_utc",
    "kalshi_liquidity",
]


def _greedy_event_mapping(
    poly_by_event: dict[str, list[dict[str, str]]],
    kalshi_by_event: dict[str, list[dict[str, str]]],
    *,
    min_event_cluster_similarity: float,
) -> dict[str, str | None]:
    """Map each Polymarket event key -> Kalshi event_ticker or None (one-to-one on Kalshi)."""
    used_kalshi_events: set[str] = set()
    mapping: dict[str, str | None] = {}
    poly_keys = sorted(poly_by_event.keys(), key=lambda k: (k.startswith("_solo"), k))
    kalshi_items = list(kalshi_by_event.items())

    for poly_key in poly_keys:
        prows = poly_by_event[poly_key]
        best_k: str | None = None
        best_score = -1.0
        for kalshi_key, krows in kalshi_items:
            if kalshi_key in used_kalshi_events:
                continue
            score, _ = _event_cluster_score(prows, krows)
            if score > best_score:
                best_score = score
                best_k = kalshi_key
        if best_k is not None and best_score >= min_event_cluster_similarity:
            mapping[poly_key] = best_k
            used_kalshi_events.add(best_k)
        else:
            mapping[poly_key] = None
    return mapping


def align_polymarket_kalshi_events(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str,
    min_similarity: float = 0.25,
    min_event_cluster_similarity: float = 0.12,
    max_end_days_gap: float | None = None,
    strict_same_event: bool = True,
) -> dict[str, pathlib.Path]:
    """Align markets using event-cluster matching, date gating, and structured tokens.

    When ``strict_same_event`` is True, only Poly rows whose event maps to a Kalshi
    event are considered, and market matches are restricted to that Kalshi event (no
    global fallback). Payoff equivalence still requires separate human/rule checks.
    """
    processed = project_root / "data" / "processed"
    poly_path = processed / f"{artifact_prefix}_poly_canonical_markets.csv"
    kalshi_path = processed / f"{artifact_prefix}_kalshi_canonical_markets.csv"

    poly_rows = _read_csv(poly_path) if poly_path.exists() else []
    kalshi_rows = _read_csv(kalshi_path) if kalshi_path.exists() else []

    poly_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in poly_rows:
        poly_by_event[_event_group_key(row)].append(row)

    kalshi_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kalshi_rows:
        kalshi_by_event[_event_group_key(row)].append(row)

    event_map = _greedy_event_mapping(
        poly_by_event,
        kalshi_by_event,
        min_event_cluster_similarity=min_event_cluster_similarity,
    )

    def _best_in_pool(
        pool: list[dict[str, str]],
    ) -> tuple[dict[str, str] | None, float, str]:
        best_s = -1.0
        best_m: dict[str, str] | None = None
        best_method = ""
        for kalshi in pool:
            kalshi_id = str(kalshi.get("exchange_market_id", ""))
            if kalshi_id in used_kalshi_markets:
                continue
            if not _end_times_compatible(
                str(poly.get("end_time_utc") or ""),
                str(kalshi.get("end_time_utc") or ""),
                max_end_days_gap,
            ):
                continue
            score, method = _pair_similarity(poly, kalshi)
            if score > best_s:
                best_s = score
                best_method = method
                best_m = kalshi
        return best_m, best_s, best_method

    candidates: list[AlignedPair] = []
    used_kalshi_markets: set[str] = set()
    skipped_no_event_map = 0
    skipped_below_threshold_in_mapped_event = 0
    for poly in poly_rows:
        poly_key = _event_group_key(poly)
        kalshi_eid = event_map.get(poly_key)
        poly_q = str(poly.get("question", ""))
        poly_et = str(poly.get("event_title", ""))

        if strict_same_event:
            if not kalshi_eid:
                skipped_no_event_map += 1
                continue
            restricted = kalshi_by_event[kalshi_eid]
            best_match, best_score, best_method = _best_in_pool(restricted)
            if best_match is None or best_score < min_similarity:
                skipped_below_threshold_in_mapped_event += 1
                continue
        else:
            if kalshi_eid:
                restricted = kalshi_by_event[kalshi_eid]
            else:
                restricted = kalshi_rows
            best_match, best_score, best_method = _best_in_pool(restricted)
            if kalshi_eid and (best_match is None or best_score < min_similarity):
                skipped_below_threshold_in_mapped_event += 1
            if (best_match is None or best_score < min_similarity) and kalshi_eid:
                global_match, global_score, global_method = _best_in_pool(kalshi_rows)
                if global_match is not None and global_score > best_score:
                    best_match, best_score, best_method = global_match, global_score, global_method
            if best_match is None or best_score < min_similarity:
                continue

        kalshi_id = str(best_match.get("exchange_market_id", ""))
        used_kalshi_markets.add(kalshi_id)
        poly_id = str(poly.get("exchange_market_id", ""))
        canonical_event_key = f"evt_{poly_id}_{kalshi_id}"
        mapped_eid = str(kalshi_eid or "")
        candidates.append(
            AlignedPair(
                canonical_event_key=canonical_event_key,
                poly_market_id=poly_id,
                poly_question=poly_q,
                poly_event_title=poly_et,
                kalshi_market_id=kalshi_id,
                kalshi_question=str(best_match.get("question", "")),
                kalshi_event_title=str(best_match.get("event_title", "")),
                score=float(best_score),
                method=best_method,
                mapped_kalshi_exchange_event_id=mapped_eid,
            )
        )

    poly_by_mid = _index_by_market_id(poly_rows)
    kalshi_by_mid = _index_by_market_id(kalshi_rows)
    pairs_path = processed / f"{artifact_prefix}_aligned_market_pairs.csv"
    _write_csv(
        pairs_path,
        rows=_enriched_aligned_rows(
            candidates,
            poly_by_mid,
            kalshi_by_mid,
            strict_same_event=strict_same_event,
        ),
        fieldnames=ALIGNED_PAIRS_FIELDNAMES,
    )

    summary_path = processed / f"{artifact_prefix}_alignment_summary.json"
    mapped_events = sum(1 for v in event_map.values() if v)
    strict_note = (
        "strict_same_event: pairs only within mapped Kalshi event buckets; no global fallback. "
        "Does not guarantee identical payoffs—verify resolution rules manually."
        if strict_same_event
        else "strict_same_event off: unmapped Poly events search all Kalshi; global fallback allowed."
    )
    summary_path.write_text(
        json.dumps(
            {
                "artifact_prefix": artifact_prefix,
                "poly_rows": len(poly_rows),
                "kalshi_rows": len(kalshi_rows),
                "matched_pairs": len(candidates),
                "strict_same_event": strict_same_event,
                "poly_markets_skipped_no_event_map": skipped_no_event_map,
                "poly_markets_skipped_below_threshold_in_mapped_event": skipped_below_threshold_in_mapped_event,
                "min_similarity": min_similarity,
                "min_event_cluster_similarity": min_event_cluster_similarity,
                "max_end_days_gap": max_end_days_gap,
                "poly_event_groups": len(poly_by_event),
                "kalshi_event_groups": len(kalshi_by_event),
                "mapped_event_pairs": mapped_events,
                "scoring": "Event clusters: max(jaccard(event titles), jaccard(text+slugs vs text+tickers), "
                "jaccard(structured slug tokens vs ticker/series)). "
                "Markets: same plus structured pair score; optional domain bonus; "
                "date gate on end_time_utc when both parseable. "
                + strict_note,
                "aligned_pairs_csv": "aligned_market_pairs.csv includes alignment_strict_same_event, "
                "mapped_kalshi_exchange_event_id (Kalshi event_ticker used for the bucket), then poly_/kalshi_ "
                "end_time_utc, domains, slugs, series_ticker, liquidity, exchange_symbol.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "aligned_pairs": pairs_path,
        "alignment_summary": summary_path,
    }
