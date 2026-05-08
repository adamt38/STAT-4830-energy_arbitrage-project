"""Rank prediction-market contracts vs equity proxies by empirical co-movement.

Defines **PFCS** (Pearson–Fisher Coupling Score), a sample-size-aware magnitude of
linear linkage:

    PFCS = |atanh(clipped_r)| * sqrt(max(0, n_eff - 3))

where ``clipped_r`` is the Pearson correlation restricted to (-1, 1) and ``n_eff``
is the number of time steps with finite PM and equity returns. This matches the
scale of the standard Fisher z-test for non-zero correlation.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import pathlib
import re
from typing import Any

import numpy as np

import pandas as pd

from src.baseline import _build_price_matrix, _compute_returns, _read_csv
from src.equity_signal import DEFAULT_DOMAIN_TICKER_MAP, _align_daily_returns_to_pm_steps, _safe_import_yf, _utc_date_int
from src.stock_oil_hedge import fetch_stooq_daily_closes


# (lowercase substring in question/title/tags, ticker) — first match wins.
_KEYWORD_TICKER_RULES: list[tuple[str, str]] = [
    ("bitcoin", "IBIT"),
    ("btc", "IBIT"),
    ("microstrategy", "MSTR"),
    ("ethereum", "ETHA"),
    (" eth", "ETHA"),
    ("etf", "SPY"),
    ("nvidia", "NVDA"),
    ("google", "GOOGL"),
    ("alphabet", "GOOGL"),
    ("amazon", "AMZN"),
    ("apple", "AAPL"),
    ("microsoft", "MSFT"),
    ("meta", "META"),
    ("tesla", "TSLA"),
    ("spacex", "RKLB"),
    ("fed ", "TLT"),
    ("interest rate", "TLT"),
    ("interest rates", "TLT"),
    ("jerome powell", "TLT"),
    ("gold", "GLD"),
    (" s&p", "SPY"),
    ("sp500", "SPY"),
    ("oil", "USO"),
    ("crude", "USO"),
    ("brent", "BNO"),
    ("semiconductor", "SMH"),
    ("chip", "SMH"),
    ("ai model", "QQQ"),
    (" openai", "MSFT"),
    ("anthropic", "GOOGL"),
    ("databricks", "SNOW"),
    ("ipo", "IPO"),
    ("defense", "ITA"),
    ("lockheed", "LMT"),
    ("raytheon", "RTX"),
    ("brazil", "EWZ"),
    ("colombia", "ICOL"),
    ("french", "EWQ"),
    ("germany", "EWG"),
    ("europe", "VGK"),
    ("world cup", "XLY"),
    ("nba", "XLY"),
    ("bundesliga", "XLY"),
    ("champions league", "XLY"),
    ("formula 1", "FWONA"),
    ("f1", "FWONA"),
    (" trump", "DJT"),
    ("republican", "IWM"),
    ("democrat", "SPY"),
    ("senate", "SPY"),
    ("election", "SPY"),
    ("midterm", "SPY"),
    ("crypto", "IBIT"),
    ("fdv", "IBIT"),
    ("airdrop", "IBIT"),
    (" opensea", "COIN"),
    ("gensyn", "IBIT"),
    (" felix", "IBIT"),
    (" gensyn", "IBIT"),
    ("megaeth", "IBIT"),
    ("extended ", "COIN"),
    ("bitmine", "ETHA"),
]


def _normalize_text(*parts: str) -> str:
    return " ".join(p.lower() for p in parts if p)


def infer_keyword_ticker(question: str, event_title: str, tag_slugs: str) -> str | None:
    blob = _normalize_text(question, event_title, tag_slugs.replace("|", " "))
    for needle, tick in _KEYWORD_TICKER_RULES:
        if needle in blob:
            return tick
    return None


def infer_domain_etf(domain: str) -> str:
    d = domain.lower().strip()
    for key, ticker in DEFAULT_DOMAIN_TICKER_MAP.items():
        if key in d or d in key:
            return ticker
    if "crypto" in d or "bitcoin" in d or "eth" in d:
        return "IBIT"
    if "sport" in d or "nba" in d or "bundesliga" in d or "fifa" in d or "formula" in d:
        return "XLY"
    if "election" in d or "politic" in d or "midterm" in d or "congress" in d:
        return "SPY"
    if "econ" in d or "fed" in d:
        return "TLT"
    if "tech" in d or "ai" in d or "anthropic" in d or "gemini" in d:
        return "QQQ"
    return "SPY"


def ticker_candidates_for_market(row: dict[str, str]) -> list[tuple[str, str]]:
    """Return list of (ticker, source_label) to try."""
    q = row.get("question", "")
    title = row.get("event_title", "")
    tags = row.get("tag_slugs", "")
    dom = row.get("domain", "")
    out: list[tuple[str, str]] = []
    kw = infer_keyword_ticker(q, title, tags)
    if kw:
        out.append((kw, "keyword"))
    out.append((infer_domain_etf(dom), "domain_etf"))
    # de-dupe tickers keep first source
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for t, src in out:
        u = t.upper()
        if u not in seen:
            seen.add(u)
            deduped.append((u, src))
    return deduped


def _stooq_symbol(ticker: str) -> str:
    t = ticker.strip().upper().replace(".", "-")
    if t in {"BRK-B", "BRK.B"}:
        return "brk-b.us"
    return f"{t.lower()}.us"


def _fetch_one_ticker_stooq(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> tuple[np.ndarray, np.ndarray] | None:
    sym = _stooq_symbol(ticker)
    d1 = start.year * 10_000 + start.month * 100 + start.day
    d2 = end.year * 10_000 + end.month * 100 + end.day
    try:
        d_int, closes = fetch_stooq_daily_closes(sym, start_date_int=d1, end_date_int=d2, timeout_sec=45.0)
    except Exception:
        return None
    if len(closes) < 3:
        return None
    rets = np.zeros(len(closes), dtype=float)
    rets[1:] = closes[1:] / np.clip(closes[:-1], 1e-8, None) - 1.0
    return d_int, rets


def _fetch_bulk_daily_close(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ticker -> (date_int64, daily simple returns). Stooq per ticker first; Yahoo fills gaps."""
    if not tickers:
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in tickers:
        tu = t.upper()
        got = _fetch_one_ticker_stooq(tu, start, end)
        if got is not None:
            out[tu] = got
    try:
        missing = [t for t in tickers if t.upper() not in out]
        if not missing:
            return out
        yf = _safe_import_yf()
        raw = yf.download(
            tickers=missing,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if raw is not None and len(raw) > 0 and isinstance(raw.columns, pd.MultiIndex):
            for t in missing:
                tu = t.upper()
                try:
                    close = raw[(t, "Close")].dropna()
                except Exception:
                    try:
                        close = raw[(tu, "Close")].dropna()
                    except Exception:
                        continue
                if len(close) < 3:
                    continue
                closes = np.array(close.values, dtype=float)
                rets = np.zeros(len(closes), dtype=float)
                rets[1:] = closes[1:] / np.clip(closes[:-1], 1e-8, None) - 1.0
                idx = close.index
                date_int = np.array([int(x.strftime("%Y%m%d")) for x in idx], dtype=np.int64)
                out[tu] = (date_int, rets)
        elif raw is not None and len(raw) > 0 and "Close" in raw.columns and len(missing) == 1:
            close = raw["Close"].dropna()
            if len(close) >= 3:
                tu = missing[0].upper()
                closes = np.array(close.values, dtype=float)
                rets = np.zeros(len(closes), dtype=float)
                rets[1:] = closes[1:] / np.clip(closes[:-1], 1e-8, None) - 1.0
                idx = close.index
                date_int = np.array([int(x.strftime("%Y%m%d")) for x in idx], dtype=np.int64)
                out[tu] = (date_int, rets)
    except Exception:
        pass
    return out


def pfcs(pearson_r: float, n: int) -> float:
    """Pearson–Fisher Coupling Score."""
    if n <= 3 or not math.isfinite(pearson_r):
        return 0.0
    r = float(np.clip(pearson_r, -0.999, 0.999))
    z = abs(0.5 * math.log((1.0 + r) / (1.0 - r)))
    return float(z * math.sqrt(max(0, n - 3)))


def run_pm_stock_pair_ranking(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str = "week8",
    top_n: int = 50,
) -> tuple[list[dict[str, Any]], pathlib.Path]:
    processed = project_root / "data" / "processed"
    markets_rows = _read_csv(processed / f"{artifact_prefix}_markets_filtered.csv")
    history_rows = _read_csv(processed / f"{artifact_prefix}_price_history.csv")
    ts_values, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)
    if returns_matrix.size == 0:
        raise RuntimeError("No PM returns.")
    pm_ts = np.array(ts_values[1:], dtype=np.int64)
    token_index = {t: j for j, t in enumerate(kept_tokens)}
    meta_by_token = {row["yes_token_id"]: row for row in markets_rows if "yes_token_id" in row}

    all_tickers: set[str] = set()
    token_candidates: dict[str, list[tuple[str, str]]] = {}
    for tok in kept_tokens:
        row = meta_by_token.get(tok, {})
        cands = ticker_candidates_for_market(row)
        token_candidates[tok] = cands
        for t, _ in cands:
            all_tickers.add(t)

    min_day = _utc_date_int(int(pm_ts.min()))
    max_day = _utc_date_int(int(pm_ts.max()))
    start_d = dt.datetime.strptime(str(min_day), "%Y%m%d").date() - dt.timedelta(days=400)
    end_d = dt.datetime.strptime(str(max_day), "%Y%m%d").date() + dt.timedelta(days=3)

    eq_cache = _fetch_bulk_daily_close(sorted(all_tickers), start_d, end_d)

    rows_out: list[dict[str, Any]] = []
    for j, tok in enumerate(kept_tokens):
        r_pm = np.nan_to_num(returns_matrix[:, j], nan=0.0)
        row = meta_by_token.get(tok, {})
        for ticker, src in token_candidates[tok]:
            pack = eq_cache.get(ticker)
            if pack is None:
                continue
            d_int, d_ret = pack
            r_eq = _align_daily_returns_to_pm_steps(pm_ts, d_int, d_ret)
            mask = np.isfinite(r_pm) & np.isfinite(r_eq)
            if not np.any(mask):
                continue
            x = r_pm[mask]
            y = r_eq[mask]
            n = int(x.size)
            if n < 8:
                continue
            rho = float(np.corrcoef(x, y)[0, 1]) if n > 1 else 0.0
            if not math.isfinite(rho):
                rho = 0.0
            score = pfcs(rho, n)
            rows_out.append(
                {
                    "rank_metric": "PFCS",
                    "pfcs": score,
                    "pearson_r": rho,
                    "abs_r": abs(rho),
                    "n_obs": n,
                    "yes_token_id": tok,
                    "market_id": row.get("market_id", ""),
                    "market_slug": row.get("market_slug", ""),
                    "question": row.get("question", "")[:200],
                    "domain": row.get("domain", ""),
                    "equity_ticker": ticker,
                    "mapping_source": src,
                }
            )

    rows_out.sort(key=lambda r: (-float(r["pfcs"]), -float(r["abs_r"])))
    # unique by (token, ticker) keep best
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows_out:
        key = (str(r["yes_token_id"]), str(r["equity_ticker"]))
        if key not in best or float(r["pfcs"]) > float(best[key]["pfcs"]):
            best[key] = r
    merged = sorted(best.values(), key=lambda r: (-float(r["pfcs"]), -float(r["abs_r"])))
    top = merged[:top_n]
    for i, r in enumerate(top, start=1):
        r["rank"] = i

    out_csv = processed / f"{artifact_prefix}_pm_stock_top_{top_n}_pairs.csv"
    out_json = processed / f"{artifact_prefix}_pm_stock_top_{top_n}_pairs_summary.json"
    if top:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
            w.writeheader()
            w.writerows(top)
    else:
        out_csv.write_text("", encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "metric": {
                    "name": "PFCS",
                    "formula": "|atanh(clip(r))| * sqrt(max(0, n - 3))",
                    "description": "Fisher z magnitude scaled by effective sample size for correlation.",
                },
                "n_pairs_emitted": len(top),
                "artifact_prefix": artifact_prefix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return top, out_csv


if __name__ == "__main__":
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    pairs, path = run_pm_stock_pair_ranking(root, artifact_prefix="week8", top_n=50)
    print(f"Wrote {len(pairs)} rows to {path}")
