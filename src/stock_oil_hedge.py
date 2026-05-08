"""Oil and equity hedge layer for Polymarket portfolio returns.

Pulls daily US equity (SPY) and oil proxy (USO) closes from Stooq, maps each
Polymarket return interval to a *lagged* trailing daily return (last completed
trading day before the bar's UTC calendar day) to avoid same-day look-ahead.

The hedge leg captures the commonly discussed *negative* equity–oil covariance
regime by blending long oil returns with a short equity exposure (``-r_spy``).

Optional analyst overlays adjust the equity-hedge intensity from a CSV of
per-date scores (e.g. EPS revision breadth or recommendation changes).
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
import math
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import requests

from src.baseline import _build_price_matrix, _compute_returns, _dynamic_portfolio_returns, _max_drawdown, _read_csv, _sortino_ratio
from src.equity_signal import EquitySignalConfig, regime_from_vix_level, hedge_allocation_for_regime


StooqSymbol = str
HedgeMode = Literal["static", "rolling_mv"]


@dataclass(frozen=True)
class StockOilHedgeConfig:
    """Configuration for equity/oil hedge on top of a Polymarket sleeve."""

    spy_stooq: StooqSymbol = "spy.us"
    oil_stooq: StooqSymbol = "uso.us"
    # Non-negative weights; normalized to sum to 1 when both > 0.
    oil_leg_weight: float = 0.5
    inverse_equity_leg_weight: float = 0.5
    # Fraction of capital allocated to the hedge sleeve; rest stays on Polymarket.
    hedge_allocation: float = 0.2
    # If True, each bar uses VIX regime (lagged daily ^VIX) to set hedge weight (Architecture 1);
    # ``hedge_allocation`` is then only a fallback when VIX history is unavailable.
    regime_switching_pm_hedge_split: bool = True
    hedge_mode: HedgeMode = "static"
    rolling_window: int = 48
    rolling_mv_clip: float = 2.0
    request_timeout_sec: float = 30.0
    analyst_csv: pathlib.Path | None = None
    analyst_date_column: str = "date_utc"
    analyst_tilt_column: str = "equity_hedge_tilt"
    # Analyst tilt in [-1, 1]: +1 scales down short-equity hedge (bullish research).
    analyst_tilt_scale: float = 1.0


def _utc_date_int(ts: int) -> int:
    d = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date()
    return d.year * 10_000 + d.month * 100 + d.day


def _ymd_to_int(y: int, m: int, day: int) -> int:
    return y * 10_000 + m * 100 + day


def _parse_stooq_date(value: str) -> int | None:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        y, m, d = value.split("-")
        return _ymd_to_int(int(y), int(m), int(d))
    if re.fullmatch(r"\d{8}", value):
        v = int(value)
        return v
    return None


def fetch_stooq_daily_closes(
    symbol: str,
    *,
    start_date_int: int,
    end_date_int: int,
    timeout_sec: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Download daily rows from Stooq; return (date_int_sorted, close_sorted)."""
    d1 = f"{start_date_int:08d}"
    d2 = f"{end_date_int:08d}"
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d&d1={d1}&d2={d2}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; STAT4830/1.0)"}
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        raise RuntimeError(f"Empty Stooq response for {symbol!r}.")

    reader = csv.DictReader(text.splitlines())
    dates: list[int] = []
    closes: list[float] = []
    for row in reader:
        raw_date = row.get("Date") or row.get("date")
        if not raw_date:
            continue
        parsed = _parse_stooq_date(str(raw_date))
        if parsed is None:
            continue
        close_raw = row.get("Close") or row.get("close")
        if close_raw in (None, ""):
            continue
        try:
            c = float(close_raw)
        except ValueError:
            continue
        if not math.isfinite(c) or c <= 0.0:
            continue
        dates.append(parsed)
        closes.append(c)
    if len(dates) < 3:
        raise RuntimeError(
            f"Insufficient Stooq history for {symbol!r} "
            f"(got {len(dates)} rows). Check symbol or date range."
        )

    order = np.argsort(np.array(dates, dtype=np.int64))
    d_arr = np.array(dates, dtype=np.int64)[order]
    c_arr = np.array(closes, dtype=float)[order]
    # De-duplicate dates (keep last close)
    if d_arr.size >= 2:
        uniq_dates: list[int] = []
        uniq_closes: list[float] = []
        for i in range(d_arr.size):
            if uniq_dates and uniq_dates[-1] == int(d_arr[i]):
                uniq_closes[-1] = float(c_arr[i])
            else:
                uniq_dates.append(int(d_arr[i]))
                uniq_closes.append(float(c_arr[i]))
        d_arr = np.array(uniq_dates, dtype=np.int64)
        c_arr = np.array(uniq_closes, dtype=float)
    return d_arr, c_arr


def _lagged_daily_returns_for_timestamps(
    timestamps: np.ndarray,
    trading_dates: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """For each unix timestamp, return a simple daily return from the last Stooq close before the bar's UTC day.

    Return is (C_s / C_{s-1}) - 1 where s is the last trading date strictly before the bar's calendar day.
    """
    out = np.zeros(len(timestamps), dtype=float)
    td = trading_dates
    for i, ts in enumerate(timestamps.astype(np.int64)):
        day = _utc_date_int(int(ts))
        j = bisect.bisect_left(td, day) - 1
        if j < 1:
            out[i] = 0.0
            continue
        c_prev = float(closes[j - 1])
        c_now = float(closes[j])
        if c_prev <= 0.0:
            out[i] = 0.0
            continue
        out[i] = c_now / c_prev - 1.0
    return out


def _normalize_pair(w_oil: float, w_inv_eq: float) -> tuple[float, float]:
    a = max(0.0, float(w_oil))
    b = max(0.0, float(w_inv_eq))
    s = a + b
    if s <= 0.0:
        return 0.5, 0.5
    return a / s, b / s


def _load_analyst_tilt_by_date(path: pathlib.Path, date_col: str, tilt_col: str) -> dict[int, float]:
    rows = _read_csv(path)
    out: dict[int, float] = {}
    for row in rows:
        raw = row.get(date_col, "")
        parsed: int | None = None
        if isinstance(raw, str) and re.fullmatch(r"\d{8}", raw.strip()):
            parsed = int(raw.strip())
        elif isinstance(raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw.strip()):
            y, m, d = raw.strip().split("-")
            parsed = _ymd_to_int(int(y), int(m), int(d))
        if parsed is None:
            continue
        try:
            tilt = float(row.get(tilt_col, "0.0"))
        except (TypeError, ValueError):
            tilt = 0.0
        tilt = float(np.clip(tilt, -1.0, 1.0))
        out[parsed] = tilt
    return out


def _analyst_tilt_for_timestamp(ts: int, mapping: dict[int, float], scale: float) -> float:
    if not mapping:
        return 0.0
    day = _utc_date_int(int(ts))
    if day in mapping:
        return float(np.clip(mapping[day] * scale, -1.0, 1.0))
    keys = sorted(mapping.keys())
    j = bisect.bisect_right(keys, day) - 1
    if j < 0:
        return 0.0
    return float(np.clip(mapping[keys[j]] * scale, -1.0, 1.0))


def build_hedge_sleeve_returns(
    timestamps: np.ndarray,
    r_poly: np.ndarray,
    r_spy: np.ndarray,
    r_oil: np.ndarray,
    cfg: StockOilHedgeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hedge sleeve returns, per-step scalar multiplier).

    Static mode uses multiplier 1. Rolling MV chooses k_t on past data so
    Var(r_poly + k_t * hedge_static) is approximately minimized, then sets
    hedge_t = k_t * hedge_static_t.
    """
    w_oil, w_eq = _normalize_pair(cfg.oil_leg_weight, cfg.inverse_equity_leg_weight)
    analyst_map: dict[int, float] = {}
    if cfg.analyst_csv is not None and cfg.analyst_csv.exists():
        analyst_map = _load_analyst_tilt_by_date(
            cfg.analyst_csv,
            date_col=cfg.analyst_date_column,
            tilt_col=cfg.analyst_tilt_column,
        )

    eq_scales = np.ones(len(timestamps), dtype=float)
    for i, ts in enumerate(timestamps.astype(np.int64)):
        tilt = _analyst_tilt_for_timestamp(int(ts), analyst_map, cfg.analyst_tilt_scale)
        # Bullish tilt (+) -> soften short equity; bearish (-) -> keep full short.
        eq_scales[i] = max(0.0, 1.0 - tilt)

    hedge_static = w_oil * r_oil - w_eq * eq_scales * r_spy
    mv_mult = np.ones(len(timestamps), dtype=float)
    if cfg.hedge_mode == "rolling_mv":
        win = max(8, int(cfg.rolling_window))
        clip = max(0.1, float(cfg.rolling_mv_clip))
        for i in range(len(hedge_static)):
            lo = max(0, i - win)
            pp = r_poly[lo:i]
            hh = hedge_static[lo:i]
            if pp.size < 6:
                mv_mult[i] = 1.0
                continue
            cov = float(np.cov(pp, hh, bias=True)[0, 1])
            var = float(np.var(hh))
            if var < 1e-12:
                h_hat = 0.0
            else:
                h_hat = -cov / var
            mv_mult[i] = float(np.clip(h_hat, -clip, clip))
    return hedge_static * mv_mult, mv_mult


def combine_sleeves(
    r_poly: np.ndarray,
    r_hedge: np.ndarray,
    hedge_allocation: float,
) -> np.ndarray:
    alpha = float(np.clip(hedge_allocation, 0.0, 1.0))
    return (1.0 - alpha) * r_poly + alpha * r_hedge


def combine_sleeves_variable(
    r_poly: np.ndarray,
    r_hedge: np.ndarray,
    hedge_allocation_per_bar: np.ndarray,
) -> np.ndarray:
    """Blend PM and hedge with a possibly different capital split each bar (regime switching)."""
    alpha = np.clip(np.asarray(hedge_allocation_per_bar, dtype=float), 0.0, 1.0)
    return (1.0 - alpha) * r_poly + alpha * r_hedge


def _lagged_vix_close_for_timestamps(
    timestamps: np.ndarray,
    trading_dates: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Last available ^VIX close strictly before each bar's UTC calendar day (same calendar logic as SPY lag)."""
    out = np.full(len(timestamps), np.nan, dtype=float)
    td = trading_dates
    for i, ts in enumerate(timestamps.astype(np.int64)):
        day = _utc_date_int(int(ts))
        j = bisect.bisect_left(td, day) - 1
        if j >= 0:
            out[i] = float(closes[j])
    return out


def _fetch_vix_daily_closes_yfinance(
    start_date_int: int,
    end_date_int: int,
    symbol: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Download daily VIX closes; return sorted (date_int, close) or (None, None) on failure."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return None, None

    y1, m1, d1 = start_date_int // 10000, (start_date_int // 100) % 100, start_date_int % 100
    y2, m2, d2 = end_date_int // 10000, (end_date_int // 100) % 100, end_date_int % 100
    start = dt.date(y1, m1, d1) - dt.timedelta(days=14)
    end = dt.date(y2, m2, d2) + dt.timedelta(days=3)

    try:
        raw = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None, None
    if raw is None or len(raw) == 0:
        return None, None
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.size < 3:
        return None, None
    idx = pd.to_datetime(close.index)
    dates = (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy(dtype=np.int64)
    vals = close.to_numpy(dtype=float)
    order = np.argsort(dates)
    dates = dates[order]
    vals = vals[order]
    uniq_dates: list[int] = []
    uniq_closes: list[float] = []
    for i in range(dates.size):
        di = int(dates[i])
        if uniq_dates and uniq_dates[-1] == di:
            uniq_closes[-1] = float(vals[i])
        else:
            uniq_dates.append(di)
            uniq_closes.append(float(vals[i]))
    return np.array(uniq_dates, dtype=np.int64), np.array(uniq_closes, dtype=float)


def compute_stock_oil_hedge_series(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    config: StockOilHedgeConfig | None = None,
) -> dict[str, Any]:
    """Return aligned PM / hedge / combined return arrays (no file IO)."""
    cfg = config or StockOilHedgeConfig()
    markets_path = project_root / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv"
    prices_path = project_root / "data" / "processed" / f"{artifact_prefix}_price_history.csv"
    markets_rows = _read_csv(markets_path)
    history_rows = _read_csv(prices_path)
    ts_values, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    if len(ts_values) < 3 or not kept_tokens:
        raise RuntimeError("Polymarket price matrix empty; run polymarket pipeline first.")

    returns_matrix = _compute_returns(price_matrix)
    token_to_domain = {row["yes_token_id"]: row["domain"] for row in markets_rows}
    unique_domains = sorted({token_to_domain[t] for t in kept_tokens})
    domain_count = max(len(unique_domains), 1)
    domain_share = 1.0 / float(domain_count)
    domain_to_tokens: dict[str, list[str]] = {}
    for token in kept_tokens:
        domain_to_tokens.setdefault(token_to_domain[token], []).append(token)

    weights = np.zeros(len(kept_tokens), dtype=float)
    token_index = {t: i for i, t in enumerate(kept_tokens)}
    for _, tokens_in_domain in domain_to_tokens.items():
        per_market_weight = domain_share / float(len(tokens_in_domain))
        for token in tokens_in_domain:
            weights[token_index[token]] = per_market_weight
    r_poly = _dynamic_portfolio_returns(returns_matrix, weights)

    ts_ret = np.array(ts_values[1:], dtype=np.int64)
    if r_poly.shape[0] != ts_ret.shape[0]:
        raise RuntimeError("Timestamp / return length mismatch in Polymarket series.")

    min_d = _utc_date_int(int(ts_ret.min())) - 10_000
    max_d = _utc_date_int(int(ts_ret.max())) + 10
    offline = os.environ.get("STAT4830_OFFLINE_HEDGE", "").strip().lower() in ("1", "true", "yes")
    n = int(r_poly.shape[0])
    eq_cfg = EquitySignalConfig()
    if offline:
        # No Stooq: zero equity/oil sleeve; combined = PM only (alpha_bar = 0).
        r_spy = np.zeros(n, dtype=float)
        r_oil = np.zeros(n, dtype=float)
        r_hedge = np.zeros(n, dtype=float)
        mv_mult = np.ones(n, dtype=float)
        alpha_bar = np.zeros(n, dtype=float)
        vix_level = np.full(n, np.nan, dtype=float)
        regime_bar = np.array([""] * n, dtype=object)
        stooq_offline_fallback = True
    else:
        d_spy, c_spy = fetch_stooq_daily_closes(
            cfg.spy_stooq,
            start_date_int=min_d,
            end_date_int=max_d,
            timeout_sec=cfg.request_timeout_sec,
        )
        d_oil, c_oil = fetch_stooq_daily_closes(
            cfg.oil_stooq,
            start_date_int=min_d,
            end_date_int=max_d,
            timeout_sec=cfg.request_timeout_sec,
        )
        r_spy = _lagged_daily_returns_for_timestamps(ts_ret, d_spy, c_spy)
        r_oil = _lagged_daily_returns_for_timestamps(ts_ret, d_oil, c_oil)
        r_hedge, mv_mult = build_hedge_sleeve_returns(ts_ret, r_poly, r_spy, r_oil, cfg)
        stooq_offline_fallback = False
        if cfg.regime_switching_pm_hedge_split:
            d_vix, c_vix = _fetch_vix_daily_closes_yfinance(min_d, max_d, eq_cfg.vix_symbol)
            if d_vix is None or d_vix.size < 3:
                alpha_bar = np.full(n, float(cfg.hedge_allocation), dtype=float)
                vix_level = np.full(n, np.nan, dtype=float)
                regime_bar = np.array(["fallback_constant_alpha"] * n, dtype=object)
            else:
                vix_level = _lagged_vix_close_for_timestamps(ts_ret, d_vix, c_vix)
                alpha_bar = np.zeros(n, dtype=float)
                regime_bar = np.empty(n, dtype=object)
                for i in range(n):
                    reg = regime_from_vix_level(float(vix_level[i]), eq_cfg)
                    alpha_bar[i] = hedge_allocation_for_regime(reg)
                    regime_bar[i] = reg.value
        else:
            alpha_bar = np.full(n, float(cfg.hedge_allocation), dtype=float)
            vix_level = np.full(n, np.nan, dtype=float)
            regime_bar = np.array([""] * n, dtype=object)

    r_combined = combine_sleeves_variable(r_poly, r_hedge, alpha_bar)

    cum_poly = np.cumprod(1.0 + r_poly)
    cum_hedge_only = np.cumprod(1.0 + r_hedge)
    cum_combo = np.cumprod(1.0 + r_combined)
    dd_poly, mdd_poly = _max_drawdown(cum_poly)
    dd_hedge, mdd_hedge = _max_drawdown(cum_hedge_only)
    dd_combo, mdd_combo = _max_drawdown(cum_combo)

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or b.size < 3:
            return 0.0
        m = np.corrcoef(np.stack([a, b], axis=0))[0, 1]
        return float(m) if np.isfinite(m) else 0.0

    corr_sp_poly = _safe_corr(r_spy, r_poly)
    corr_oil_poly = _safe_corr(r_oil, r_poly)
    corr_hedge_poly = _safe_corr(r_hedge, r_poly)

    metrics: dict[str, Any] = {
        "strategy": "equal_weight_polymarket_plus_oil_equity_hedge",
        "spy_symbol": cfg.spy_stooq,
        "oil_symbol": cfg.oil_stooq,
        "hedge_allocation": cfg.hedge_allocation,
        "regime_switching_pm_hedge_split": cfg.regime_switching_pm_hedge_split,
        "mean_hedge_allocation_bar": float(np.mean(alpha_bar)),
        "oil_leg_weight": cfg.oil_leg_weight,
        "inverse_equity_leg_weight": cfg.inverse_equity_leg_weight,
        "hedge_mode": cfg.hedge_mode,
        "rolling_window": cfg.rolling_window,
        "sortino_polymarket": _sortino_ratio(r_poly),
        "sortino_hedge_sleeve": _sortino_ratio(r_hedge),
        "sortino_combined": _sortino_ratio(r_combined),
        "max_drawdown_polymarket": mdd_poly,
        "max_drawdown_hedge_sleeve": mdd_hedge,
        "max_drawdown_combined": mdd_combo,
        "capital_gain_polymarket": float(cum_poly[-1] - 1.0),
        "capital_gain_hedge_sleeve": float(cum_hedge_only[-1] - 1.0),
        "capital_gain_combined": float(cum_combo[-1] - 1.0),
        "corr_lagged_spy_polymarket": corr_sp_poly,
        "corr_lagged_oil_polymarket": corr_oil_poly,
        "corr_hedge_sleeve_polymarket": corr_hedge_poly,
        "analyst_csv": str(cfg.analyst_csv) if cfg.analyst_csv else None,
        "stooq_offline_fallback": stooq_offline_fallback,
        "stooq_offline_note": (
            "Stooq fetch skipped (STAT4830_OFFLINE_HEDGE=1); combined returns equal Polymarket sleeve. "
            "Unset env and rerun for live SPY/USO hedge."
            if stooq_offline_fallback
            else None
        ),
    }

    return {
        "ts_ret": ts_ret,
        "r_poly": r_poly,
        "r_spy": r_spy,
        "r_oil": r_oil,
        "r_hedge": r_hedge,
        "r_combined": r_combined,
        "alpha_bar": alpha_bar,
        "vix_level": vix_level,
        "regime_bar": regime_bar,
        "mv_mult": mv_mult,
        "cum_poly": cum_poly,
        "cum_hedge_only": cum_hedge_only,
        "cum_combo": cum_combo,
        "dd_poly": dd_poly,
        "dd_hedge": dd_hedge,
        "dd_combo": dd_combo,
        "metrics": metrics,
    }


def run_stock_oil_hedge_experiment(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
    config: StockOilHedgeConfig | None = None,
) -> dict[str, pathlib.Path]:
    """Fetch market data, align to Polymarket baseline timestamps, write metrics + series."""
    cfg = config or StockOilHedgeConfig()
    result = compute_stock_oil_hedge_series(project_root, artifact_prefix, cfg)
    ts_ret = result["ts_ret"]
    r_poly = result["r_poly"]
    r_spy = result["r_spy"]
    r_oil = result["r_oil"]
    r_hedge = result["r_hedge"]
    r_combined = result["r_combined"]
    alpha_bar = result["alpha_bar"]
    vix_level = result["vix_level"]
    regime_bar = result["regime_bar"]
    mv_mult = result["mv_mult"]
    cum_poly = result["cum_poly"]
    cum_hedge_only = result["cum_hedge_only"]
    cum_combo = result["cum_combo"]
    dd_poly = result["dd_poly"]
    dd_combo = result["dd_combo"]
    metrics = dict(result["metrics"])

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = out_dir / f"{artifact_prefix}_stock_oil_hedge_timeseries.csv"
    with series_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "polymarket_return",
                "lagged_spy_daily_return",
                "lagged_oil_daily_return",
                "hedge_sleeve_return",
                "rolling_mv_multiplier",
                "combined_return",
                "hedge_allocation_bar",
                "lagged_vix_close",
                "vix_regime",
                "cumulative_polymarket",
                "cumulative_hedge_sleeve",
                "cumulative_combined",
                "drawdown_polymarket",
                "drawdown_combined",
            ],
        )
        w.writeheader()
        for i in range(len(ts_ret)):
            w.writerow(
                {
                    "timestamp": int(ts_ret[i]),
                    "polymarket_return": float(r_poly[i]),
                    "lagged_spy_daily_return": float(r_spy[i]),
                    "lagged_oil_daily_return": float(r_oil[i]),
                    "hedge_sleeve_return": float(r_hedge[i]),
                    "rolling_mv_multiplier": float(mv_mult[i]),
                    "combined_return": float(r_combined[i]),
                    "hedge_allocation_bar": float(alpha_bar[i]),
                    "lagged_vix_close": (
                        f"{float(vix_level[i]):.6f}"
                        if np.isfinite(vix_level[i])
                        else ""
                    ),
                    "vix_regime": str(regime_bar[i]),
                    "cumulative_polymarket": float(cum_poly[i]),
                    "cumulative_hedge_sleeve": float(cum_hedge_only[i]),
                    "cumulative_combined": float(cum_combo[i]),
                    "drawdown_polymarket": float(dd_poly[i]),
                    "drawdown_combined": float(dd_combo[i]),
                },
            )

    metrics_path = out_dir / f"{artifact_prefix}_stock_oil_hedge_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = out_dir / f"{artifact_prefix}_stock_oil_hedge_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "intent": (
                    "Blend a long-oil / short-equity daily sleeve with Polymarket returns "
                    "to hedge episodes where energy shocks and equity moves diverge; "
                    "correlations are regime-dependent so weights are tunable and optional "
                    "rolling minimum-variance scaling reduces variance vs. Polymarket alone."
                ),
                "analyst_overlay": (
                    "Optional CSV maps UTC dates to equity_hedge_tilt in [-1,1]: bullish "
                    "research reduces the short-equity intensity; bearish increases it."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "stock_oil_hedge_timeseries": series_path,
        "stock_oil_hedge_metrics": metrics_path,
        "stock_oil_hedge_summary": summary_path,
    }
