"""Equity signal ingestion and alignment for Polymarket domains.

Builds an aligned per-domain/per-asset equity signal matrix using yfinance data,
then computes PM-domain vs stock-sector correlation pairs for diagnostics.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.baseline import _build_price_matrix, _compute_returns, _read_csv


@dataclass(frozen=True)
class EquitySignalConfig:
    mapping_csv: pathlib.Path | None = None
    history_fidelity_minutes: int = 10
    default_ticker: str = "SPY"
    vix_symbol: str = "^VIX"
    vix_low_threshold: float = 18.0
    vix_high_threshold: float = 25.0
    vix_crisis_threshold: float = 35.0


DEFAULT_DOMAIN_TICKER_MAP: dict[str, str] = {
    "finance": "XLF",
    "economy": "XLI",
    "crypto": "IBIT",
    "politics": "SPY",
    "sports": "SPY",
    "science": "XLK",
    "culture": "XLC",
    "other": "SPY",
}


class MarketRegime(Enum):
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


REGIME_PENALTY_SCALES: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.LOW_VOL: {
        "variance_penalty": 0.5,
        "downside_penalty": 1.0,
        "max_domain_exposure_threshold": 0.18,
    },
    MarketRegime.NORMAL: {
        "variance_penalty": 1.0,
        "downside_penalty": 2.0,
        "max_domain_exposure_threshold": 0.12,
    },
    MarketRegime.HIGH_VOL: {
        "variance_penalty": 2.0,
        "downside_penalty": 4.0,
        "max_domain_exposure_threshold": 0.08,
    },
    MarketRegime.CRISIS: {
        "variance_penalty": 5.0,
        "downside_penalty": 8.0,
        "max_domain_exposure_threshold": 0.05,
    },
}


def _utc_date_int(ts: int) -> int:
    d = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date()
    return d.year * 10_000 + d.month * 100 + d.day


def _parse_date_to_int(raw: str) -> int | None:
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return int(raw)
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return int(raw.replace("-", ""))
    return None


def _load_mapping(path: pathlib.Path | None, fallback: dict[str, str]) -> dict[str, str]:
    mapping = {k.lower(): v.upper() for k, v in fallback.items()}
    if path is None or not path.exists():
        return mapping
    rows = _read_csv(path)
    for row in rows:
        domain = row.get("domain", "").strip().lower()
        ticker = row.get("ticker", "").strip().upper()
        if domain and ticker:
            mapping[domain] = ticker
    return mapping


def _safe_import_yf():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is required for equity signals. Install it via requirements.txt") from exc
    return yf


def get_regime(config: EquitySignalConfig, vix_override: float | None = None) -> tuple[MarketRegime, float | None]:
    """Return market regime from VIX level with optional override."""
    if vix_override is not None:
        vix = float(vix_override)
    else:
        try:
            yf = _safe_import_yf()
            data = yf.download(
                config.vix_symbol,
                period="7d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data is None or len(data) == 0:
                return MarketRegime.NORMAL, None
            close = data["Close"]
            vix = float(close.iloc[-1])
        except Exception:
            return MarketRegime.NORMAL, None
    if vix < config.vix_low_threshold:
        return MarketRegime.LOW_VOL, vix
    if vix < config.vix_high_threshold:
        return MarketRegime.NORMAL, vix
    if vix < config.vix_crisis_threshold:
        return MarketRegime.HIGH_VOL, vix
    return MarketRegime.CRISIS, vix


def _fetch_yfinance_daily_returns(ticker: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    yf = _safe_import_yf()

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or len(data) < 3:
        raise RuntimeError(f"Insufficient daily bars for ticker {ticker}.")
    close = data["Close"]
    idx = close.index
    closes = np.array(close.values, dtype=float)
    returns = np.zeros(len(closes), dtype=float)
    returns[1:] = closes[1:] / np.clip(closes[:-1], 1e-8, None) - 1.0
    date_int = np.array([int(ts.strftime("%Y%m%d")) for ts in idx], dtype=np.int64)
    return date_int, returns


def equity_implied_prob(
    ticker: str,
    strike: float,
    expiry: str,
    direction: str = "above",
) -> float | None:
    """Options-delta probability proxy from yfinance chain."""
    try:
        yf = _safe_import_yf()
        tk = yf.Ticker(ticker)
        if expiry not in tk.options:
            if not tk.options:
                return None
            expiry = min(
                tk.options,
                key=lambda d: abs((dt.date.fromisoformat(d) - dt.date.today()).days),
            )
        chain = tk.option_chain(expiry)
        if direction == "above":
            calls = chain.calls
            subset = calls[calls["strike"] >= strike]
            if subset.empty:
                return None
            row = subset.iloc[0]
            delta = row.get("delta", None)
            if delta is None or not np.isfinite(delta):
                spot = tk.fast_info.get("last_price", None)
                if spot is None:
                    return None
                iv = float(row.get("impliedVolatility", 0.30))
                dte = max((dt.date.fromisoformat(expiry) - dt.date.today()).days, 1) / 365.0
                sigma_t = max(iv * math.sqrt(dte), 1e-8)
                z = float(math.log(max(float(spot), 1e-8) / max(strike, 1e-8)) / sigma_t)
                delta = 1.0 / (1.0 + math.exp(-z))
            return float(np.clip(float(delta), 0.0, 1.0))
        puts = chain.puts
        subset = puts[puts["strike"] <= strike]
        if subset.empty:
            return None
        row = subset.iloc[-1]
        delta = row.get("delta", None)
        if delta is None or not np.isfinite(delta):
            return None
        return float(np.clip(abs(float(delta)), 0.0, 1.0))
    except Exception:
        return None


def pm_vs_options_divergence(
    pm_prob: float,
    ticker: str,
    strike: float,
    expiry: str,
    direction: str = "above",
    divergence_threshold: float = 0.10,
) -> dict[str, Any]:
    options_prob = equity_implied_prob(ticker=ticker, strike=strike, expiry=expiry, direction=direction)
    if options_prob is None:
        return {"pm_prob": pm_prob, "options_prob": None, "divergence": None, "signal": "no_data"}
    divergence = float(pm_prob - options_prob)
    if abs(divergence) < divergence_threshold:
        signal = "neutral"
    elif divergence > 0.0:
        signal = "underweight"
    else:
        signal = "overweight"
    return {
        "pm_prob": float(pm_prob),
        "options_prob": float(options_prob),
        "divergence": divergence,
        "signal": signal,
    }


def _align_daily_returns_to_pm_steps(pm_timestamps: np.ndarray, equity_date_int: np.ndarray, equity_rets: np.ndarray) -> np.ndarray:
    """Map each PM timestamp to last completed equity day return before that UTC day."""
    out = np.zeros(len(pm_timestamps), dtype=float)
    for i, ts in enumerate(pm_timestamps.astype(np.int64)):
        day = _utc_date_int(int(ts))
        j = int(np.searchsorted(equity_date_int, day, side="left") - 1)
        out[i] = float(equity_rets[j]) if j >= 0 else 0.0
    return out


def build_asset_equity_signal_matrix(
    project_root: pathlib.Path,
    *,
    artifact_prefix: str = "week8",
    asset_domains: list[str],
    config: EquitySignalConfig | None = None,
) -> tuple[np.ndarray, dict[str, pathlib.Path]]:
    """Create T x N asset-level equity signal matrix and pair diagnostics CSVs."""
    cfg = config or EquitySignalConfig()
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    markets_rows = _read_csv(processed / f"{artifact_prefix}_markets_filtered.csv")
    history_rows = _read_csv(processed / f"{artifact_prefix}_price_history.csv")
    ts_values, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)
    if returns_matrix.size == 0:
        raise RuntimeError("No PM returns available for equity signal alignment.")
    pm_ts = np.array(ts_values[1:], dtype=np.int64)
    if returns_matrix.shape[0] != pm_ts.shape[0]:
        raise RuntimeError("PM timestamp/return mismatch.")

    mapping = _load_mapping(cfg.mapping_csv, DEFAULT_DOMAIN_TICKER_MAP)
    unique_domains = sorted(set(asset_domains))
    domain_to_indices: dict[str, list[int]] = {}
    for i, d in enumerate(asset_domains):
        domain_to_indices.setdefault(d, []).append(i)

    min_day = _utc_date_int(int(pm_ts.min()))
    max_day = _utc_date_int(int(pm_ts.max()))
    start_date = dt.datetime.strptime(str(min_day), "%Y%m%d").date() - dt.timedelta(days=400)
    end_date = dt.datetime.strptime(str(max_day), "%Y%m%d").date() + dt.timedelta(days=3)

    domain_signal: dict[str, np.ndarray] = {}
    domain_ticker_rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, float | str]] = []

    for domain in unique_domains:
        ticker = mapping.get(domain.lower(), cfg.default_ticker).upper()
        d_int, d_ret = _fetch_yfinance_daily_returns(
            ticker=ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
        aligned = _align_daily_returns_to_pm_steps(pm_ts, d_int, d_ret)
        domain_signal[domain] = aligned
        domain_ticker_rows.append({"domain": domain, "ticker": ticker})

        idxs = domain_to_indices.get(domain, [])
        if idxs:
            pm_domain_ret = np.nan_to_num(returns_matrix[:, idxs], nan=0.0).mean(axis=1)
            corr = float(np.corrcoef(np.stack([pm_domain_ret, aligned], axis=0))[0, 1]) if pm_domain_ret.size > 2 else 0.0
            if not np.isfinite(corr):
                corr = 0.0
            pair_rows.append(
                {
                    "domain": domain,
                    "ticker": ticker,
                    "pm_stock_corr": corr,
                    "pm_mean_return": float(np.mean(pm_domain_ret)),
                    "stock_mean_return": float(np.mean(aligned)),
                    "pm_volatility": float(np.std(pm_domain_ret)),
                    "stock_volatility": float(np.std(aligned)),
                    "n_obs": int(pm_domain_ret.size),
                }
            )

    signal_matrix = np.zeros((len(pm_ts), len(asset_domains)), dtype=float)
    for j, domain in enumerate(asset_domains):
        signal_matrix[:, j] = domain_signal.get(domain, np.zeros(len(pm_ts), dtype=float))

    pair_path = processed / f"{artifact_prefix}_domain_equity_pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        if pair_rows:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
        else:
            handle.write("")

    mapping_path = processed / f"{artifact_prefix}_domain_equity_ticker_map.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        if domain_ticker_rows:
            writer = csv.DictWriter(handle, fieldnames=["domain", "ticker"])
            writer.writeheader()
            writer.writerows(domain_ticker_rows)
        else:
            handle.write("")

    signal_path = processed / f"{artifact_prefix}_domain_equity_signal_timeseries.csv"
    with signal_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["timestamp", *[f"{d}__equity_signal" for d in asset_domains]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, ts in enumerate(pm_ts):
            row: dict[str, float | int] = {"timestamp": int(ts)}
            for j, d in enumerate(asset_domains):
                row[f"{d}__equity_signal"] = float(signal_matrix[i, j])
            writer.writerow(row)

    regime, vix_value = get_regime(cfg)
    regime_path = processed / f"{artifact_prefix}_equity_regime_summary.json"
    regime_payload: dict[str, Any] = {
        "regime": regime.value,
        "vix_value": vix_value,
        "penalty_scales": REGIME_PENALTY_SCALES[regime],
    }
    regime_path.write_text(json.dumps(regime_payload, indent=2), encoding="utf-8")

    return signal_matrix, {
        "domain_equity_pairs": pair_path,
        "domain_equity_ticker_map": mapping_path,
        "domain_equity_signal_timeseries": signal_path,
        "equity_regime_summary": regime_path,
    }
