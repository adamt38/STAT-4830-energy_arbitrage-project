"""Yahoo Finance exogenous features aligned to Polymarket return-step timestamps."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS: list[str] = ["SPY", "QQQ", "XLE", "TLT", "BTC-USD"]
MERGE_BUFFER_SEC: int = 300
STALE_THRESHOLD_HOURLY_SEC: int = 2 * 3600
STALE_THRESHOLD_DAILY_SEC: int = 36 * 3600
Z_EPS: float = 1e-8


def ticker_to_prefix(ticker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ticker.lower()).strip("_")


def fetch_yahoo_close(
    ticker: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    interval: str,
) -> pd.Series:
    """Download Close only; UTC tz-aware index."""
    df = yf.download(
        ticker,
        start=start_utc,
        end=end_utc,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float, name="Close")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            col = ticker if ticker in close.columns else close.columns[0]
            close = close[col]
    close = close.dropna()
    close = close[~close.index.duplicated(keep="last")]
    idx = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    return pd.Series(close.values.astype(float), index=idx, name="Close")


def compute_bar_features(close: pd.Series, prefix: str, mode: str) -> pd.DataFrame:
    """mode: 'hourly' or 'daily'."""
    if close.empty:
        return pd.DataFrame()
    s = close.astype(float)
    log_p = np.log(np.clip(s.values, 1e-12, None))
    ret1 = np.diff(log_p, prepend=np.nan)
    col_ret1 = f"{prefix}_ret_1"
    feat = pd.DataFrame({col_ret1: ret1}, index=s.index)

    if mode == "hourly":
        lag_long = 6
        roll = 24
    elif mode == "daily":
        lag_long = 5
        roll = 20
    else:
        raise ValueError(f"Unknown mode: {mode}")

    feat[f"{prefix}_ret_6"] = np.log(
        np.clip(s / s.shift(lag_long), 1e-12, None)
    )
    feat[f"{prefix}_vol_24"] = feat[col_ret1].rolling(roll, min_periods=max(2, roll // 2)).std()
    feat[f"{prefix}_trend_24"] = feat[col_ret1].rolling(roll, min_periods=max(2, roll // 2)).mean()
    return feat


def compute_regime_raw(feat_df: pd.DataFrame) -> pd.DataFrame:
    out = feat_df.copy()
    out["risk_on_raw"] = (
        0.6 * out["spy_trend_24"]
        + 0.4 * out["qqq_trend_24"]
        - 0.8 * out["spy_vol_24"]
    )
    out["energy_raw"] = out["xle_trend_24"] - 0.5 * out["spy_trend_24"]
    out["rates_raw"] = out["tlt_trend_24"] - 0.5 * out["spy_trend_24"]
    return out


def is_equity_market_open(dt_utc: pd.Timestamp) -> int:
    ts = pd.Timestamp(dt_utc)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert("America/New_York")
    if et.weekday() >= 5:
        return 0
    t = et.time()
    open_t = pd.Timestamp("09:30").time()
    close_t = pd.Timestamp("16:00").time()
    return 1 if open_t <= t < close_t else 0


def align_to_polymarket_step_ts(
    step_ts: list[int],
    exog_feat_df: pd.DataFrame,
    merge_buffer_sec: int,
    stale_threshold_sec: int,
) -> pd.DataFrame:
    if not step_ts:
        return pd.DataFrame()
    poly_dt = pd.to_datetime(pd.Series(step_ts, dtype="int64"), unit="s", utc=True)
    left = pd.DataFrame(
        {
            "timestamp": step_ts,
            "poly_dt": poly_dt.values,
            "poly_dt_effective": poly_dt - pd.Timedelta(seconds=merge_buffer_sec),
        }
    )
    right = exog_feat_df.copy()
    if right.empty:
        right = pd.DataFrame({"exog_ts": pd.DatetimeIndex([], tz="UTC")})
    else:
        if not isinstance(right.index, pd.DatetimeIndex):
            right.index = pd.to_datetime(right.index, utc=True)
        right = right.sort_index()
        right = right[~right.index.duplicated(keep="last")]
        right = right.reset_index(names="exog_ts")
    left = left.sort_values("poly_dt_effective")
    if right.empty or len(right) == 0:
        merged = left.sort_values("timestamp").reset_index(drop=True)
        merged["exog_ts"] = pd.NaT
        feat_cols = list(exog_feat_df.columns) if not exog_feat_df.empty else []
        for c in feat_cols:
            merged[c] = np.nan
    else:
        merged = pd.merge_asof(
            left,
            right,
            left_on="poly_dt_effective",
            right_on="exog_ts",
            direction="backward",
        )
        merged = merged.sort_values("timestamp").reset_index(drop=True)
    if "exog_ts" not in merged.columns:
        merged["exog_ts"] = pd.NaT
    ref = pd.to_datetime(merged["exog_ts"], utc=True, errors="coerce")
    poly = pd.to_datetime(merged["poly_dt"], utc=True)
    age = (poly - ref).dt.total_seconds()
    merged["datetime_utc"] = poly.map(lambda x: pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ"))
    merged["exog_ref_datetime_utc"] = ref.map(
        lambda x: ""
        if pd.isna(x)
        else pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    merged["exog_age_sec"] = age.astype(float)
    merged["exog_is_stale"] = (
        (merged["exog_age_sec"] > stale_threshold_sec) | merged["exog_ts"].isna()
    ).astype(int)
    merged["is_equity_open"] = [is_equity_market_open(pd.Timestamp(x)) for x in poly]
    merged["is_weekend"] = poly.dt.dayofweek.isin([5, 6]).astype(int)
    out = merged.drop(columns=["poly_dt", "poly_dt_effective", "exog_ts"], errors="ignore")
    out = out.set_index(poly.rename("poly_dt"))
    return out


def _try_fetch_interval(ticker: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, interval: str) -> pd.Series:
    s = fetch_yahoo_close(ticker, start_utc, end_utc, interval)
    return s


def _interval_sufficient(series: pd.Series, mode: str) -> bool:
    if series.empty:
        return False
    need = 30 if mode == "hourly" else 25
    return len(series.dropna()) >= need


def build_exogenous_frame(
    step_ts: list[int],
    *,
    tickers: list[str] | None = None,
    merge_buffer_sec: int = MERGE_BUFFER_SEC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch Yahoo data, compute features, join on union index (ffill), align to steps."""
    tickers = list(tickers or TICKERS)
    if not step_ts:
        return pd.DataFrame(), {"error": "empty_step_ts"}

    t_min = int(min(step_ts))
    t_max = int(max(step_ts))
    poly_start = pd.Timestamp(t_min, unit="s", tz="UTC")
    poly_end = pd.Timestamp(t_max, unit="s", tz="UTC")
    start_utc = poly_start - pd.Timedelta(days=45)
    end_utc = poly_end + pd.Timedelta(days=1)

    intervals_used: dict[str, str] = {}
    any_daily = False
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        prefix = ticker_to_prefix(ticker)
        is_btc = "BTC" in ticker.upper()
        preferred = ("1h", "1d") if is_btc else ("1h", "1d")
        series: pd.Series | None = None
        mode = "hourly"
        used_iv = preferred[0]
        for iv in preferred:
            cand = _try_fetch_interval(ticker, start_utc, end_utc, iv)
            mode = "hourly" if iv == "1h" else "daily"
            if _interval_sufficient(cand, mode):
                series = cand
                used_iv = iv
                break
        if series is None:
            series = _try_fetch_interval(ticker, start_utc, end_utc, "1d")
            used_iv = "1d"
            mode = "daily"
        intervals_used[ticker] = used_iv
        if used_iv == "1d":
            any_daily = True
        feat = compute_bar_features(series, prefix, mode)
        if not feat.empty:
            feat.columns = [str(c) for c in feat.columns]
            frames.append(feat)

    if not frames:
        combined = pd.DataFrame()
    else:
        combined = frames[0]
        for f in frames[1:]:
            combined = combined.join(f, how="outer")
        combined = combined.sort_index()
        combined = combined.ffill()

    required = ["spy_trend_24", "spy_vol_24", "qqq_trend_24", "xle_trend_24", "tlt_trend_24"]
    for col in required:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = compute_regime_raw(combined)
    stale_sec = STALE_THRESHOLD_DAILY_SEC if any_daily else STALE_THRESHOLD_HOURLY_SEC
    aligned = align_to_polymarket_step_ts(step_ts, combined, merge_buffer_sec, stale_sec)

    meta: dict[str, Any] = {
        "tickers": tickers,
        "intervals_used": intervals_used,
        "merge_buffer_sec": merge_buffer_sec,
        "stale_threshold_sec": stale_sec,
        "stale_rule": "daily_threshold_if_any_ticker_1d_else_hourly",
        "any_ticker_daily": any_daily,
        "fetch_start_utc": start_utc.isoformat(),
        "fetch_end_utc": end_utc.isoformat(),
        "poly_range_start_utc": poly_start.isoformat(),
        "poly_range_end_utc": poly_end.isoformat(),
    }
    return aligned, meta


def standardize_regime_columns(df: pd.DataFrame, split_idx: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Training-only z-score on [0:split_idx] for risk_on, energy, rates raw columns."""
    out = df.copy()
    for raw_col, _z in [
        ("risk_on_raw", "risk_on_z"),
        ("energy_raw", "energy_z"),
        ("rates_raw", "rates_z"),
    ]:
        if raw_col not in out.columns:
            out[raw_col] = np.nan
    train_end = max(0, min(split_idx, len(out)))
    train = out.iloc[:train_end]
    stats: dict[str, Any] = {"split_idx": split_idx, "train_rows_used": train_end, "means": {}, "stds": {}}
    for raw_col, z_col in [
        ("risk_on_raw", "risk_on_z"),
        ("energy_raw", "energy_z"),
        ("rates_raw", "rates_z"),
    ]:
        s = pd.to_numeric(train[raw_col], errors="coerce")
        mu = float(np.nanmean(s.values)) if train_end > 0 else 0.0
        if not np.isfinite(mu):
            mu = 0.0
        sig = float(np.nanstd(s.values)) if train_end > 0 else 0.0
        if not np.isfinite(sig):
            sig = 0.0
        sig = max(sig, Z_EPS)
        stats["means"][raw_col] = mu
        stats["stds"][raw_col] = sig
        vals = pd.to_numeric(out[raw_col], errors="coerce")
        out[z_col] = ((vals - mu) / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, stats


def assert_no_leakage_vs_buffer(aligned: pd.DataFrame, merge_buffer_sec: int) -> None:
    buf = pd.Timedelta(seconds=merge_buffer_sec)
    for _, row in aligned.iterrows():
        ref_s = row.get("exog_ref_datetime_utc")
        if ref_s is None or ref_s == "" or (isinstance(ref_s, float) and np.isnan(ref_s)):
            continue
        poly = pd.Timestamp(row["datetime_utc"])
        if poly.tz is None:
            poly = poly.tz_localize("UTC")
        ref = pd.Timestamp(str(ref_s))
        if ref.tz is None:
            ref = ref.tz_localize("UTC")
        assert ref <= poly - buf, (ref, poly, buf)


def save_exogenous_artifacts(
    project_root: pathlib.Path,
    artifact_prefix: str,
    aligned: pd.DataFrame,
    meta: dict[str, Any],
    z_stats: dict[str, Any] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{artifact_prefix}_exogenous_features.csv"
    json_path = out_dir / f"{artifact_prefix}_exogenous_quality.json"

    export = aligned.reset_index(drop=True)
    export = export.drop(columns=["poly_dt"], errors="ignore")
    if "timestamp" not in export.columns:
        idx = aligned.index
        if isinstance(idx, pd.DatetimeIndex):
            export.insert(0, "timestamp", (idx.astype("int64") // 10**9).astype(int))
    export.to_csv(csv_path, index=False)

    n = len(export)
    miss_frac = float(export[[c for c in export.columns if c.endswith("_ret_1")]].isna().mean().mean()) if n else 0.0
    stale_frac = float(export["exog_is_stale"].mean()) if n and "exog_is_stale" in export.columns else 0.0
    open_frac = float(export["is_equity_open"].mean()) if n and "is_equity_open" in export.columns else 0.0

    quality = {
        **meta,
        "row_count": n,
        "missingness_fraction_approx": miss_frac,
        "stale_fraction": stale_frac,
        "equity_open_fraction": open_frac,
    }
    if z_stats:
        quality["z_score_params"] = z_stats

    json_path.write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def build_and_save_exogenous(
    project_root: pathlib.Path,
    artifact_prefix: str,
    step_ts: list[int],
    split_idx: int,
    *,
    baseline_timestamps: list[int] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Build raw+aligned features, z-score on training split, save CSV + quality JSON."""
    if baseline_timestamps is not None:
        if len(baseline_timestamps) != len(step_ts) or baseline_timestamps != step_ts:
            raise ValueError("step_ts must exactly match baseline_timeseries timestamps (order and values)")
    aligned, meta = build_exogenous_frame(step_ts)
    if len(aligned) != len(step_ts):
        raise RuntimeError(f"Exogenous rows {len(aligned)} != step_ts {len(step_ts)}")
    aligned_num = aligned.reset_index()
    if aligned_num.shape[1] > 0:
        first_col = aligned_num.columns[0]
        aligned_num = aligned_num.rename(columns={first_col: "poly_dt"})
    if "timestamp" not in aligned_num.columns:
        aligned_num.insert(0, "timestamp", step_ts)
    aligned_std, z_stats = standardize_regime_columns(aligned_num, split_idx)
    assert_no_leakage_vs_buffer(aligned_std, meta.get("merge_buffer_sec", MERGE_BUFFER_SEC))
    return save_exogenous_artifacts(project_root, artifact_prefix, aligned_std, meta, z_stats=z_stats)


def apply_zscores_to_saved_csv(
    project_root: pathlib.Path,
    artifact_prefix: str,
    split_idx: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Recompute z columns from existing raw columns and update CSV + JSON."""
    out_dir = project_root / "data" / "processed"
    csv_path = out_dir / f"{artifact_prefix}_exogenous_features.csv"
    json_path = out_dir / f"{artifact_prefix}_exogenous_quality.json"
    df = pd.read_csv(csv_path)
    df_std, z_stats = standardize_regime_columns(df, split_idx)
    meta: dict[str, Any] = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    meta["z_score_params"] = z_stats
    df_std.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path
