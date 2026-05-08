"""PM-only risk overlays: equity-informed domain tilts and correlation-based internal spreads.

All capital stays in Polymarket weights; equity is used only as a signal to tilt which
domains/contracts get relative weight. A second layer adds small zero-investment spreads
between negatively correlated PM *categories* using a precomputed correlation matrix
(e.g. ``week8_category_correlation.csv`` from ``covariance_diagnostics``).
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

import numpy as np

from src.baseline import _read_csv


def load_domain_ticker_tilt_map(path: pathlib.Path | None) -> dict[str, str]:
    """CSV columns: ``domain``, ``ticker`` (ticker drives multiplicative tilt for that domain)."""
    out: dict[str, str] = {}
    if path is None or not path.is_file():
        return out
    for row in _read_csv(path):
        dom = str(row.get("domain", "")).strip().lower()
        tick = str(row.get("ticker", "")).strip().upper()
        if dom and tick:
            out[dom] = tick
    return out


def read_category_correlation_csv(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Read matrix written by ``covariance_diagnostics._write_matrix_csv`` (row0 = header)."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise RuntimeError(f"Empty or invalid correlation CSV: {path}")
    header = rows[0]
    labels = [str(x).strip() for x in header[1:]]
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, row in enumerate(rows[1 : 1 + len(labels)]):
        if len(row) < 1 + len(labels):
            raise RuntimeError(f"Short row in {path} at line {i + 2}")
        mat[i, :] = np.array([float(x) for x in row[1 : 1 + len(labels)]], dtype=float)
    return labels, mat


def top_negative_correlation_pairs(
    labels: list[str],
    corr: np.ndarray,
    *,
    max_pairs: int = 5,
    max_correlation: float = -0.002,
) -> list[tuple[str, str, float]]:
    """Return up to ``max_pairs`` off-diagonal pairs with correlation <= ``max_correlation``."""
    n = len(labels)
    cand: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr[i, j])
            if c <= max_correlation:
                cand.append((labels[i], labels[j], c))
    cand.sort(key=lambda x: x[2])
    return cand[:max_pairs]


def domain_mean_returns(rmat: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    """Per time step, equal-weight mean return inside each domain (NaNs → 0)."""
    uniq = sorted(set(domains))
    idxs: dict[str, list[int]] = defaultdict(list)
    for j, d in enumerate(domains):
        idxs[d].append(j)
    out = np.zeros((rmat.shape[0], len(uniq)), dtype=float)
    for k, d in enumerate(uniq):
        ix = idxs[d]
        if ix:
            sub = np.nan_to_num(rmat[:, ix], nan=0.0)
            out[:, k] = np.mean(sub, axis=1)
    return uniq, out


def pm_category_spread_returns(
    rmat: np.ndarray,
    domains: list[str],
    pairs: list[tuple[str, str, float]],
) -> np.ndarray:
    """Zero-investment spread: mean_a (r_A - r_B) over pairs (equal-weight domain baskets)."""
    if not pairs:
        return np.zeros(rmat.shape[0], dtype=float)
    uniq, R = domain_mean_returns(rmat, domains)
    d_index = {d: i for i, d in enumerate(uniq)}
    acc = np.zeros(rmat.shape[0], dtype=float)
    for a, b, _c in pairs:
        ia = d_index.get(a)
        ib = d_index.get(b)
        if ia is None or ib is None:
            continue
        acc += R[:, ia] - R[:, ib]
    acc /= float(len(pairs))
    return acc


def build_equity_domain_tilt_multiplier(
    pm_timestamps: np.ndarray,
    domains: list[str],
    domain_to_ticker: dict[str, str],
    *,
    tilt_strength: float = 33.333,
    max_multiplier: float = 2.0,
) -> tuple[np.ndarray, dict[str, str]]:
    """``tilt_strength`` × max(0, lagged daily return) added to 1, clipped to ``max_multiplier``.

    Example: daily return +3% and strength 33.33 → multiplier ≈ 2.0 (double weight cap).
    """
    from src.equity_signal import _align_daily_returns_to_pm_steps, _fetch_yfinance_daily_returns

    uniq = sorted(set(domains))
    n_dom = len(uniq)
    T = int(pm_timestamps.shape[0])
    mult = np.ones((T, n_dom), dtype=np.float64)
    if not domain_to_ticker:
        return mult, {}

    d_idx = {d: i for i, d in enumerate(uniq)}
    tickers_needed = sorted({domain_to_ticker[d] for d in domain_to_ticker if d in d_idx})

    import datetime as dt

    ts = np.asarray(pm_timestamps, dtype=np.int64)
    d0 = dt.datetime.fromtimestamp(int(ts.min()), tz=dt.timezone.utc).date()
    d1 = dt.datetime.fromtimestamp(int(ts.max()), tz=dt.timezone.utc).date()
    start_date = d0 - dt.timedelta(days=400)
    end_date = d1 + dt.timedelta(days=3)

    ticker_ret: dict[str, np.ndarray] = {}
    for tick in tickers_needed:
        d_int, d_ret = _fetch_yfinance_daily_returns(
            tick, start=start_date.isoformat(), end=end_date.isoformat()
        )
        ticker_ret[tick] = _align_daily_returns_to_pm_steps(ts, d_int, d_ret)

    strength = float(max(0.0, tilt_strength))
    cap = float(max(1.0, max_multiplier))
    for dom, tick in domain_to_ticker.items():
        j = d_idx.get(dom)
        if j is None:
            continue
        r = ticker_ret.get(tick)
        if r is None or r.shape[0] != T:
            continue
        pos = np.maximum(r, 0.0)
        mult[:, j] = np.clip(1.0 + strength * pos, 1.0, cap)

    return mult, {d: domain_to_ticker[d] for d in domain_to_ticker if d in d_idx}


def token_resolution_tilt_multiplier(
    returns_matrix: np.ndarray,
    *,
    lookback: int = 64,
    strength: float = 8.0,
    max_multiplier: float = 2.0,
) -> np.ndarray:
    """Per asset and time: boost weights on tokens with large recent |returns| (causal).

    Shape ``(T, N)``. Uses rows ``[t-lookback, t)`` only. Concentrates within the optimizer
    distribution differently from domain-equal baseline → more divergent portfolio paths.
    """
    r = np.asarray(returns_matrix, dtype=np.float64)
    r = np.nan_to_num(r, nan=0.0)
    T, N = r.shape
    out = np.ones((T, N), dtype=np.float64)
    L = max(2, int(lookback))
    cap = float(max(1.0, max_multiplier))
    s = float(max(0.0, strength))
    for t in range(T):
        lo = max(0, t - L)
        hist = r[lo:t, :]
        if hist.shape[0] < 1:
            continue
        burst = np.max(np.abs(hist), axis=0)
        med = float(np.median(burst) + 1e-12)
        z = burst / med
        out[t, :] = np.clip(1.0 + s * np.tanh(z - 1.0), 1.0, cap)
    return out


def resolution_shock_domain_multiplier(
    returns_matrix: np.ndarray,
    domains: list[str],
    *,
    lookback: int = 72,
    strength: float = 10.0,
    max_multiplier: float = 2.0,
) -> np.ndarray:
    """Causal boost for domains with large recent |returns| (resolution / shock proxy).

    Uses strict-past window ``[t-lookback, t)`` of **domain-mean** returns; scales multipliers
    with ``tanh`` so a few volatile domains pick up extra weight without exploding.
    """
    _, R = domain_mean_returns(returns_matrix, domains)
    T, K = R.shape
    mult = np.ones((T, K), dtype=np.float64)
    L = max(2, int(lookback))
    ref = float(np.median(np.abs(R)) + 1e-12)
    cap = float(max(1.0, max_multiplier))
    s = float(max(0.0, strength))
    for t in range(T):
        lo = max(0, t - L)
        hist = R[lo:t, :]
        if hist.shape[0] < 2:
            continue
        burst = np.max(np.abs(hist), axis=0)
        z = burst / ref
        mult[t, :] = np.clip(1.0 + s * np.tanh(z - 1.0), 1.0, cap)
    return mult
