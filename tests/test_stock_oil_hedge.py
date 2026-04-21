"""Unit tests for oil/equity hedge helpers (no live market download)."""

import datetime as dt

import numpy as np

from src.stock_oil_hedge import (
    StockOilHedgeConfig,
    _lagged_daily_returns_for_timestamps,
    _utc_date_int,
    build_hedge_sleeve_returns,
    combine_sleeves,
    combine_sleeves_variable,
)


def test_utc_date_int():
    ts = int(dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc).timestamp())
    assert _utc_date_int(ts) == 20240102


def test_lagged_daily_returns_prior_trading_day():
    trading = np.array([20240101, 20240102, 20240103], dtype=np.int64)
    closes = np.array([100.0, 101.0, 99.0], dtype=float)
    ts = np.array(
        [int(dt.datetime(2024, 1, 3, 12, 0, tzinfo=dt.timezone.utc).timestamp())],
        dtype=np.int64,
    )
    r = _lagged_daily_returns_for_timestamps(ts, trading, closes)
    assert r.shape == (1,)
    assert abs(r[0] - (101.0 / 100.0 - 1.0)) < 1e-9


def test_combine_sleeves_weights():
    r_p = np.array([0.1, -0.05])
    r_h = np.array([0.02, 0.01])
    c = combine_sleeves(r_p, r_h, 0.25)
    assert np.allclose(c, 0.75 * r_p + 0.25 * r_h)


def test_combine_sleeves_variable():
    r_p = np.array([0.1, -0.05, 0.0])
    r_h = np.array([0.02, 0.01, 0.03])
    alpha = np.array([0.0, 0.5, 1.0])
    c = combine_sleeves_variable(r_p, r_h, alpha)
    assert np.allclose(c[0], r_p[0])
    assert np.allclose(c[1], 0.5 * r_p[1] + 0.5 * r_h[1])
    assert np.allclose(c[2], r_h[2])


def test_build_hedge_static_vs_inverse_equity():
    n = 10
    ts = np.arange(n, dtype=np.int64)
    r_poly = np.linspace(-0.01, 0.01, n)
    r_spy = np.full(n, 0.02)
    r_oil = np.full(n, -0.01)
    cfg = StockOilHedgeConfig(hedge_mode="static", oil_leg_weight=1.0, inverse_equity_leg_weight=0.0)
    h, m = build_hedge_sleeve_returns(ts, r_poly, r_spy, r_oil, cfg)
    assert np.allclose(m, 1.0)
    assert np.allclose(h, r_oil)
    cfg2 = StockOilHedgeConfig(hedge_mode="static", oil_leg_weight=0.5, inverse_equity_leg_weight=0.5)
    h2, _ = build_hedge_sleeve_returns(ts, r_poly, r_spy, r_oil, cfg2)
    assert np.allclose(h2, 0.5 * r_oil - 0.5 * r_spy)
