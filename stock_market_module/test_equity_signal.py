import numpy as np

from src.equity_signal import (
    EquitySignalConfig,
    MarketRegime,
    _align_daily_returns_to_pm_steps,
    _parse_date_to_int,
    get_regime,
    pm_vs_options_divergence,
)


def test_parse_date_to_int():
    assert _parse_date_to_int("20260220") == 20260220
    assert _parse_date_to_int("2026-02-20") == 20260220
    assert _parse_date_to_int("02/20/2026") is None


def test_align_daily_returns_to_pm_steps():
    # PM bars on Jan 3 and Jan 4 should map to returns ending Jan 2 and Jan 3.
    pm_ts = np.array([1704283200, 1704369600], dtype=np.int64)  # 2024-01-03 12:00Z, 2024-01-04 12:00Z
    eq_dates = np.array([20240101, 20240102, 20240103], dtype=np.int64)
    eq_rets = np.array([0.0, 0.01, -0.02], dtype=float)
    out = _align_daily_returns_to_pm_steps(pm_ts, eq_dates, eq_rets)
    assert np.allclose(out, np.array([0.01, -0.02]))


def test_regime_with_override():
    cfg = EquitySignalConfig()
    regime, vix = get_regime(cfg, vix_override=27.0)
    assert regime == MarketRegime.HIGH_VOL
    assert vix == 27.0


def test_pm_vs_options_no_data_path():
    # Invalid ticker/expiry should safely degrade to no_data (no exception).
    out = pm_vs_options_divergence(
        pm_prob=0.45,
        ticker="NOT_A_REAL_TICKER",
        strike=100.0,
        expiry="2099-12-31",
    )
    assert out["pm_prob"] == 0.45
    assert out["signal"] in {"no_data", "neutral", "underweight", "overweight"}
