import numpy as np
import pytest

import src.equity_signal as equity_signal_mod

from src.equity_signal import (
    EquitySignalConfig,
    MarketRegime,
    REGIME_PM_HEDGE_ALLOCATION,
    _align_daily_returns_to_pm_steps,
    _parse_date_to_int,
    compute_pair_weights,
    get_regime,
    hedge_allocation_for_regime,
    pm_vs_options_divergence,
    regime_from_vix_level,
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


def test_regime_from_vix_level_bands():
    cfg = EquitySignalConfig(vix_low_threshold=18.0, vix_high_threshold=25.0, vix_crisis_threshold=35.0)
    assert regime_from_vix_level(15.0, cfg) == MarketRegime.LOW_VOL
    assert regime_from_vix_level(20.0, cfg) == MarketRegime.NORMAL
    assert regime_from_vix_level(30.0, cfg) == MarketRegime.HIGH_VOL
    assert regime_from_vix_level(40.0, cfg) == MarketRegime.CRISIS


def test_regime_pm_hedge_allocation_matches_architecture_one():
    assert REGIME_PM_HEDGE_ALLOCATION[MarketRegime.CRISIS] == 0.7
    assert REGIME_PM_HEDGE_ALLOCATION[MarketRegime.HIGH_VOL] == 0.5
    assert REGIME_PM_HEDGE_ALLOCATION[MarketRegime.NORMAL] == 0.3
    assert REGIME_PM_HEDGE_ALLOCATION[MarketRegime.LOW_VOL] == 0.1
    assert hedge_allocation_for_regime(MarketRegime.NORMAL) == 0.3


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


def test_compute_pair_weights_overweight_long_equity(monkeypatch):
    def _fake_pm_vs_options(*_a, **_k):
        return {"signal": "overweight", "divergence": -0.2, "options_prob": 0.6, "pm_prob": 0.4}

    monkeypatch.setattr(equity_signal_mod, "pm_vs_options_divergence", _fake_pm_vs_options)
    pm_w, eq_dir = compute_pair_weights(0.4, "SPY", 400.0, "2026-12-19", baseline_weight=0.02)
    assert pm_w == pytest.approx(0.03)
    assert eq_dir == 1


def test_compute_pair_weights_underweight_short_equity(monkeypatch):
    def _fake_pm_vs_options(*_a, **_k):
        return {"signal": "underweight", "divergence": 0.2, "options_prob": 0.3, "pm_prob": 0.5}

    monkeypatch.setattr(equity_signal_mod, "pm_vs_options_divergence", _fake_pm_vs_options)
    pm_w, eq_dir = compute_pair_weights(0.5, "SPY", 400.0, "2026-12-19", baseline_weight=0.02)
    assert pm_w == pytest.approx(0.01)
    assert eq_dir == -1


def test_compute_pair_weights_neutral_and_no_data(monkeypatch):
    def _neutral(*_a, **_k):
        return {"signal": "neutral", "divergence": 0.0, "options_prob": 0.5, "pm_prob": 0.5}

    monkeypatch.setattr(equity_signal_mod, "pm_vs_options_divergence", _neutral)
    pm_w, eq_dir = compute_pair_weights(0.5, "SPY", 400.0, "2026-12-19", baseline_weight=0.025)
    assert pm_w == pytest.approx(0.025)
    assert eq_dir == 0

    def _nodata(*_a, **_k):
        return {"signal": "no_data", "options_prob": None, "divergence": None}

    monkeypatch.setattr(equity_signal_mod, "pm_vs_options_divergence", _nodata)
    pm_w2, eq_dir2 = compute_pair_weights(0.5, "SPY", 400.0, "2026-12-19", baseline_weight=0.025)
    assert pm_w2 == pytest.approx(0.025)
    assert eq_dir2 == 0
