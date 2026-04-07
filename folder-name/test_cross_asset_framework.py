import pathlib

import numpy as np

from src.cross_asset_framework import _rolling_r2, _sigmoid, _load_options_prior, beta_weighted_hedge_notional


def test_beta_weighted_hedge_notional():
    out = beta_weighted_hedge_notional(pm_notional=10_000.0, target_corr=0.4, beta=2.0)
    assert abs(out - 2000.0) < 1e-9


def test_sigmoid_bounds():
    assert 0.0 < _sigmoid(-100.0) < 0.01
    assert 0.99 < _sigmoid(100.0) <= 1.0


def test_rolling_r2_high_when_linear_relation():
    x = np.linspace(-1.0, 1.0, 200)
    y = 0.5 + 1.8 * x
    r2 = _rolling_r2(y=y, x=x, window=64)
    assert float(np.max(r2)) > 0.95


def test_load_options_prior(tmp_path: pathlib.Path):
    csv_path = tmp_path / "opt.csv"
    csv_path.write_text(
        "date_utc,symbol,delta\n"
        "20260220,xle,0.3\n"
        "2026-02-20,xle,0.5\n"
        "20260220,nvda,-0.4\n",
        encoding="utf-8",
    )
    m = _load_options_prior(csv_path)
    assert abs(m[(20260220, "xle")] - 0.4) < 1e-9
    assert abs(m[(20260220, "nvda")] - 0.4) < 1e-9
