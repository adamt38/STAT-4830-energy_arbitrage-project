"""Seven-day-window hedge allocation (α) and leg-weight optimization."""

import numpy as np

from src.hedge_window_opt import (
    cap_gain,
    combine_pm_stock_enhancer_gated,
    excess_gain_vs_baseline,
    momentum_gated_alpha_bar,
    optimize_hedge_on_window,
    rolling_mean_past_only,
    static_hedge_leg_returns,
)


def test_optimize_hedge_alpha_prefers_full_hedge_when_pm_flat_hedge_positive():
    n = 20
    m7 = np.zeros(n, dtype=bool)
    m7[-7:] = True
    r_pm = np.zeros(n, dtype=float)
    r_base = np.zeros(n, dtype=float)
    r_hedge_leg = np.zeros(n, dtype=float)
    r_hedge_leg[-7:] = 0.01
    r_spy = np.zeros(n, dtype=float)
    r_oil = np.zeros(n, dtype=float)
    # static leg = oil - spy with 0.5/0.5 -> 0.5*r_oil - 0.5*r_spy; use direct leg as "oil" with spy=0 by setting
    # r_oil = 2 * r_hedge_leg, weights 0.5,0.5 -> 0.5*2h - 0 = h
    r_oil = 2.0 * r_hedge_leg
    out = optimize_hedge_on_window(
        r_pm,
        r_base,
        r_spy,
        r_oil,
        m7,
        alpha_grid=np.linspace(0.0, 1.0, 51),
        leg_weight_pairs=((0.5, 0.5),),
    )
    assert out["best_hedge_allocation_alpha"] >= 0.98
    assert out["excess_gain_vs_baseline_window"] > 0.0
    assert cap_gain(out["best_combined_returns"][m7]) > cap_gain(r_pm[m7])


def test_excess_gain_matches_manual_two_step():
    r_p = np.array([0.1, 0.0, 0.0], dtype=float)
    r_b = np.array([0.05, 0.0, 0.0], dtype=float)
    m = np.array([True, True, False], dtype=bool)
    ex = excess_gain_vs_baseline(r_p, r_b, m)
    manual = (1.1 * 1.0) - (1.05 * 1.0)
    assert abs(ex - manual) < 1e-9


def test_static_hedge_leg_normalization():
    r_oil = np.array([0.02, -0.01], dtype=float)
    r_spy = np.array([0.01, 0.01], dtype=float)
    h = static_hedge_leg_returns(r_oil, r_spy, 1.0, 1.0)
    assert np.allclose(h, 0.5 * r_oil - 0.5 * r_spy)


def test_momentum_gate_zeros_alpha_when_signal_nonpositive():
    roll = np.array([-1.0, 0.0, 0.01, 0.02], dtype=float)
    bar = momentum_gated_alpha_bar(0.07, roll)
    assert bar[0] == 0.0 and bar[1] == 0.0
    assert bar[2] == 0.07 and bar[3] == 0.07


def test_combine_pm_stock_enhancer_matches_unweighted_pm_when_alpha_zero():
    n = 10
    r_pm = np.linspace(0.001, 0.002, n)
    r_spy = np.full(n, -0.01)
    combined, _, alpha_bar = combine_pm_stock_enhancer_gated(
        r_pm, r_spy, hedge_weight=0.07, momentum_gate_window=3
    )
    assert np.all(alpha_bar == 0.0)
    assert np.allclose(combined, r_pm)


def test_rolling_mean_past_only_first_bar_zero():
    x = np.array([5.0, -1.0, 2.0], dtype=float)
    r = rolling_mean_past_only(x, 5)
    assert r[0] == 0.0
    assert r[1] == 5.0
