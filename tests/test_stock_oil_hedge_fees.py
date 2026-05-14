"""Tests for stock/PM sleeve transaction-cost accounting."""

from __future__ import annotations

import numpy as np

from src.stock_oil_hedge import (
    combine_sleeves_variable,
    combine_sleeves_variable_with_fees,
    sleeve_allocation_turnover,
)
from src.hedge_window_opt import (
    combine_pm_stock_enhancer_gated,
    optimize_hedge_on_window,
)
from src.kelly_stock_pm_combined import combine_kelly_stock_pm_series


def test_sleeve_turnover_uses_pm_and_hedge_weight_changes():
    alpha = np.array([0.0, 0.2, 0.1], dtype=float)

    turnover = sleeve_allocation_turnover(alpha)

    np.testing.assert_allclose(turnover, np.array([0.0, 0.4, 0.2]))


def test_fee_aware_sleeve_combiner_subtracts_allocation_costs():
    r_pm = np.array([0.01, 0.02, 0.03], dtype=float)
    r_hedge = np.array([0.00, 0.01, -0.01], dtype=float)
    alpha = np.array([0.0, 0.2, 0.1], dtype=float)

    net, turnover, costs = combine_sleeves_variable_with_fees(
        r_pm,
        r_hedge,
        alpha,
        fee_rate=0.001,
    )

    gross = combine_sleeves_variable(r_pm, r_hedge, alpha)
    np.testing.assert_allclose(turnover, np.array([0.0, 0.4, 0.2]))
    np.testing.assert_allclose(costs, 0.001 * turnover)
    np.testing.assert_allclose(net, gross - costs)


def test_pm_stock_enhancer_gated_deducts_sleeve_turnover_fee():
    r_pm = np.array([0.01, 0.01, 0.01, 0.01], dtype=float)
    r_spy = np.array([0.10, 0.10, -0.10, -0.10], dtype=float)

    gross, _hedge, alpha = combine_pm_stock_enhancer_gated(
        r_pm,
        r_spy,
        hedge_weight=0.2,
        momentum_gate_window=1,
        fee_rate=0.0,
    )
    net, _hedge, _alpha = combine_pm_stock_enhancer_gated(
        r_pm,
        r_spy,
        hedge_weight=0.2,
        momentum_gate_window=1,
        fee_rate=0.001,
    )

    np.testing.assert_allclose(net, gross - 0.001 * sleeve_allocation_turnover(alpha))


def test_optimize_hedge_on_window_uses_fee_aware_combined_returns():
    r_pm = np.array([0.01, 0.01, 0.01], dtype=float)
    r_baseline = np.zeros(3, dtype=float)
    r_spy = np.zeros(3, dtype=float)
    r_oil = np.array([0.02, 0.02, 0.02], dtype=float)
    alpha_grid = np.array([0.5], dtype=float)

    result = optimize_hedge_on_window(
        r_pm,
        r_baseline,
        r_spy,
        r_oil,
        np.array([True, True, True]),
        alpha_grid=alpha_grid,
        leg_weight_pairs=((1.0, 0.0),),
        fee_rate=0.001,
    )

    gross = 0.5 * r_pm + 0.5 * r_oil
    expected = gross.copy()
    expected[0] -= 0.001
    np.testing.assert_allclose(result["best_combined_returns"], expected)


def test_kelly_stock_pm_combined_uses_risk_gate_and_sleeve_fees():
    kelly = np.array([0.01, 0.02, 0.03, 0.04], dtype=float)
    equity = np.array([0.00, 0.02, 0.02, -0.01], dtype=float)

    out = combine_kelly_stock_pm_series(
        kelly,
        equity,
        hedge_allocation=0.2,
        pm_in_enhancer_leg=0.85,
        stock_in_enhancer_leg=0.15,
        momentum_window=1,
        fee_rate=0.001,
    )

    expected_alpha = np.array([0.0, 0.0, 0.2, 0.2], dtype=float)
    enhancer = 0.85 * kelly + 0.15 * equity
    gross = (1.0 - expected_alpha) * kelly + expected_alpha * enhancer
    expected_turnover = np.array([0.0, 0.0, 0.4, 0.0], dtype=float)

    np.testing.assert_allclose(out["effective_alpha"], expected_alpha)
    np.testing.assert_allclose(out["sleeve_turnover_l1"], expected_turnover)
    np.testing.assert_allclose(out["combined_return"], gross - 0.001 * expected_turnover)
