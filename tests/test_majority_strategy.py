"""Tests for the fee-aware Majority strategy."""

from __future__ import annotations

import numpy as np

from src.majority_strategy import _compute_majority_side_path


def test_majority_strategy_uses_current_majority_side_without_lookahead():
    prices = np.array(
        [
            [0.60, 0.40],
            [0.66, 0.30],
        ],
        dtype=float,
    )
    weights = np.array([0.5, 0.5], dtype=float)

    net, gross, turnover, yes_w, no_w = _compute_majority_side_path(prices, weights)

    expected_yes_return = 0.66 / 0.60 - 1.0
    expected_no_return = (1.0 - 0.30) / (1.0 - 0.40) - 1.0
    np.testing.assert_allclose(gross, np.array([0.5 * expected_yes_return + 0.5 * expected_no_return]))
    np.testing.assert_allclose(net, gross)
    np.testing.assert_allclose(turnover, np.array([1.0]))
    np.testing.assert_allclose(yes_w, np.array([0.5]))
    np.testing.assert_allclose(no_w, np.array([0.5]))


def test_majority_strategy_charges_turnover_when_side_flips():
    prices = np.array(
        [
            [0.60],
            [0.40],
            [0.30],
        ],
        dtype=float,
    )
    weights = np.array([1.0], dtype=float)

    net, gross, turnover, yes_w, no_w = _compute_majority_side_path(
        prices,
        weights,
        fee_rate=0.001,
    )

    np.testing.assert_allclose(yes_w, np.array([1.0, 0.0]))
    np.testing.assert_allclose(no_w, np.array([0.0, 1.0]))
    np.testing.assert_allclose(turnover, np.array([1.0, 2.0]))
    np.testing.assert_allclose(net, gross - 0.001 * turnover)
