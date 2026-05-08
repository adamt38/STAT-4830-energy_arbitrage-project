"""Tests for hedge vs risk-free experiment helpers (no network)."""

import numpy as np

from src.hedge_riskfree_experiment import total_capital_gain


def test_total_capital_gain_empty():
    assert total_capital_gain(np.array([], dtype=float)) == 0.0


def test_total_capital_gain_simple():
    r = np.array([0.1, -0.05, 0.02], dtype=float)
    expected = (1.1 * 0.95 * 1.02) - 1.0
    assert abs(total_capital_gain(r) - expected) < 1e-12
