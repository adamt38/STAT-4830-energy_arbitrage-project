"""Tests for baseline vs hedge alignment (no matplotlib files written)."""

import pathlib

import pytest

from src.baseline_vs_hedge_figures import align_baseline_to_hedge_rows


def test_align_baseline_to_hedge_rows_order_and_filter():
    baseline = [
        {"timestamp": "1", "portfolio_return": "0.1", "cumulative_return": "1.1", "drawdown": "0"},
        {"timestamp": "2", "portfolio_return": "0.0", "cumulative_return": "1.1", "drawdown": "-0.01"},
        {"timestamp": "3", "portfolio_return": "-0.05", "cumulative_return": "1.045", "drawdown": "-0.05"},
    ]
    hedge = [
        {"timestamp": "2", "combined_return": "0.01", "cumulative_combined": "1.0", "drawdown_combined": "0"},
        {"timestamp": "3", "combined_return": "0.02", "cumulative_combined": "1.02", "drawdown_combined": "0"},
    ]
    b, h = align_baseline_to_hedge_rows(baseline, hedge)
    assert len(b) == len(h) == 2
    assert [int(x["timestamp"]) for x in b] == [2, 3]
    assert float(b[0]["portfolio_return"]) == 0.0


def test_make_figures_requires_files(tmp_path: pathlib.Path):
    from src.baseline_vs_hedge_figures import make_baseline_vs_hedge_figures

    with pytest.raises(FileNotFoundError):
        make_baseline_vs_hedge_figures(tmp_path, artifact_prefix="missing")
