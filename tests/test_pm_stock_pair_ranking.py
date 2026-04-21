import math

from src.pm_stock_pair_ranking import pfcs


def test_pfcs_zero_small_n():
    assert pfcs(0.9, n=2) == 0.0


def test_pfcs_increases_with_n():
    s10 = pfcs(0.5, n=10)
    s100 = pfcs(0.5, n=100)
    assert s100 > s10 > 0.0


def test_pfcs_monotone_in_abs_r():
    assert pfcs(0.8, n=50) > pfcs(0.3, n=50)
