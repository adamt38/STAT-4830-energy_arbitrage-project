"""Tests for ETF/defensive risk regime → concentration mapping."""

from src.equity_signal import get_dynamic_concentration_params


def test_get_dynamic_concentration_params_risk_on() -> None:
    p = get_dynamic_concentration_params(0.51)
    assert p["max_domain_exposure_threshold"] == 0.18
    assert p["concentration_penalty_lambda"] == 10.0


def test_get_dynamic_concentration_params_risk_off() -> None:
    p = get_dynamic_concentration_params(-0.51)
    assert p["max_domain_exposure_threshold"] == 0.06
    assert p["concentration_penalty_lambda"] == 80.0


def test_get_dynamic_concentration_params_neutral_mid() -> None:
    p = get_dynamic_concentration_params(0.0)
    assert p["max_domain_exposure_threshold"] == 0.12
    assert p["concentration_penalty_lambda"] == 32.0


def test_get_dynamic_concentration_params_boundary_inclusive_neutral() -> None:
    p_hi = get_dynamic_concentration_params(0.5)
    assert p_hi["max_domain_exposure_threshold"] == 0.12
    p_lo = get_dynamic_concentration_params(-0.5)
    assert p_lo["max_domain_exposure_threshold"] == 0.12
