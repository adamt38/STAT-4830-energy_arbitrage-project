"""Tests for macro_integration modes in constrained online optimizer."""

from __future__ import annotations

import numpy as np

from src.constrained_optimizer import _run_online_pass


def _synthetic_returns(n_steps: int = 14, n_assets: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=(n_steps, n_assets)).astype(np.float64)


def _aligned_exog(n_steps: int, risk_z: float = 3.0) -> tuple[np.ndarray, ...]:
    rz = np.full(n_steps, risk_z, dtype=float)
    ez = np.zeros(n_steps, dtype=float)
    ratesz = np.zeros(n_steps, dtype=float)
    st = np.zeros(n_steps, dtype=float)
    op = np.ones(n_steps, dtype=float)
    return rz, ez, ratesz, st, op


def _domains(n: int) -> list[str]:
    return [f"d{i % 2}" for i in range(n)]


def _base_kwargs(returns: np.ndarray, domains: list[str], **overrides):
    n = returns.shape[0]
    rz, ez, ratesz, st, op = _aligned_exog(n)
    kw = dict(
        returns_matrix=returns,
        domains=domains,
        lr=0.05,
        penalty_lambda=0.1,
        rolling_window=5,
        steps_per_window=2,
        domain_limit=0.5,
        max_weight=0.35,
        concentration_penalty_lambda=5.0,
        covariance_penalty_lambda=1.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.1,
        seed=7,
        evaluation_start_t=8,
        update_after_eval_start=True,
        objective="mean_downside",
        variance_penalty=1.0,
        downside_penalty=2.0,
        optimizer_type="adam",
        weight_parameterization="softmax",
        regime_k=2.0,
        risk_on_z=rz,
        energy_z=ez,
        rates_z=ratesz,
        exog_is_stale=st,
        is_equity_open=op,
        time_offset=0,
    )
    kw.update(overrides)
    return kw


def test_rescale_lambda_zero_is_deterministic():
    returns = _synthetic_returns()
    domains = _domains(returns.shape[1])
    kw = _base_kwargs(
        returns,
        domains,
        macro_integration="rescale",
        lambda_macro_explicit=0.0,
    )
    a = _run_online_pass(**kw)
    b = _run_online_pass(**kw)
    np.testing.assert_array_almost_equal(a["portfolio_returns"], b["portfolio_returns"])
    np.testing.assert_array_almost_equal(a["avg_weights"], b["avg_weights"])


def test_explicit_differs_from_rescale_with_strong_regime_k():
    returns = _synthetic_returns(seed=99)
    domains = _domains(returns.shape[1])
    base = _base_kwargs(returns, domains, regime_k=5.0)
    r = _run_online_pass(**base, macro_integration="rescale", lambda_macro_explicit=0.0)
    e = _run_online_pass(**base, macro_integration="explicit", lambda_macro_explicit=0.0)
    assert not np.allclose(r["avg_weights"], e["avg_weights"], atol=1e-6) or not np.allclose(
        r["portfolio_returns"], e["portfolio_returns"], atol=1e-6
    )


def test_explicit_j_macro_lambda_changes_weights():
    returns = _synthetic_returns(seed=101)
    domains = _domains(returns.shape[1])
    base = _base_kwargs(returns, domains, regime_k=0.1, max_weight=0.2)
    z0 = _run_online_pass(**base, macro_integration="explicit", lambda_macro_explicit=0.0)
    z1 = _run_online_pass(**base, macro_integration="explicit", lambda_macro_explicit=15.0)
    assert not np.allclose(z0["avg_weights"], z1["avg_weights"], atol=1e-5)
