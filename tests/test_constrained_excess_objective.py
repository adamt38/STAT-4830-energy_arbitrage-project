"""Excess-vs-baseline objective and payload baseline_eval_returns."""

import numpy as np

from src.constrained_optimizer import ExperimentConfig, _run_online_pass


def test_excess_objective_requires_baseline():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(80, 5)).astype(np.float64)
    domains = ["a", "a", "b", "b", "c"]
    try:
        _run_online_pass(
            returns_matrix=r,
            domains=domains,
            lr=0.05,
            penalty_lambda=1.0,
            rolling_window=24,
            steps_per_window=1,
            domain_limit=0.5,
            max_weight=0.5,
            concentration_penalty_lambda=1.0,
            covariance_penalty_lambda=1.0,
            covariance_shrinkage=0.05,
            entropy_lambda=0.0,
            uniform_mix=0.1,
            seed=1,
            objective="excess_mean_downside",
            baseline_step_returns=None,
        )
    except ValueError as e:
        assert "requires baseline_step_returns" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_baseline_eval_returns_aligned_with_portfolio():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(60, 4)).astype(np.float64)
    b = rng.normal(0, 0.008, size=(60,)).astype(np.float64)
    domains = ["a", "a", "b", "b"]
    cfg = ExperimentConfig()
    out = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.05,
        penalty_lambda=1.0,
        rolling_window=12,
        steps_per_window=1,
        domain_limit=0.6,
        max_weight=0.45,
        concentration_penalty_lambda=5.0,
        covariance_penalty_lambda=1.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.01,
        uniform_mix=0.05,
        seed=2,
        evaluation_start_t=40,
        update_after_eval_start=True,
        objective="excess_mean_downside",
        baseline_step_returns=b,
        macro_geopolitics_boost_lambda=0.0,
        macro_target_domains=frozenset(),
    )
    pr = out["portfolio_returns"]
    be = out["baseline_eval_returns"]
    assert pr.size == be.size
    assert be.size > 0
    assert pr.shape == be.shape


def test_sortino_numpy_finite():
    from src.constrained_optimizer import _sortino_numpy

    x = np.array([0.01, -0.002, 0.003], dtype=float)
    s = _sortino_numpy(x)
    assert np.isfinite(s)


def test_token_resolution_multiplier_runs():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, size=(44, 3)).astype(np.float64)
    domains = ["a", "b", "c"]
    from src.pm_risk_overlay import token_resolution_tilt_multiplier

    tok = token_resolution_tilt_multiplier(r, lookback=8, strength=3.0, max_multiplier=1.8)
    out = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.12,
        penalty_lambda=0.5,
        rolling_window=10,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=8,
        evaluation_start_t=10,
        update_after_eval_start=True,
        objective="mean_return",
        token_resolution_multiplier=tok,
    )
    assert out["portfolio_returns"].size > 0


def test_eval_baseline_weight_mix_moves_returns_toward_baseline():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, size=(45, 3)).astype(np.float64)
    domains = ["a", "b", "c"]
    from src.baseline import baseline_static_token_weights

    w0 = baseline_static_token_weights(domains)
    out0 = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.15,
        penalty_lambda=0.5,
        rolling_window=10,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=6,
        evaluation_start_t=10,
        update_after_eval_start=True,
        objective="mean_return",
        baseline_static_weights=w0,
        eval_baseline_weight_mix=0.0,
    )
    out1 = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.15,
        penalty_lambda=0.5,
        rolling_window=10,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=6,
        evaluation_start_t=10,
        update_after_eval_start=True,
        objective="mean_return",
        baseline_static_weights=w0,
        eval_baseline_weight_mix=0.85,
    )
    assert out0["portfolio_returns"].shape == out1["portfolio_returns"].shape
    assert not np.allclose(out0["portfolio_returns"], out1["portfolio_returns"])


def test_mean_return_objective_runs():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, size=(50, 3)).astype(np.float64)
    domains = ["a", "b", "c"]
    out = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.1,
        penalty_lambda=0.5,
        rolling_window=10,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=4,
        evaluation_start_t=10,
        update_after_eval_start=True,
        objective="mean_return",
        baseline_step_returns=None,
    )
    assert out["portfolio_returns"].size > 0
