import numpy as np

from src.constrained_optimizer import _run_online_pass
from src.baseline import baseline_static_token_weights
from src.pm_risk_overlay import (
    pm_category_spread_returns,
    read_category_correlation_csv,
    resolution_shock_domain_multiplier,
    token_resolution_tilt_multiplier,
    top_negative_correlation_pairs,
)


def test_read_category_correlation_roundtrip_shape():
    import pathlib

    p = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "week11_v2_category_correlation.csv"
    labels, corr = read_category_correlation_csv(p)
    assert len(labels) == corr.shape[0] == corr.shape[1]
    assert np.allclose(corr, corr.T, rtol=0, atol=1e-9)


def test_top_negative_pairs_sorted():
    labels = ["a", "b", "c"]
    corr = np.array([[1.0, -0.5, 0.1], [-0.5, 1.0, -0.01], [0.1, -0.01, 1.0]], dtype=float)
    pairs = top_negative_correlation_pairs(labels, corr, max_pairs=2, max_correlation=-0.002)
    assert len(pairs) == 2
    assert pairs[0][2] <= pairs[1][2]


def test_baseline_static_token_weights_sum_to_one():
    dom = ["a", "a", "b", "c", "c", "c"]
    w = baseline_static_token_weights(dom)
    assert abs(float(np.sum(w)) - 1.0) < 1e-9
    assert w[0] == w[1]


def test_token_resolution_tilt_multiplier_shape():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.02, size=(25, 3)).astype(np.float64)
    m = token_resolution_tilt_multiplier(r, lookback=6, strength=4.0, max_multiplier=2.0)
    assert m.shape == (25, 3) and np.all(m >= 1.0) and np.all(m <= 2.0)


def test_resolution_shock_multiplier_shape():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.02, size=(30, 4)).astype(np.float64)
    dom = ["x", "x", "y", "y"]
    m = resolution_shock_domain_multiplier(r, dom, lookback=8, strength=5.0, max_multiplier=2.0)
    assert m.shape == (30, 2) and np.all(m >= 1.0) and np.all(m <= 2.0)


def test_pm_category_spread_zero_without_pairs():
    r = np.random.default_rng(0).normal(0, 0.01, size=(20, 5))
    dom = ["x", "x", "y", "y", "z"]
    s = pm_category_spread_returns(r, dom, [])
    assert s.shape == (20,) and np.allclose(s, 0.0)


def test_equity_domain_tilt_multiplier_changes_weights():
    rng = np.random.default_rng(1)
    T, n = 40, 4
    r = rng.normal(0, 0.002, size=(T, n)).astype(np.float64)
    domains = ["a", "b", "a", "b"]
    uniq = sorted(set(domains))
    mult = np.ones((T, len(uniq)), dtype=np.float64)
    ia, ib = uniq.index("a"), uniq.index("b")
    mult[:, ia] = 2.0
    out = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.2,
        penalty_lambda=0.1,
        rolling_window=8,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=3,
        evaluation_start_t=8,
        update_after_eval_start=True,
        objective="mean_downside",
        equity_domain_tilt_multiplier=mult,
    )
    out_flat = _run_online_pass(
        returns_matrix=r,
        domains=domains,
        lr=0.2,
        penalty_lambda=0.1,
        rolling_window=8,
        steps_per_window=1,
        domain_limit=0.9,
        max_weight=0.9,
        concentration_penalty_lambda=0.0,
        covariance_penalty_lambda=0.0,
        covariance_shrinkage=0.05,
        entropy_lambda=0.0,
        uniform_mix=0.0,
        seed=3,
        evaluation_start_t=8,
        update_after_eval_start=True,
        objective="mean_downside",
        equity_domain_tilt_multiplier=None,
    )
    assert out["portfolio_returns"].shape == out_flat["portfolio_returns"].shape
    assert not np.allclose(out["portfolio_returns"], out_flat["portfolio_returns"])
