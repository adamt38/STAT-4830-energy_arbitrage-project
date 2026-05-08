"""Optimize PM–hedge capital split (and optional oil vs −equity leg mix) on a masked return window."""

from __future__ import annotations

import numpy as np

from src.stock_oil_hedge import combine_sleeves, combine_sleeves_variable


def cap_gain(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) - 1.0) if r.size else 0.0


def growth_of_one(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r)) if r.size else 1.0


def excess_gain_vs_baseline(r_portfolio: np.ndarray, r_baseline: np.ndarray, mask: np.ndarray) -> float:
    """Compound gain on portfolio minus compound gain on baseline over masked steps."""
    return cap_gain(r_portfolio[mask]) - cap_gain(r_baseline[mask])


def rolling_mean_past_only(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean using **strictly past** bars (excludes current index) to avoid same-bar look-ahead."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    out = np.zeros(n, dtype=float)
    w = max(1, int(window))
    for i in range(n):
        lo = max(0, i - w)
        seg = x[lo:i]
        out[i] = float(np.mean(seg)) if seg.size > 0 else 0.0
    return out


def pm_stock_enhancer_leg_returns(
    r_pm: np.ndarray,
    r_spy: np.ndarray,
    *,
    pm_weight: float = 0.85,
    stock_weight: float = 0.15,
) -> np.ndarray:
    """Hedge sleeve = mostly optimized PM return + small long stock (lagged SPY) as signal enhancer."""
    p = float(max(pm_weight, 0.0))
    s = float(max(stock_weight, 0.0))
    denom = p + s
    if denom <= 1e-12:
        return np.zeros_like(r_pm, dtype=float)
    return (p / denom) * np.asarray(r_pm, dtype=float) + (s / denom) * np.asarray(r_spy, dtype=float)


def momentum_gated_alpha_bar(
    hedge_weight: float,
    rolling_signal_mean: np.ndarray,
) -> np.ndarray:
    """Per bar: full ``hedge_weight`` only when rolling mean signal > 0, else 0 (no hedge)."""
    alpha = float(np.clip(hedge_weight, 0.0, 1.0))
    sig = np.asarray(rolling_signal_mean, dtype=float)
    hedge_active = sig > 0.0
    return np.where(hedge_active, alpha, 0.0)


def combine_pm_stock_enhancer_gated(
    r_pm: np.ndarray,
    r_spy: np.ndarray,
    *,
    hedge_weight: float = 0.07,
    pm_in_hedge_leg: float = 0.85,
    stock_in_hedge_leg: float = 0.15,
    momentum_gate_window: int = 48,
    signal_for_gate: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend PM with PM/stock enhancer sleeve; α is momentum-gated on lagged SPY trend.

    Returns ``(combined_returns, hedge_leg_returns, effective_alpha_per_bar)``.
    """
    r_pm = np.asarray(r_pm, dtype=float)
    r_spy = np.asarray(r_spy, dtype=float)
    if r_pm.shape != r_spy.shape:
        raise ValueError("r_pm and r_spy must match shape.")
    r_hedge = pm_stock_enhancer_leg_returns(
        r_pm, r_spy, pm_weight=pm_in_hedge_leg, stock_weight=stock_in_hedge_leg
    )
    sig = r_spy if signal_for_gate is None else np.asarray(signal_for_gate, dtype=float)
    roll = rolling_mean_past_only(sig, momentum_gate_window)
    alpha_bar = momentum_gated_alpha_bar(hedge_weight, roll)
    combined = combine_sleeves_variable(r_pm, r_hedge, alpha_bar)
    return combined, r_hedge, alpha_bar


def static_hedge_leg_returns(
    r_oil: np.ndarray,
    r_spy: np.ndarray,
    oil_leg_weight: float,
    inverse_equity_leg_weight: float,
) -> np.ndarray:
    """Match static sleeve: normalized oil on r_oil plus −equity on r_spy (no analyst tilt)."""
    o = float(max(oil_leg_weight, 0.0))
    e = float(max(inverse_equity_leg_weight, 0.0))
    s = o + e
    if s <= 1e-12:
        return np.zeros_like(r_oil, dtype=float)
    return (o / s) * np.asarray(r_oil, dtype=float) - (e / s) * np.asarray(r_spy, dtype=float)


def optimize_hedge_on_window(
    r_pm: np.ndarray,
    r_baseline: np.ndarray,
    r_spy: np.ndarray,
    r_oil: np.ndarray,
    mask: np.ndarray,
    *,
    alpha_grid: np.ndarray,
    leg_weight_pairs: tuple[tuple[float, float], ...] = ((0.5, 0.5),),
) -> dict[str, float | tuple[float, float] | np.ndarray]:
    """Maximize excess compound return vs baseline on ``mask`` over α and optional (oil, −eq) pairs.

    Combined return per bar: (1−α)*r_pm + α*r_hedge, with r_hedge from static_hedge_leg_returns.
    """
    if r_pm.shape != r_baseline.shape or r_pm.shape != r_spy.shape or r_pm.shape != r_oil.shape:
        raise ValueError("r_pm, r_baseline, r_spy, r_oil must have the same shape.")
    if mask.shape != r_pm.shape:
        raise ValueError("mask must match return vector length.")
    if not np.any(mask):
        raise ValueError("mask selects no steps.")

    best_excess = float("-inf")
    best_alpha = 0.0
    best_pair = leg_weight_pairs[0]
    best_combined = r_pm.copy()

    alphas = np.clip(np.asarray(alpha_grid, dtype=float), 0.0, 1.0)
    for wo, we in leg_weight_pairs:
        r_hedge = static_hedge_leg_returns(r_oil, r_spy, wo, we)
        for a in alphas:
            rc = combine_sleeves(r_pm, r_hedge, float(a))
            ex = excess_gain_vs_baseline(rc, r_baseline, mask)
            if ex > best_excess:
                best_excess = ex
                best_alpha = float(a)
                best_pair = (float(wo), float(we))
                best_combined = rc

    return {
        "best_hedge_allocation_alpha": best_alpha,
        "best_oil_leg_weight": best_pair[0],
        "best_inverse_equity_leg_weight": best_pair[1],
        "excess_gain_vs_baseline_window": best_excess,
        "baseline_window_gain": cap_gain(r_baseline[mask]),
        "best_combined_window_gain": cap_gain(best_combined[mask]),
        "best_combined_returns": best_combined,
    }
