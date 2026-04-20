"""Dynamic-Copula End-to-End Kelly OGD optimizer for Polymarket portfolios.

This module replaces the mean-variance / Sortino objective in
``constrained_optimizer.py`` with an expected-log-wealth (Kelly) objective
evaluated by Monte-Carlo sampling from a dynamic Gaussian copula. The
copula's correlation matrix is generated at every time step by a small
MLP that consumes exogenous macro features (SPY / QQQ / BTC log returns).

Portfolio weights ``w`` and MLP parameters ``theta`` are optimized
*simultaneously* by Adam-OGD on the projected simplex with an L1
turnover penalty ``lambda * ||w_t - w_{t-1}||_1``. The bilinear
coupling ``w^T payoff(y(theta))``, the standard-normal CDF, the
straight-through Bernoulli sampler, and the dynamic shrinkage of the
correlation matrix all break convexity, yielding the strictly
non-convex joint problem required by the project specification.

The module exposes a small, faithful subset of the constrained-optimizer
public API:

- :class:`KellyExperimentConfig` (frozen dataclass of hyperparameter grids)
- :class:`DynamicCopulaMLP` (PyTorch module mapping features -> R_t)
- :func:`kelly_log_wealth` (vectorized Monte-Carlo Kelly objective)
- :func:`split_index_for_returns`
- :func:`run_kelly_optuna_search` (Optuna QMC search; writes
  ``{prefix}_kelly_*`` artifacts under ``data/processed``)

Outputs (for a stem ``S = output_artifact_prefix``):

- ``S_kelly_experiment_grid.csv``     — one row per Optuna trial.
- ``S_kelly_best_metrics.json``       — selected hyperparameters + holdout stats.
- ``S_kelly_best_timeseries.csv``     — per-step holdout return / log-wealth / turnover.
- ``S_kelly_best_weights.csv``        — per-step holdout weight matrix.
- ``S_kelly_copula_diagnostics.json`` — average correlation, PD-fallback rates, etc.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch

from src.baseline import _build_price_matrix, _read_csv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KellyExperimentConfig:
    """Hyperparameter grids for the dynamic-copula Kelly OGD search.

    The grids are interpreted as ``[min, max]`` ranges by Optuna's QMC
    sampler (matching the convention in
    :class:`src.constrained_optimizer.ExperimentConfig`).

    Attributes
    ----------
    learning_rates_w
        Adam learning rate for the simplex weight parameters ``w``.
    learning_rates_theta
        Adam learning rate for the dynamic copula MLP parameters.
    rolling_windows
        Categorical choices for the look-back window ``W`` (in steps).
    mc_samples
        Categorical choices for ``S``, the Monte-Carlo sample count.
    turnover_lambdas
        L1 turnover penalty coefficients ``lambda_turn``.
    copula_shrinkages
        Off-diagonal shrinkage toward the identity (PD-stabilizer).
    copula_temperatures
        Temperature ``tau`` of the straight-through Bernoulli surrogate.
    mlp_hidden_dims
        Width of the single hidden layer in :class:`DynamicCopulaMLP`.
    concentration_penalty_lambdas
        Quadratic ReLU penalty on weight excess above ``max_weight``.
    max_weights
        Per-asset cap entering the concentration penalty.
    """

    learning_rates_w: tuple[float, ...] = (0.01, 0.02, 0.05)
    learning_rates_theta: tuple[float, ...] = (1e-3, 5e-3, 1e-2)
    rolling_windows: tuple[int, ...] = (24, 48, 96)
    mc_samples: tuple[int, ...] = (256, 512, 1024)
    turnover_lambdas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1)
    copula_shrinkages: tuple[float, ...] = (0.05, 0.1, 0.25)
    copula_temperatures: tuple[float, ...] = (0.05, 0.1, 0.25)
    mlp_hidden_dims: tuple[int, ...] = (8, 16, 32)
    concentration_penalty_lambdas: tuple[float, ...] = (10.0, 50.0)
    max_weights: tuple[float, ...] = (0.06, 0.10, 0.15)
    steps_per_window: int = 3
    holdout_fraction: float = 0.2
    walkforward_train_steps: int = 240
    walkforward_test_steps: int = 48
    weight_parameterization: str = "projected_simplex"
    optimizer_type: str = "adam"
    seed: int = 7
    optuna_n_jobs: int = 1
    exogenous_feature_columns: tuple[str, ...] = (
        "spy_ret_1",
        "qqq_ret_1",
        "btc_usd_ret_1",
    )
    #: When True, R_t = I at every step (independence baseline). The MLP is
    #: still constructed for code-path uniformity but its outputs are unused.
    disable_copula: bool = False
    #: Selection objective subtracts ``lambda * mean_turnover`` so high-churn
    #: solutions are penalized at trial-selection time too. Set to 0.0 to
    #: disable (Kelly-only selection).
    selection_turnover_penalty: float = 0.05
    #: PD fallback hard ceiling (we keep doubling shrinkage until R is PD or
    #: alpha exceeds this; final fallback is R = I).
    max_pd_fallback_shrinkage: float = 0.95


class KellyOnlinePassPayload(TypedDict):
    """Return type from :func:`_run_kelly_online_pass`."""

    portfolio_returns: np.ndarray
    log_wealth_increments: np.ndarray
    turnovers: np.ndarray
    eval_weights: np.ndarray
    avg_weights: np.ndarray
    avg_correlation_matrix: np.ndarray
    pd_fallback_count: int
    correlation_off_diag_abs_mean: float


# ---------------------------------------------------------------------------
# Helpers shared with constrained_optimizer (re-implemented locally so this
# module has no cross-engine import coupling).
# ---------------------------------------------------------------------------


def _compute_returns(price_matrix: np.ndarray) -> np.ndarray:
    """Per-step simple returns; pre-listing rows that contain NaN are kept."""
    if price_matrix.shape[0] < 2:
        return np.array([], dtype=float)
    prev = price_matrix[:-1]
    nxt = price_matrix[1:]
    return (nxt - prev) / np.clip(prev, 1e-6, None)


def _project_to_simplex(weights: torch.Tensor) -> torch.Tensor:
    """Standard sort-based Euclidean projection onto the probability simplex."""
    if weights.numel() == 0:
        return weights
    sorted_vals, _ = torch.sort(weights, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    idx = torch.arange(1, weights.numel() + 1, device=weights.device, dtype=weights.dtype)
    cond = sorted_vals - (cumsum - 1.0) / idx > 0
    if not torch.any(cond):
        return torch.full_like(weights, 1.0 / float(weights.numel()))
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return torch.clamp(weights - theta, min=0.0)


def _project_parameter_vector_inplace(
    params: torch.Tensor, available_mask: torch.Tensor
) -> None:
    """Project the *active* coordinates of ``params`` onto the simplex in-place."""
    with torch.no_grad():
        active_idx = torch.nonzero(available_mask > 0.0, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            params.zero_()
            return
        active_vals = params[active_idx]
        projected_active = _project_to_simplex(active_vals)
        params.zero_()
        params[active_idx] = projected_active


def _parameterized_weights(
    params: torch.Tensor, available_mask: torch.Tensor
) -> torch.Tensor:
    """Project-and-renormalize active params onto the simplex (differentiable)."""
    masked = torch.clamp(params, min=0.0) * available_mask
    masked_sum = torch.sum(masked)
    if float(masked_sum.detach().cpu().item()) <= 0.0:
        masked = available_mask
        masked_sum = torch.sum(masked)
    return masked / torch.clamp(masked_sum, min=1e-8)


def split_index_for_returns(n_returns: int, cfg: KellyExperimentConfig) -> int:
    """First holdout index; same rule as constrained_optimizer.split_index_for_returns."""
    if n_returns <= 0:
        return 0
    split_idx = max(int((1.0 - cfg.holdout_fraction) * n_returns), cfg.walkforward_train_steps)
    return min(split_idx, n_returns - 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_kelly_inputs(
    project_root: pathlib.Path,
    artifact_prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load price matrix, return matrix, kept tokens, and per-token domains.

    Returns
    -------
    price_matrix : (T, K) array of Yes-token prices (current implied probabilities).
    returns_matrix : (T-1, K) array of per-step simple returns.
    kept_tokens : list of length K
    domains : list of length K aligned to kept_tokens
    """
    processed = project_root / "data" / "processed"
    markets_rows = _read_csv(processed / f"{artifact_prefix}_markets_filtered.csv")
    history_rows = _read_csv(processed / f"{artifact_prefix}_price_history.csv")
    _, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)
    token_to_domain = {row["yes_token_id"]: row.get("domain", "other") for row in markets_rows}
    domains = [token_to_domain.get(t, "other") for t in kept_tokens]
    return price_matrix, returns_matrix, kept_tokens, domains


def _load_exogenous_feature_matrix(
    project_root: pathlib.Path,
    artifact_prefix: str,
    n_steps: int,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """Load aligned exogenous feature matrix of shape (n_steps, d).

    Missing CSV / row-count mismatch / missing columns -> all-zero matrix
    (the MLP then degenerates to a constant-correlation copula).
    """
    d = len(feature_columns)
    if n_steps <= 0:
        return np.zeros((0, d), dtype=np.float64)
    path = project_root / "data" / "processed" / f"{artifact_prefix}_exogenous_features.csv"
    if not path.exists():
        return np.zeros((n_steps, d), dtype=np.float64)
    try:
        df = pd.read_csv(path)
    except OSError:
        return np.zeros((n_steps, d), dtype=np.float64)
    if len(df) != n_steps:
        return np.zeros((n_steps, d), dtype=np.float64)
    out = np.zeros((n_steps, d), dtype=np.float64)
    for j, col in enumerate(feature_columns):
        if col not in df.columns:
            continue
        out[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Dynamic copula MLP
# ---------------------------------------------------------------------------


class DynamicCopulaMLP(torch.nn.Module):
    """MLP mapping exogenous features -> strict upper triangle of R_t.

    Architecture: ``d -> hidden -> tanh -> hidden -> tanh -> K(K-1)/2``
    followed by ``0.95 * tanh(.)`` so each off-diagonal lies in
    ``(-0.95, 0.95)``. The final correlation matrix is shrunk toward
    the identity by ``alpha`` and PD-stabilized via Cholesky retry.

    Parameters
    ----------
    n_features : int
        Number of exogenous inputs (typically 3: SPY, QQQ, BTC log returns).
    n_assets : int
        Number of Polymarket assets ``K``.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(self, n_features: int, n_assets: int, hidden_dim: int = 16):
        super().__init__()
        self.n_features = int(n_features)
        self.n_assets = int(n_assets)
        self.hidden_dim = int(hidden_dim)
        self.n_off_diag = self.n_assets * (self.n_assets - 1) // 2
        self.fc1 = torch.nn.Linear(self.n_features, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.head = torch.nn.Linear(self.hidden_dim, max(self.n_off_diag, 1))
        # Initialize the head small so initial correlations are near-zero
        # (closer to independence) and Cholesky never fails out of the box.
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.05)
        # Pre-cache upper-triangle indices (excluding diagonal).
        triu = torch.triu_indices(self.n_assets, self.n_assets, offset=1)
        self.register_buffer("_triu_row", triu[0])
        self.register_buffer("_triu_col", triu[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the strict upper triangle vector (length K(K-1)/2) in (-0.95, 0.95)."""
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return 0.95 * torch.tanh(self.head(h))

    def correlation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Build the K x K correlation matrix from features ``x`` (shape (d,))."""
        if self.n_assets <= 1:
            return torch.eye(self.n_assets, dtype=x.dtype, device=x.device)
        off = self.forward(x)
        R = torch.eye(self.n_assets, dtype=x.dtype, device=x.device)
        # Place the off-diagonals symmetrically.
        R = R.clone()
        R[self._triu_row, self._triu_col] = off
        R[self._triu_col, self._triu_row] = off
        return R


def cholesky_safe(
    R: torch.Tensor,
    base_shrinkage: float,
    max_shrinkage: float = 0.95,
) -> tuple[torch.Tensor, float, bool]:
    """Return (L, alpha_used, fallback_fired).

    Tries ``alpha = base_shrinkage, 2*base_shrinkage, 4*..., max_shrinkage, 1.0``
    and returns the first ``L`` such that ``L L^T = (1-alpha) R + alpha I`` is PSD.
    ``alpha = 1.0`` reduces to the identity so we always return *something*.
    """
    K = R.shape[0]
    eye = torch.eye(K, dtype=R.dtype, device=R.device)
    alpha = max(float(base_shrinkage), 0.0)
    fallback_fired = False
    for _ in range(8):
        R_a = (1.0 - alpha) * R + alpha * eye
        try:
            L = torch.linalg.cholesky(R_a)
            return L, alpha, fallback_fired
        except RuntimeError:
            fallback_fired = True
            if alpha <= 0.0:
                alpha = 0.05
            else:
                alpha = min(2.0 * alpha, max_shrinkage)
    # Final safety: independence copula.
    return eye.clone(), 1.0, True


# ---------------------------------------------------------------------------
# Core Kelly objective (vectorized Monte-Carlo)
# ---------------------------------------------------------------------------


_SQRT2 = math.sqrt(2.0)


def _standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Differentiable Phi(z) using torch.erf."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def kelly_log_wealth(
    weights: torch.Tensor,
    prices_t: torch.Tensor,
    L_t: torch.Tensor,
    eps: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Vectorized Monte-Carlo expected log-wealth at one time step.

    Parameters
    ----------
    weights : (K,) tensor on the simplex.
    prices_t : (K,) tensor of current implied probabilities (Yes-token prices).
    L_t : (K, K) Cholesky factor of the copula correlation matrix.
    eps : (S, K) standard-normal noise (already reparameterized).
    temperature : float, surrogate sigmoid temperature for the
        straight-through Bernoulli sampler.

    Returns
    -------
    Scalar tensor: ``(1/S) sum_s log(1 + clamp(w^T payoff_s, -0.999, +inf))``.
    """
    # Latent Gaussian samples with copula correlation: z = eps @ L^T -> (S, K)
    z = eps @ L_t.transpose(0, 1)
    u = _standard_normal_cdf(z)  # (S, K) uniform marginals
    # Straight-through Bernoulli: forward = hard threshold, backward = sigmoid surrogate.
    p = prices_t.unsqueeze(0)  # (1, K)
    y_soft = torch.sigmoid((p - u) / max(temperature, 1e-6))
    y_hard = (u <= p).to(dtype=u.dtype)
    y = y_hard + (y_soft - y_soft.detach())  # (S, K), STE
    # Per-share Polymarket payoff on a Yes-token bought at price p:
    #   y=1 -> +(1 - p),   y=0 -> -p
    payoff = y * (1.0 - p) + (1.0 - y) * (-p)  # (S, K)
    rho = payoff @ weights  # (S,)
    # Clamp from below so log1p never sees -1.
    rho_clamped = torch.clamp(rho, min=-0.999)
    return torch.mean(torch.log1p(rho_clamped))


# ---------------------------------------------------------------------------
# Joint OGD pass (weights + MLP simultaneously)
# ---------------------------------------------------------------------------


def _run_kelly_online_pass(
    *,
    returns_matrix: np.ndarray,
    price_matrix: np.ndarray,
    exogenous_matrix: np.ndarray,
    cfg: KellyExperimentConfig,
    lr_w: float,
    lr_theta: float,
    rolling_window: int,
    mc_samples: int,
    turnover_lambda: float,
    copula_shrinkage: float,
    copula_temperature: float,
    mlp_hidden_dim: int,
    concentration_penalty_lambda: float,
    max_weight: float,
    seed: int,
    evaluation_start_t: int | None = None,
    update_after_eval_start: bool = True,
    time_offset: int = 0,
    capture_diagnostics: bool = False,
    mlp: DynamicCopulaMLP | None = None,
) -> KellyOnlinePassPayload:
    """Mirrors ``_run_online_pass`` from constrained_optimizer but optimizes
    weights AND MLP parameters jointly under the Kelly objective.

    The ``returns_matrix`` is used **only** to produce realized portfolio
    returns at evaluation steps (and to decide when an asset is available).
    The Kelly objective itself is computed from MC-sampled binary outcomes
    drawn from the dynamic copula at the *current* implied probabilities
    in ``price_matrix``.

    ``returns_matrix`` rows correspond to indices ``[1, 2, ..., T-1]`` of
    ``price_matrix`` (the i-th return moves price[i] -> price[i+1]). At
    online step ``t`` (0-indexed in the segment), we therefore use:

    - past window: returns_matrix[t - rolling_window : t]
    - implied probabilities now: price_matrix[t + 1] (post-return-t price,
      which is the price you would buy at *before* the next step)
    - exogenous features now: exogenous_matrix[t]
    """
    torch.manual_seed(int(seed))
    n_returns, n_assets = returns_matrix.shape
    if n_returns <= rolling_window or n_assets == 0:
        empty_w = np.zeros(n_assets, dtype=float)
        return KellyOnlinePassPayload(
            portfolio_returns=np.array([], dtype=float),
            log_wealth_increments=np.array([], dtype=float),
            turnovers=np.array([], dtype=float),
            eval_weights=np.zeros((0, n_assets), dtype=float),
            avg_weights=empty_w,
            avg_correlation_matrix=np.zeros((n_assets, n_assets), dtype=float),
            pd_fallback_count=0,
            correlation_off_diag_abs_mean=0.0,
        )

    eval_start = evaluation_start_t if evaluation_start_t is not None else rolling_window

    # Initialize weights uniformly on the simplex (matches projected_simplex
    # initialization in constrained_optimizer).
    params = torch.full(
        (n_assets,), 1.0 / float(n_assets), dtype=torch.float32, requires_grad=True
    )

    n_features = exogenous_matrix.shape[1] if exogenous_matrix.size else 0
    if mlp is None:
        mlp = DynamicCopulaMLP(
            n_features=max(n_features, 1), n_assets=n_assets, hidden_dim=mlp_hidden_dim
        )
    mlp.train()

    optimizer = torch.optim.Adam(
        [
            {"params": [params], "lr": float(lr_w)},
            {"params": list(mlp.parameters()), "lr": float(lr_theta)},
        ]
    )

    realized_returns: list[float] = []
    log_wealth_increments: list[float] = []
    turnovers: list[float] = []
    eval_weights: list[np.ndarray] = []

    prev_weights_eval = np.full(n_assets, 1.0 / n_assets, dtype=float)
    prev_weights_train = torch.full(
        (n_assets,), 1.0 / float(n_assets), dtype=torch.float32
    )

    # Diagnostics
    R_accum = torch.zeros(n_assets, n_assets, dtype=torch.float32)
    R_count = 0
    pd_fallback_count = 0

    for t in range(rolling_window, n_returns):
        step_returns_np = np.array(returns_matrix[t], dtype=float)
        available_mask_np = np.isfinite(step_returns_np)
        if not np.any(available_mask_np):
            continue
        available_mask = torch.tensor(
            available_mask_np.astype(np.float32), dtype=torch.float32
        )

        # Implied probabilities at "now" = price_matrix[t + 1]; clipped to (eps, 1-eps)
        # to keep payoffs bounded and log1p well-defined.
        if t + 1 < price_matrix.shape[0]:
            p_now_np = np.array(price_matrix[t + 1], dtype=float)
        else:
            p_now_np = np.array(price_matrix[-1], dtype=float)
        p_now_np = np.where(np.isfinite(p_now_np), p_now_np, 0.5)
        p_now_np = np.clip(p_now_np, 1e-3, 1.0 - 1e-3)
        prices_t = torch.tensor(p_now_np.astype(np.float32), dtype=torch.float32)

        # Mask unavailable assets to a neutral 0.5 so the Bernoulli draw is
        # symmetric (and the payoff term contributes nothing because the
        # weight is forced to 0 by the simplex projection on `available_mask`).
        prices_t = torch.where(available_mask > 0.0, prices_t, torch.full_like(prices_t, 0.5))

        # Exogenous features at this step (with global offset for walk-forward folds).
        g = time_offset + t
        if exogenous_matrix.size and 0 <= g < exogenous_matrix.shape[0]:
            x_np = exogenous_matrix[g]
        else:
            x_np = np.zeros(max(n_features, 1), dtype=np.float64)
        x_t = torch.tensor(x_np.astype(np.float32), dtype=torch.float32)

        should_update = update_after_eval_start or (t < eval_start)
        if should_update:
            for _ in range(cfg.steps_per_window):
                optimizer.zero_grad()
                weights = _parameterized_weights(params, available_mask)

                if cfg.disable_copula or n_assets <= 1:
                    R = torch.eye(n_assets, dtype=torch.float32)
                else:
                    R = mlp.correlation_matrix(x_t)
                L, _alpha_used, fallback_fired = cholesky_safe(
                    R, base_shrinkage=copula_shrinkage,
                    max_shrinkage=cfg.max_pd_fallback_shrinkage,
                )
                if fallback_fired:
                    pd_fallback_count += 1

                # Resample epsilon every inner step (true SGD on the MC estimator).
                eps_noise = torch.randn(int(mc_samples), n_assets, dtype=torch.float32)
                kelly = kelly_log_wealth(
                    weights=weights,
                    prices_t=prices_t,
                    L_t=L,
                    eps=eps_noise,
                    temperature=copula_temperature,
                )

                concentration_penalty = torch.sum(
                    torch.clamp(weights - max_weight, min=0.0).pow(2)
                )
                turnover = torch.sum(torch.abs(weights - prev_weights_train))

                loss = (
                    -kelly
                    + turnover_lambda * turnover
                    + concentration_penalty_lambda * concentration_penalty
                )
                loss.backward()
                optimizer.step()
                _project_parameter_vector_inplace(params, available_mask)

        # Snapshot R for diagnostics outside the inner loop (eval-mode forward).
        with torch.no_grad():
            if not cfg.disable_copula and n_assets > 1:
                R_eval = mlp.correlation_matrix(x_t)
                R_accum = R_accum + R_eval.detach().cpu()
                R_count += 1

            current_weights_t = _parameterized_weights(params, available_mask).detach()
            prev_weights_train = current_weights_t.detach().clone()
            current_weights_np = current_weights_t.cpu().numpy()

        if t >= eval_start:
            # Realized step return on the *actual* continuous price path
            # (apples-to-apples with the equal-weight baseline).
            step_returns_safe = np.nan_to_num(step_returns_np, nan=0.0)
            r_t = float(step_returns_safe @ current_weights_np)
            realized_returns.append(r_t)
            log_wealth_increments.append(float(np.log1p(max(r_t, -0.999))))
            turnovers.append(
                float(np.sum(np.abs(current_weights_np - prev_weights_eval)))
            )
            prev_weights_eval = current_weights_np
            if capture_diagnostics:
                eval_weights.append(current_weights_np.copy())

    eval_weights_arr = (
        np.array(eval_weights, dtype=float)
        if eval_weights
        else np.zeros((0, n_assets), dtype=float)
    )
    avg_weights = (
        np.mean(eval_weights_arr, axis=0)
        if eval_weights_arr.size
        else np.zeros(n_assets, dtype=float)
    )
    avg_corr = (
        (R_accum / max(R_count, 1)).numpy()
        if R_count > 0
        else np.eye(n_assets, dtype=float)
    )
    if n_assets > 1:
        triu_iu = np.triu_indices(n_assets, k=1)
        off_diag_abs_mean = float(np.mean(np.abs(avg_corr[triu_iu])))
    else:
        off_diag_abs_mean = 0.0

    return KellyOnlinePassPayload(
        portfolio_returns=np.array(realized_returns, dtype=float),
        log_wealth_increments=np.array(log_wealth_increments, dtype=float),
        turnovers=np.array(turnovers, dtype=float),
        eval_weights=eval_weights_arr,
        avg_weights=avg_weights,
        avg_correlation_matrix=avg_corr,
        pd_fallback_count=int(pd_fallback_count),
        correlation_off_diag_abs_mean=off_diag_abs_mean,
    )


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def _max_drawdown(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if returns.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, 0.0
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / np.clip(peak, 1e-8, None) - 1.0
    return cumulative, drawdown, float(np.min(drawdown))


def _walkforward_kelly_score(
    *,
    tuning_returns: np.ndarray,
    tuning_prices: np.ndarray,
    tuning_exogenous: np.ndarray,
    cfg: KellyExperimentConfig,
    lr_w: float,
    lr_theta: float,
    rolling_window: int,
    mc_samples: int,
    turnover_lambda: float,
    copula_shrinkage: float,
    copula_temperature: float,
    mlp_hidden_dim: int,
    concentration_penalty_lambda: float,
    max_weight: float,
) -> tuple[float, float]:
    """Run walk-forward folds and return ``(mean_log_wealth_per_step, mean_turnover_per_step)``.

    Uses the same fold geometry as :func:`src.constrained_optimizer._run_walkforward_validation`:
    fit on ``train_steps`` and evaluate the next ``test_steps``, slide by ``test_steps``.
    """
    train_steps = max(cfg.walkforward_train_steps, rolling_window + 1)
    test_steps = max(cfg.walkforward_test_steps, 1)
    fold_log_wealth: list[float] = []
    fold_turnovers: list[float] = []
    start = 0
    while start + train_steps + test_steps <= tuning_returns.shape[0]:
        seg_returns = tuning_returns[start : start + train_steps + test_steps]
        # price_matrix has one extra row vs returns; carry the same offset.
        seg_prices = tuning_prices[start : start + train_steps + test_steps + 1]
        payload = _run_kelly_online_pass(
            returns_matrix=seg_returns,
            price_matrix=seg_prices,
            exogenous_matrix=tuning_exogenous,
            cfg=cfg,
            lr_w=lr_w,
            lr_theta=lr_theta,
            rolling_window=rolling_window,
            mc_samples=mc_samples,
            turnover_lambda=turnover_lambda,
            copula_shrinkage=copula_shrinkage,
            copula_temperature=copula_temperature,
            mlp_hidden_dim=mlp_hidden_dim,
            concentration_penalty_lambda=concentration_penalty_lambda,
            max_weight=max_weight,
            seed=cfg.seed,
            evaluation_start_t=train_steps,
            update_after_eval_start=True,
            time_offset=start,
        )
        if payload["log_wealth_increments"].size > 0:
            fold_log_wealth.extend(payload["log_wealth_increments"].tolist())
            fold_turnovers.extend(payload["turnovers"].tolist())
        start += test_steps

    if not fold_log_wealth:
        # Short-data fallback: use the whole tuning slice as a single pass.
        seg_prices = tuning_prices[: tuning_returns.shape[0] + 1]
        payload = _run_kelly_online_pass(
            returns_matrix=tuning_returns,
            price_matrix=seg_prices,
            exogenous_matrix=tuning_exogenous,
            cfg=cfg,
            lr_w=lr_w,
            lr_theta=lr_theta,
            rolling_window=rolling_window,
            mc_samples=mc_samples,
            turnover_lambda=turnover_lambda,
            copula_shrinkage=copula_shrinkage,
            copula_temperature=copula_temperature,
            mlp_hidden_dim=mlp_hidden_dim,
            concentration_penalty_lambda=concentration_penalty_lambda,
            max_weight=max_weight,
            seed=cfg.seed,
            evaluation_start_t=rolling_window,
            update_after_eval_start=True,
            time_offset=0,
        )
        fold_log_wealth = payload["log_wealth_increments"].tolist()
        fold_turnovers = payload["turnovers"].tolist()

    if not fold_log_wealth:
        return 0.0, 0.0
    return float(np.mean(fold_log_wealth)), float(np.mean(fold_turnovers))


# ---------------------------------------------------------------------------
# Optuna driver
# ---------------------------------------------------------------------------


def run_kelly_optuna_search(
    project_root: pathlib.Path,
    *,
    input_artifact_prefix: str = "week8",
    output_artifact_prefix: str = "week10_kelly",
    config: KellyExperimentConfig | None = None,
    n_trials: int = 100,
    timeout_sec: int | None = None,
    mc_samples_override: int | None = None,
    turnover_lambda_override: float | None = None,
    disable_copula: bool | None = None,
) -> dict[str, pathlib.Path]:
    """Optuna QMC search over the joint (weights, MLP) Kelly objective.

    Inputs are read from ``{input_artifact_prefix}_*`` (markets, prices,
    exogenous CSVs). Outputs are written under ``{output_artifact_prefix}_kelly_*``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = config or KellyExperimentConfig()
    if disable_copula is not None:
        cfg = KellyExperimentConfig(**{**cfg.__dict__, "disable_copula": bool(disable_copula)})

    price_matrix, returns_matrix, kept_tokens, domains = _load_kelly_inputs(
        project_root, artifact_prefix=input_artifact_prefix
    )
    if returns_matrix.size == 0 or len(kept_tokens) == 0:
        raise RuntimeError(
            "No returns data available. Run the data-build stage first "
            f"(input_artifact_prefix={input_artifact_prefix!r})."
        )
    n_returns, n_assets = returns_matrix.shape
    split_idx = split_index_for_returns(n_returns, cfg)
    tuning_returns = returns_matrix[:split_idx]
    tuning_prices = price_matrix[: split_idx + 1]
    exogenous_matrix = _load_exogenous_feature_matrix(
        project_root,
        artifact_prefix=input_artifact_prefix,
        n_steps=n_returns,
        feature_columns=cfg.exogenous_feature_columns,
    )
    tuning_exog = exogenous_matrix[:split_idx]

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_artifact_prefix

    def _suggest(
        trial: "optuna.Trial",
        name: str,
        values: tuple[float, ...],
        log: bool = False,
    ) -> float:
        lo, hi = float(min(values)), float(max(values))
        if lo == hi:
            return lo
        if log and lo <= 0:
            return trial.suggest_float(name, lo, hi, log=False)
        return trial.suggest_float(name, lo, hi, log=log)

    search_start = time.perf_counter()
    trial_records: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        lr_w = _suggest(trial, "lr_w", cfg.learning_rates_w, log=True)
        lr_theta = _suggest(trial, "lr_theta", cfg.learning_rates_theta, log=True)
        rolling_window = int(
            trial.suggest_categorical("rolling_window", list(cfg.rolling_windows))
        )
        if mc_samples_override is not None:
            mc_samples = int(mc_samples_override)
            trial.set_user_attr("mc_samples_override", int(mc_samples_override))
        else:
            mc_samples = int(
                trial.suggest_categorical("mc_samples", list(cfg.mc_samples))
            )
        if turnover_lambda_override is not None:
            turnover_lambda = float(turnover_lambda_override)
            trial.set_user_attr(
                "turnover_lambda_override", float(turnover_lambda_override)
            )
        else:
            turnover_lambda = _suggest(
                trial, "turnover_lambda", cfg.turnover_lambdas, log=False
            )
        copula_shrinkage = _suggest(
            trial, "copula_shrinkage", cfg.copula_shrinkages, log=False
        )
        copula_temperature = _suggest(
            trial, "copula_temperature", cfg.copula_temperatures, log=True
        )
        mlp_hidden = int(
            trial.suggest_categorical("mlp_hidden_dim", list(cfg.mlp_hidden_dims))
        )
        conc_lambda = _suggest(
            trial,
            "concentration_penalty_lambda",
            cfg.concentration_penalty_lambdas,
            log=True,
        )
        max_weight = _suggest(trial, "max_weight", cfg.max_weights, log=False)

        mean_log_wealth, mean_turnover = _walkforward_kelly_score(
            tuning_returns=tuning_returns,
            tuning_prices=tuning_prices,
            tuning_exogenous=tuning_exog,
            cfg=cfg,
            lr_w=lr_w,
            lr_theta=lr_theta,
            rolling_window=rolling_window,
            mc_samples=mc_samples,
            turnover_lambda=turnover_lambda,
            copula_shrinkage=copula_shrinkage,
            copula_temperature=copula_temperature,
            mlp_hidden_dim=mlp_hidden,
            concentration_penalty_lambda=conc_lambda,
            max_weight=max_weight,
        )
        score = mean_log_wealth - cfg.selection_turnover_penalty * mean_turnover
        trial.set_user_attr("walkforward_mean_log_wealth_per_step", mean_log_wealth)
        trial.set_user_attr("walkforward_mean_turnover_per_step", mean_turnover)
        trial.set_user_attr("selection_score", score)
        trial_records.append(
            {
                "trial_number": trial.number,
                "lr_w": lr_w,
                "lr_theta": lr_theta,
                "rolling_window": rolling_window,
                "mc_samples": mc_samples,
                "turnover_lambda": turnover_lambda,
                "copula_shrinkage": copula_shrinkage,
                "copula_temperature": copula_temperature,
                "mlp_hidden_dim": mlp_hidden,
                "concentration_penalty_lambda": conc_lambda,
                "max_weight": max_weight,
                "walkforward_mean_log_wealth_per_step": mean_log_wealth,
                "walkforward_mean_turnover_per_step": mean_turnover,
                "selection_score": score,
            }
        )
        return score

    sampler = optuna.samplers.QMCSampler(qmc_type="sobol", scramble=True, seed=cfg.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_jobs = 1 if cfg.optuna_n_jobs is None else int(cfg.optuna_n_jobs)
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=timeout_sec,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    search_elapsed = time.perf_counter() - search_start

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if not completed_trials:
        raise RuntimeError("No Optuna trials completed for the Kelly search.")
    best_trial = study.best_trial

    # ----- Write per-trial grid -----
    grid_path = out_dir / f"{stem}_kelly_experiment_grid.csv"
    grid_fieldnames = [
        "trial_number",
        "lr_w",
        "lr_theta",
        "rolling_window",
        "mc_samples",
        "turnover_lambda",
        "copula_shrinkage",
        "copula_temperature",
        "mlp_hidden_dim",
        "concentration_penalty_lambda",
        "max_weight",
        "walkforward_mean_log_wealth_per_step",
        "walkforward_mean_turnover_per_step",
        "selection_score",
    ]
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grid_fieldnames)
        writer.writeheader()
        for row in trial_records:
            writer.writerow({k: row.get(k, "") for k in grid_fieldnames})

    # ----- Re-run the best trial on the holdout slice -----
    bp = best_trial.params
    holdout_lr_w = float(bp.get("lr_w", cfg.learning_rates_w[0]))
    holdout_lr_theta = float(bp.get("lr_theta", cfg.learning_rates_theta[0]))
    holdout_rolling_window = int(bp.get("rolling_window", cfg.rolling_windows[0]))
    holdout_mc_samples = (
        int(mc_samples_override)
        if mc_samples_override is not None
        else int(bp.get("mc_samples", cfg.mc_samples[0]))
    )
    holdout_turnover_lambda = (
        float(turnover_lambda_override)
        if turnover_lambda_override is not None
        else float(bp.get("turnover_lambda", cfg.turnover_lambdas[0]))
    )
    holdout_copula_shrinkage = float(
        bp.get("copula_shrinkage", cfg.copula_shrinkages[0])
    )
    holdout_copula_temperature = float(
        bp.get("copula_temperature", cfg.copula_temperatures[0])
    )
    holdout_mlp_hidden = int(bp.get("mlp_hidden_dim", cfg.mlp_hidden_dims[0]))
    holdout_conc_lambda = float(
        bp.get("concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0])
    )
    holdout_max_weight = float(bp.get("max_weight", cfg.max_weights[0]))

    holdout_payload = _run_kelly_online_pass(
        returns_matrix=returns_matrix,
        price_matrix=price_matrix,
        exogenous_matrix=exogenous_matrix,
        cfg=cfg,
        lr_w=holdout_lr_w,
        lr_theta=holdout_lr_theta,
        rolling_window=holdout_rolling_window,
        mc_samples=holdout_mc_samples,
        turnover_lambda=holdout_turnover_lambda,
        copula_shrinkage=holdout_copula_shrinkage,
        copula_temperature=holdout_copula_temperature,
        mlp_hidden_dim=holdout_mlp_hidden,
        concentration_penalty_lambda=holdout_conc_lambda,
        max_weight=holdout_max_weight,
        seed=cfg.seed,
        evaluation_start_t=split_idx,
        update_after_eval_start=True,
        time_offset=0,
        capture_diagnostics=True,
    )

    holdout_returns = holdout_payload["portfolio_returns"]
    holdout_log_wealth = holdout_payload["log_wealth_increments"]
    holdout_turnovers = holdout_payload["turnovers"]
    holdout_weights = holdout_payload["eval_weights"]
    if holdout_returns.size == 0:
        raise RuntimeError(
            "Holdout window too short for the selected rolling_window. "
            "Increase holdout_fraction or shrink rolling_windows."
        )
    cumulative, drawdown, max_dd = _max_drawdown(holdout_returns)
    cumulative_log_wealth = np.cumsum(holdout_log_wealth)

    holdout_domain_exposure: dict[str, float] = {}
    avg_holdout_weights = holdout_payload["avg_weights"]
    for idx, domain in enumerate(domains):
        holdout_domain_exposure[domain] = holdout_domain_exposure.get(domain, 0.0) + float(
            avg_holdout_weights[idx]
        )

    # ----- Best metrics JSON -----
    _probe_mlp = DynamicCopulaMLP(
        n_features=max(exogenous_matrix.shape[1] if exogenous_matrix.size else 1, 1),
        n_assets=n_assets,
        hidden_dim=holdout_mlp_hidden,
    )
    mlp_param_count = int(sum(p.numel() for p in _probe_mlp.parameters()))

    selected_params: dict[str, Any] = {
        "lr_w": holdout_lr_w,
        "lr_theta": holdout_lr_theta,
        "rolling_window": holdout_rolling_window,
        "mc_samples": holdout_mc_samples,
        "turnover_lambda": holdout_turnover_lambda,
        "copula_shrinkage": holdout_copula_shrinkage,
        "copula_temperature": holdout_copula_temperature,
        "mlp_hidden_dim": holdout_mlp_hidden,
        "concentration_penalty_lambda": holdout_conc_lambda,
        "max_weight": holdout_max_weight,
        "weight_parameterization": cfg.weight_parameterization,
        "optimizer_type": cfg.optimizer_type,
        "selection_source": "optuna_qmc_kelly",
        "max_domain_exposure": (
            float(max(holdout_domain_exposure.values())) if holdout_domain_exposure else 0.0
        ),
    }

    kelly_metrics = {
        "holdout_log_wealth_total": float(np.sum(holdout_log_wealth)),
        "holdout_log_wealth_per_step": float(np.mean(holdout_log_wealth))
        if holdout_log_wealth.size
        else 0.0,
        "holdout_total_turnover_l1": float(np.sum(holdout_turnovers)),
        "holdout_avg_turnover_l1": float(np.mean(holdout_turnovers))
        if holdout_turnovers.size
        else 0.0,
        "holdout_max_turnover_l1": float(np.max(holdout_turnovers))
        if holdout_turnovers.size
        else 0.0,
        "holdout_max_drawdown": float(max_dd),
        "holdout_mean_realized_return": float(np.mean(holdout_returns))
        if holdout_returns.size
        else 0.0,
        "holdout_volatility": float(np.std(holdout_returns))
        if holdout_returns.size
        else 0.0,
        "holdout_steps": int(holdout_returns.size),
        "mlp_param_count": mlp_param_count,
        "copula_off_diag_abs_mean": float(holdout_payload["correlation_off_diag_abs_mean"]),
        "pd_fallback_steps": int(holdout_payload["pd_fallback_count"]),
    }

    metrics_payload = {
        "strategy": "dynamic_copula_kelly_ogd",
        "best_params": selected_params,
        "kelly_metrics": kelly_metrics,
        "domain_exposure": holdout_domain_exposure,
        "data_split": {
            "tuning_steps": int(tuning_returns.shape[0]),
            "holdout_steps_total": int(returns_matrix.shape[0] - split_idx),
            "holdout_fraction": cfg.holdout_fraction,
            "walkforward_train_steps": cfg.walkforward_train_steps,
            "walkforward_test_steps": cfg.walkforward_test_steps,
            "rolling_window_used": holdout_rolling_window,
        },
        "optuna_summary": {
            "n_trials": int(n_trials),
            "n_jobs": n_jobs,
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "best_trial_number": int(best_trial.number),
            "search_time_sec": float(search_elapsed),
            "sampler": "QMCSampler(sobol)",
            "selection_objective": "mean_log_wealth - lambda_sel * mean_turnover",
            "selection_turnover_penalty": cfg.selection_turnover_penalty,
            "objective_value_best": float(best_trial.value)
            if best_trial.value is not None
            else None,
        },
    }
    metrics_path = out_dir / f"{stem}_kelly_best_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # ----- Best timeseries CSV -----
    timeseries_path = out_dir / f"{stem}_kelly_best_timeseries.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "portfolio_return",
                "cumulative_return",
                "drawdown",
                "log_wealth_increment",
                "cumulative_log_wealth",
                "turnover_l1",
            ],
        )
        writer.writeheader()
        for idx, (ret, cum, dd, lw, clw, tov) in enumerate(
            zip(
                holdout_returns.tolist(),
                cumulative.tolist(),
                drawdown.tolist(),
                holdout_log_wealth.tolist(),
                cumulative_log_wealth.tolist(),
                holdout_turnovers.tolist(),
                strict=False,
            )
        ):
            writer.writerow(
                {
                    "step": idx,
                    "portfolio_return": float(ret),
                    "cumulative_return": float(cum),
                    "drawdown": float(dd),
                    "log_wealth_increment": float(lw),
                    "cumulative_log_wealth": float(clw),
                    "turnover_l1": float(tov),
                }
            )

    # ----- Per-step holdout weights CSV -----
    weights_path = out_dir / f"{stem}_kelly_best_weights.csv"
    with weights_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *kept_tokens])
        for idx in range(holdout_weights.shape[0]):
            writer.writerow([idx, *[float(v) for v in holdout_weights[idx]]])

    # ----- Copula diagnostics JSON -----
    avg_corr = holdout_payload["avg_correlation_matrix"]
    if n_assets > 1:
        triu_iu = np.triu_indices(n_assets, k=1)
        off = avg_corr[triu_iu]
        cond = float(np.linalg.cond(avg_corr)) if avg_corr.size else 0.0
    else:
        off = np.array([], dtype=float)
        cond = 1.0
    diag_payload = {
        "average_correlation_off_diag_mean": float(np.mean(off)) if off.size else 0.0,
        "average_correlation_off_diag_abs_mean": float(np.mean(np.abs(off)))
        if off.size
        else 0.0,
        "average_correlation_off_diag_max_abs": float(np.max(np.abs(off)))
        if off.size
        else 0.0,
        "average_correlation_condition_number": cond,
        "pd_fallback_steps": int(holdout_payload["pd_fallback_count"]),
        "pd_fallback_step_fraction": (
            float(holdout_payload["pd_fallback_count"]) / float(max(holdout_returns.size, 1))
        ),
        "n_assets": int(n_assets),
        "n_holdout_steps": int(holdout_returns.size),
        "exogenous_features_used": list(cfg.exogenous_feature_columns),
        "disable_copula": bool(cfg.disable_copula),
    }
    diagnostics_path = out_dir / f"{stem}_kelly_copula_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")

    return {
        "kelly_grid": grid_path,
        "kelly_best_metrics": metrics_path,
        "kelly_best_timeseries": timeseries_path,
        "kelly_best_weights": weights_path,
        "kelly_copula_diagnostics": diagnostics_path,
    }
