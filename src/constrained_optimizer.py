"""First-pass constrained OGD/SGD experiments for domain-aware allocation.

Macro variants (``ExperimentConfig.macro_integration``): ``rescale`` scales risk
penalties with exogenous z-scores; ``explicit`` adds an optional concentration
macro term; ``both`` combines them. See ``ExperimentConfig`` and
``run_optuna_search`` docstrings for Optuna study options and fair holdout comparison.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import torch

from src.baseline import _build_price_matrix, _compute_returns, _max_drawdown, _read_csv


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for rudimentary constrained optimizer runs.

    macro_integration controls how exogenous z-scores affect the online objective:

    - ``rescale`` (default): ``m_risk`` scales downside / variance / covariance penalties;
      ``uniform_mix`` gets time-varying bumps from stale feed and equity session (legacy).
    - ``explicit``: base penalties only (no ``m_risk`` scaling, no macro ``uniform_mix`` bumps).
      Optional additive macro term ``J_macro = lambda_macro_explicit * tanh(z_eff) * P_conc``
      on the mean-downside objective only (``P_conc`` is the same squared clamp concentration
      penalty already in the loop). Omitted when exogenous series are unavailable.
    - ``both``: rescale path plus the same ``J_macro`` term when enabled.

    For fair holdout comparison across modes, keep the same data split and exogenous CSV;
    only the objective path changes. Sortino objective (``objective == "sortino"``) never
    adds ``J_macro`` so ratio semantics stay unchanged.

    Optuna: either run three studies with ``macro_integration`` fixed per run (cleaner for
    papers) or one joint study with categorical ``macro_mode`` plus always-on draws for
    ``regime_k`` / ``lambda_macro_explicit`` masked by mode (QMCSampler needs a static space).
    """

    learning_rates: tuple[float, ...] = (0.02, 0.05, 0.1)
    penalties_lambda: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    rolling_windows: tuple[int, ...] = (24, 48)
    domain_limits: tuple[float, ...] = (0.35,)
    max_weights: tuple[float, ...] = (0.25,)
    concentration_penalty_lambdas: tuple[float, ...] = (50.0,)
    covariance_penalty_lambdas: tuple[float, ...] = (5.0,)
    covariance_shrinkages: tuple[float, ...] = (0.05,)
    entropy_lambdas: tuple[float, ...] = (0.02,)
    uniform_mixes: tuple[float, ...] = (0.4,)
    steps_per_window: int = 3
    objective: str = "mean_downside"
    variance_penalty: float = 1.0
    downside_penalty: float = 2.0
    variance_penalties: tuple[float, ...] = ()
    downside_penalties: tuple[float, ...] = ()
    optimizer_type: str = "sgd"
    weight_parameterization: str = "softmax"
    evaluation_modes: tuple[str, ...] = ("online",)
    primary_evaluation_mode: str = "online"
    enable_two_stage_search: bool = True
    stage2_top_k: int = 8
    max_parallel_workers: int = 1
    #: Parallel Optuna trials (``study.optimize(..., n_jobs=...)``). Use ``1`` for sequential
    #: runs (default). Use ``-1`` to let Optuna pick ``os.cpu_count()``. On large machines,
    #: set e.g. ``16`` or ``32`` instead of matching every hardware thread to avoid oversubscription
    #: when PyTorch/OpenMP also spawn threads; prefer ``OMP_NUM_THREADS=1`` and
    #: ``TORCH_NUM_THREADS=1`` (or ``MKL_NUM_THREADS=1``) when ``optuna_n_jobs`` is high.
    optuna_n_jobs: int = 1
    #: When True, Optuna tunes ``lambda_etf_tracking``; inner loss adds a term that pulls the
    #: rolling-window Polymarket portfolio returns toward an equal-weight ETF blend (see
    #: ``_load_etf_tracking_matrix`` / ``*_ret_1`` columns in exogenous CSV).
    use_etf_tracking: bool = False
    etf_tracking_lambdas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    early_prune_enabled: bool = True
    early_prune_exposure_factor: float = 1.5
    max_domain_exposure_threshold: float = 1.0
    holdout_fraction: float = 0.2
    walkforward_train_steps: int = 240
    walkforward_test_steps: int = 48
    seed: int = 7
    regime_k: float = 0.25
    macro_integration: Literal["rescale", "explicit", "both"] = "rescale"
    lambda_macro_explicit: float = 0.0


class OnlinePassPayload(TypedDict):
    """Typed payload returned by one online optimization pass."""

    portfolio_returns: np.ndarray
    avg_weights: np.ndarray
    eval_weights: np.ndarray
    eval_asset_returns: np.ndarray


class SelectedExperimentPayload(TypedDict):
    """Typed payload for a selected experiment candidate."""

    params: dict[str, object]
    portfolio_returns: np.ndarray
    cumulative: np.ndarray
    drawdown: np.ndarray
    domain_exposure: dict[str, float]


@dataclass(frozen=True)
class CandidateConfig:
    """One candidate hyperparameter setting for grid evaluation."""

    lr: float
    penalty_lambda: float
    rolling_window: int
    domain_limit: float
    max_weight: float
    concentration_penalty_lambda: float
    covariance_penalty_lambda: float
    covariance_shrinkage: float
    entropy_lambda: float
    uniform_mix: float
    evaluation_mode: str


def _sortino_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean_return = torch.mean(returns)
    downside = torch.clamp(-returns, min=0.0)
    downside_semivar = torch.mean(downside * downside)
    downside_dev = torch.sqrt(downside_semivar + eps)
    return mean_return / downside_dev


def _mean_downside_objective(
    returns: torch.Tensor,
    variance_penalty: float = 1.0,
    downside_penalty: float = 2.0,
) -> torch.Tensor:
    """Gradient-friendly surrogate for Sortino-like risk-adjusted return.

    Decomposes the Sortino ratio's intent into additive terms:
      mean(r) - variance_penalty * var(r) - downside_penalty * mean(max(0,-r)^2)

    Avoids the ratio form whose gradient degrades when the denominator is
    small, and whose estimate is noisy over short rolling windows.
    """
    mean_return = torch.mean(returns)
    variance = torch.var(returns, correction=0)
    downside = torch.clamp(-returns, min=0.0)
    downside_semivar = torch.mean(downside * downside)
    return mean_return - variance_penalty * variance - downside_penalty * downside_semivar


def _load_returns_and_domains(
    project_root: pathlib.Path,
    artifact_prefix: str,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str], dict[str, dict[str, str]]]:
    markets_rows = _read_csv(project_root / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv")
    history_rows = _read_csv(project_root / "data" / "processed" / f"{artifact_prefix}_price_history.csv")
    _, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)
    token_to_domain = {row["yes_token_id"]: row["domain"] for row in markets_rows}
    token_to_meta = {
        row["yes_token_id"]: {
            "domain": row.get("domain", "other"),
            "question": row.get("question", ""),
            "market_id": row.get("market_id", ""),
            "market_slug": row.get("market_slug", ""),
        }
        for row in markets_rows
    }
    kept_domains = [token_to_domain.get(token, "other") for token in kept_tokens]
    return returns_matrix, kept_tokens, kept_domains, token_to_domain, token_to_meta


def split_index_for_returns(n_returns: int, cfg: ExperimentConfig) -> int:
    """First holdout step index; same rule as run_optuna_search / run_experiment_grid."""
    if n_returns <= 0:
        return 0
    split_idx = max(int((1.0 - cfg.holdout_fraction) * n_returns), cfg.walkforward_train_steps)
    return min(split_idx, n_returns - 1)


def _default_exogenous_regime(n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z0 = np.zeros(n_steps, dtype=float)
    return (
        z0,
        z0.copy(),
        z0.copy(),
        np.ones(n_steps, dtype=float),
        z0.copy(),
    )


def _load_exogenous_regime(
    project_root: pathlib.Path,
    artifact_prefix: str,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = project_root / "data" / "processed" / f"{artifact_prefix}_exogenous_features.csv"
    if not path.exists() or n_steps <= 0:
        return _default_exogenous_regime(max(n_steps, 0))
    try:
        df = pd.read_csv(path)
    except OSError:
        return _default_exogenous_regime(n_steps)
    if len(df) != n_steps:
        return _default_exogenous_regime(n_steps)
    for col in ("risk_on_z", "exog_is_stale", "is_equity_open"):
        if col not in df.columns:
            return _default_exogenous_regime(n_steps)
    rz = pd.to_numeric(df["risk_on_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ez = (
        pd.to_numeric(df["energy_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "energy_z" in df.columns
        else np.zeros(n_steps, dtype=float)
    )
    ratesz = (
        pd.to_numeric(df["rates_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "rates_z" in df.columns
        else np.zeros(n_steps, dtype=float)
    )
    st = pd.to_numeric(df["exog_is_stale"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    op = pd.to_numeric(df["is_equity_open"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return rz, ez, ratesz, st, op


# Columns must match ``exogenous_features.TICKERS`` order (spy, qqq, xle, tlt, btc-usd prefixes).
_ETF_RET_1_COLUMNS: tuple[str, ...] = (
    "spy_ret_1",
    "qqq_ret_1",
    "xle_ret_1",
    "tlt_ret_1",
    "btc_usd_ret_1",
)


def _load_etf_tracking_matrix(
    project_root: pathlib.Path,
    artifact_prefix: str,
    n_steps: int,
) -> np.ndarray | None:
    """Load per-step ETF log returns (ret_1) aligned to Polymarket return rows, shape (n_steps, n_etfs)."""
    path = project_root / "data" / "processed" / f"{artifact_prefix}_exogenous_features.csv"
    if not path.exists() or n_steps <= 0:
        return None
    try:
        df = pd.read_csv(path)
    except OSError:
        return None
    if len(df) != n_steps:
        return None
    missing = [c for c in _ETF_RET_1_COLUMNS if c not in df.columns]
    if missing:
        return None
    mat = np.zeros((n_steps, len(_ETF_RET_1_COLUMNS)), dtype=np.float64)
    for j, col in enumerate(_ETF_RET_1_COLUMNS):
        mat[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return mat


def _domain_penalty(
    weights: torch.Tensor,
    domains: list[str],
    domain_limit: float,
) -> torch.Tensor:
    penalty = torch.tensor(0.0, dtype=weights.dtype)
    domain_to_indices: dict[str, list[int]] = {}
    for idx, domain in enumerate(domains):
        domain_to_indices.setdefault(domain, []).append(idx)
    for indices in domain_to_indices.values():
        domain_weight = torch.sum(weights[indices])
        excess = torch.clamp(domain_weight - domain_limit, min=0.0)
        penalty = penalty + excess * excess
    return penalty


def _covariance_penalty(
    weights: torch.Tensor,
    window_returns: torch.Tensor,
    shrinkage: float,
) -> torch.Tensor:
    """Risk penalty w^T Sigma w on rolling-window returns."""
    steps = window_returns.shape[0]
    if steps <= 1:
        return torch.tensor(0.0, dtype=weights.dtype, device=weights.device)

    centered = window_returns - torch.mean(window_returns, dim=0, keepdim=True)
    cov = (centered.transpose(0, 1) @ centered) / float(steps - 1)
    if shrinkage > 0.0:
        diag = torch.diag(torch.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
    return weights @ cov @ weights


def _project_to_simplex(weights: torch.Tensor) -> torch.Tensor:
    """Project a vector onto the probability simplex."""
    if weights.numel() == 0:
        return weights
    sorted_vals, _ = torch.sort(weights, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    idx = torch.arange(1, weights.numel() + 1, device=weights.device, dtype=weights.dtype)
    cond = sorted_vals - (cumsum - 1.0) / idx > 0
    if not torch.any(cond):
        # Fallback for numerical edge cases.
        return torch.full_like(weights, 1.0 / float(weights.numel()))
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return torch.clamp(weights - theta, min=0.0)


def _parameterized_weights(
    params: torch.Tensor,
    available_mask: torch.Tensor,
    uniform_mix: float,
    weight_parameterization: str,
) -> torch.Tensor:
    """Map unconstrained parameters to valid per-step portfolio weights."""
    if weight_parameterization == "projected_simplex":
        masked = torch.clamp(params, min=0.0) * available_mask
        masked_sum = torch.sum(masked)
        if float(masked_sum.detach().cpu().item()) <= 0.0:
            masked = available_mask
            masked_sum = torch.sum(masked)
        masked = masked / torch.clamp(masked_sum, min=1e-8)
    else:
        weights_raw = torch.softmax(params, dim=0)
        masked = weights_raw * available_mask
        masked_sum = torch.sum(masked)
        if float(masked_sum.detach().cpu().item()) <= 0.0:
            masked = available_mask
            masked_sum = torch.sum(masked)
        masked = masked / torch.clamp(masked_sum, min=1e-8)
    uniform_weights = available_mask / torch.clamp(torch.sum(available_mask), min=1e-8)
    return (1.0 - uniform_mix) * masked + uniform_mix * uniform_weights


def _project_parameter_vector_inplace(
    params: torch.Tensor,
    available_mask: torch.Tensor,
    weight_parameterization: str,
) -> None:
    """Project parameters after gradient step when using projected-simplex mode."""
    if weight_parameterization != "projected_simplex":
        return
    with torch.no_grad():
        active_idx = torch.nonzero(available_mask > 0.0, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            params.zero_()
            return
        active_vals = params[active_idx]
        projected_active = _project_to_simplex(active_vals)
        params.zero_()
        params[active_idx] = projected_active


def _run_online_pass(
    returns_matrix: np.ndarray,
    domains: list[str],
    lr: float,
    penalty_lambda: float,
    rolling_window: int,
    steps_per_window: int,
    domain_limit: float,
    max_weight: float,
    concentration_penalty_lambda: float,
    covariance_penalty_lambda: float,
    covariance_shrinkage: float,
    entropy_lambda: float,
    uniform_mix: float,
    seed: int,
    evaluation_start_t: int | None = None,
    update_after_eval_start: bool = True,
    capture_diagnostics: bool = False,
    objective: str = "mean_downside",
    variance_penalty: float = 1.0,
    downside_penalty: float = 2.0,
    optimizer_type: str = "sgd",
    weight_parameterization: str = "softmax",
    regime_k: float = 0.0,
    risk_on_z: np.ndarray | None = None,
    energy_z: np.ndarray | None = None,
    rates_z: np.ndarray | None = None,
    exog_is_stale: np.ndarray | None = None,
    is_equity_open: np.ndarray | None = None,
    time_offset: int = 0,
    macro_integration: str = "rescale",
    lambda_macro_explicit: float = 0.0,
    etf_step_returns: np.ndarray | None = None,
    lambda_etf_tracking: float = 0.0,
) -> OnlinePassPayload:
    if macro_integration not in {"rescale", "explicit", "both"}:
        raise RuntimeError(
            f"Invalid macro_integration {macro_integration!r}. "
            "Expected 'rescale', 'explicit', or 'both'."
        )
    torch.manual_seed(seed)
    if returns_matrix.shape[0] <= rolling_window:
        return OnlinePassPayload(
            portfolio_returns=np.array([], dtype=float),
            avg_weights=np.array([], dtype=float),
            eval_weights=np.zeros((0, 0), dtype=float),
            eval_asset_returns=np.zeros((0, 0), dtype=float),
        )

    n_assets = returns_matrix.shape[1]
    if weight_parameterization not in {"softmax", "projected_simplex"}:
        raise RuntimeError(
            "Unsupported weight_parameterization. Valid values: ['softmax', 'projected_simplex']."
        )
    # Softmax(0) is uniform and differentiable w.r.t. logits. For projected_simplex,
    # all-zero params clamp to zero, masked_sum becomes 0, and _parameterized_weights
    # falls back to a constant mask — no graph to params, so backward() fails.
    if weight_parameterization == "projected_simplex":
        params = torch.full(
            (n_assets,), 1.0 / float(max(n_assets, 1)), dtype=torch.float32, requires_grad=True
        )
    else:
        params = torch.zeros(n_assets, dtype=torch.float32, requires_grad=True)
    optimizer = (
        torch.optim.Adam([params], lr=lr)
        if optimizer_type == "adam"
        else torch.optim.SGD([params], lr=lr)
    )

    realized_returns: list[float] = []
    weight_snapshots: list[np.ndarray] = []
    eval_asset_returns: list[np.ndarray] = []

    eval_start = evaluation_start_t if evaluation_start_t is not None else rolling_window
    use_regime = (
        risk_on_z is not None
        and energy_z is not None
        and rates_z is not None
        and exog_is_stale is not None
        and is_equity_open is not None
        and risk_on_z.shape[0] >= time_offset + returns_matrix.shape[0]
    )
    for t in range(rolling_window, returns_matrix.shape[0]):
        step_returns_np = np.array(returns_matrix[t], dtype=float)
        available_mask_np = np.isfinite(step_returns_np)
        if not np.any(available_mask_np):
            continue
        available_mask = torch.tensor(available_mask_np.astype(np.float32), dtype=torch.float32)
        window_np = np.array(returns_matrix[t - rolling_window : t], dtype=float)
        window_np = np.nan_to_num(window_np, nan=0.0)
        window = torch.tensor(window_np, dtype=torch.float32)

        g = time_offset + t
        z_eff: float | None = None
        if use_regime:
            assert (
                risk_on_z is not None
                and energy_z is not None
                and rates_z is not None
                and exog_is_stale is not None
                and is_equity_open is not None
            )
            rz = float(risk_on_z[g])
            ez = float(energy_z[g])
            rtz = float(rates_z[g])
            st = float(exog_is_stale[g])
            eq = float(is_equity_open[g])
            shrink = 1.0
            if st >= 0.5:
                shrink *= 0.25
            if eq < 0.5:
                shrink *= 0.50
            z_bar = (rz + ez + rtz) / 3.0
            z_eff = shrink * z_bar
            if macro_integration == "explicit":
                downside_penalty_t = downside_penalty
                covariance_penalty_lambda_t = covariance_penalty_lambda
                variance_penalty_t = variance_penalty
                uniform_mix_t = uniform_mix
            else:
                m_risk = max(0.70, min(1.50, math.exp(-regime_k * z_eff)))
                downside_penalty_t = downside_penalty * m_risk
                covariance_penalty_lambda_t = covariance_penalty_lambda * m_risk
                variance_penalty_t = variance_penalty * m_risk
                uniform_mix_t = max(
                    0.0,
                    min(
                        0.40,
                        uniform_mix + 0.05 * (1.0 - eq) + 0.05 * st,
                    ),
                )
        else:
            downside_penalty_t = downside_penalty
            covariance_penalty_lambda_t = covariance_penalty_lambda
            variance_penalty_t = variance_penalty
            uniform_mix_t = uniform_mix

        should_update = update_after_eval_start or (t < eval_start)
        if should_update:
            for _ in range(steps_per_window):
                optimizer.zero_grad()
                # Guaranteed diversification floor to prevent single-domain collapse.
                weights = _parameterized_weights(
                    params=params,
                    available_mask=available_mask,
                    uniform_mix=uniform_mix_t,
                    weight_parameterization=weight_parameterization,
                )
                portfolio = window @ weights
                # Penalize excessive concentration and reward spread.
                concentration_penalty = torch.sum(torch.clamp(weights - max_weight, min=0.0).pow(2))
                covariance_penalty = _covariance_penalty(
                    weights=weights,
                    window_returns=window,
                    shrinkage=covariance_shrinkage,
                )
                entropy_bonus = -torch.sum(weights * torch.log(weights + 1e-8))
                if objective == "mean_downside":
                    return_term = _mean_downside_objective(
                        portfolio,
                        variance_penalty=variance_penalty_t,
                        downside_penalty=downside_penalty_t,
                    )
                else:
                    return_term = _sortino_torch(portfolio)
                obj = (
                    return_term
                    - penalty_lambda
                    * _domain_penalty(
                        weights=weights,
                        domains=domains,
                        domain_limit=domain_limit,
                    )
                    - concentration_penalty_lambda * concentration_penalty
                    - covariance_penalty_lambda_t * covariance_penalty
                    + entropy_lambda * entropy_bonus
                )
                if (
                    objective == "mean_downside"
                    and macro_integration in {"explicit", "both"}
                    and lambda_macro_explicit != 0.0
                    and z_eff is not None
                ):
                    macro_scale = math.tanh(z_eff)
                    obj = obj - lambda_macro_explicit * macro_scale * concentration_penalty
                if (
                    lambda_etf_tracking > 0.0
                    and etf_step_returns is not None
                    and etf_step_returns.ndim == 2
                ):
                    g0 = time_offset + t - rolling_window
                    g1 = time_offset + t
                    if 0 <= g0 and g1 <= etf_step_returns.shape[0]:
                        etf_np = etf_step_returns[g0:g1]
                        if etf_np.shape[0] == window.shape[0] and etf_np.shape[1] > 0:
                            etf_win_t = torch.tensor(etf_np, dtype=torch.float32, device=window.device)
                            bench = torch.mean(etf_win_t, dim=1)
                            align = torch.mean((portfolio - bench) ** 2)
                            obj = obj - lambda_etf_tracking * align
                loss = -obj
                loss.backward()
                optimizer.step()
                _project_parameter_vector_inplace(
                    params=params,
                    available_mask=available_mask,
                    weight_parameterization=weight_parameterization,
                )

        current_weights = _parameterized_weights(
            params=params,
            available_mask=available_mask,
            uniform_mix=uniform_mix_t,
            weight_parameterization=weight_parameterization,
        ).detach().cpu().numpy()
        if t >= eval_start:
            weight_snapshots.append(current_weights)
            realized_returns.append(float(np.nan_to_num(step_returns_np, nan=0.0) @ current_weights))
            if capture_diagnostics:
                eval_asset_returns.append(np.nan_to_num(step_returns_np, nan=0.0))

    avg_weights = (
        np.mean(np.array(weight_snapshots), axis=0)
        if weight_snapshots
        else np.zeros(n_assets, dtype=float)
    )
    eval_weights_arr = (
        np.array(weight_snapshots, dtype=float)
        if weight_snapshots and capture_diagnostics
        else np.zeros((0, n_assets), dtype=float)
    )
    eval_asset_returns_arr = (
        np.array(eval_asset_returns, dtype=float)
        if eval_asset_returns and capture_diagnostics
        else np.zeros((0, n_assets), dtype=float)
    )
    return OnlinePassPayload(
        portfolio_returns=np.array(realized_returns, dtype=float),
        avg_weights=avg_weights,
        eval_weights=eval_weights_arr,
        eval_asset_returns=eval_asset_returns_arr,
    )


def _write_matrix_csv(path: pathlib.Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *matrix[idx, :].tolist()])


def _build_attribution_artifacts(
    out_dir: pathlib.Path,
    artifact_prefix: str,
    *,
    mode_suffix: str,
    holdout_payload: OnlinePassPayload,
    kept_tokens: list[str],
    token_to_meta: dict[str, dict[str, str]],
) -> dict[str, pathlib.Path]:
    eval_weights = holdout_payload["eval_weights"]
    eval_asset_returns = holdout_payload["eval_asset_returns"]
    portfolio_returns = holdout_payload["portfolio_returns"]
    if (
        eval_weights.size == 0
        or eval_asset_returns.size == 0
        or eval_weights.shape != eval_asset_returns.shape
    ):
        return {}

    contrib_matrix = eval_weights * eval_asset_returns
    total_contrib = np.sum(contrib_matrix, axis=0)
    mean_contrib = np.mean(contrib_matrix, axis=0)
    mean_weight = np.mean(eval_weights, axis=0)
    total_portfolio_return = float(np.sum(portfolio_returns)) if portfolio_returns.size else 0.0

    market_rows: list[dict[str, float | str]] = []
    for idx, token in enumerate(kept_tokens):
        meta = token_to_meta.get(token, {})
        contribution_share = (
            float(total_contrib[idx] / total_portfolio_return)
            if abs(total_portfolio_return) > 1e-12
            else 0.0
        )
        market_rows.append(
            {
                "token_id": token,
                "domain": meta.get("domain", "other"),
                "question": meta.get("question", ""),
                "market_id": meta.get("market_id", ""),
                "market_slug": meta.get("market_slug", ""),
                "mean_weight": float(mean_weight[idx]),
                "total_contribution": float(total_contrib[idx]),
                "mean_step_contribution": float(mean_contrib[idx]),
                "contribution_share_of_total_return": contribution_share,
            }
        )
    market_rows_sorted = sorted(
        market_rows, key=lambda row: abs(float(row["total_contribution"])), reverse=True
    )

    market_path = out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_market_return_contributions.csv"
    with market_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "token_id",
                "domain",
                "question",
                "market_id",
                "market_slug",
                "mean_weight",
                "total_contribution",
                "mean_step_contribution",
                "contribution_share_of_total_return",
            ],
        )
        writer.writeheader()
        writer.writerows(market_rows_sorted)

    domain_totals: dict[str, float] = {}
    for row in market_rows:
        domain = str(row["domain"])
        domain_totals[domain] = domain_totals.get(domain, 0.0) + float(row["total_contribution"])
    domain_rows = [
        {
            "domain": domain,
            "total_contribution": value,
            "contribution_share_of_total_return": (
                value / total_portfolio_return if abs(total_portfolio_return) > 1e-12 else 0.0
            ),
        }
        for domain, value in sorted(domain_totals.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ]
    domain_path = out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_domain_return_contributions.csv"
    with domain_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "total_contribution", "contribution_share_of_total_return"],
        )
        writer.writeheader()
        writer.writerows(domain_rows)

    step_rows: list[dict[str, float | str | int]] = []
    for step_idx in range(contrib_matrix.shape[0]):
        step_contrib = contrib_matrix[step_idx]
        top_market_idx = int(np.argmax(np.abs(step_contrib)))
        top_token = kept_tokens[top_market_idx]
        top_meta = token_to_meta.get(top_token, {})
        domain_step: dict[str, float] = {}
        for idx, token in enumerate(kept_tokens):
            domain = token_to_meta.get(token, {}).get("domain", "other")
            domain_step[domain] = domain_step.get(domain, 0.0) + float(step_contrib[idx])
        top_domain, top_domain_contrib = max(domain_step.items(), key=lambda kv: abs(kv[1]))
        step_rows.append(
            {
                "step": step_idx,
                "portfolio_return": float(portfolio_returns[step_idx]),
                "top_market_token_id": top_token,
                "top_market_domain": top_meta.get("domain", "other"),
                "top_market_question": top_meta.get("question", ""),
                "top_market_contribution": float(step_contrib[top_market_idx]),
                "top_domain": top_domain,
                "top_domain_contribution": float(top_domain_contrib),
            }
        )
    step_path = out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_step_top_contributors.csv"
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "portfolio_return",
                "top_market_token_id",
                "top_market_domain",
                "top_market_question",
                "top_market_contribution",
                "top_domain",
                "top_domain_contribution",
            ],
        )
        writer.writeheader()
        writer.writerows(step_rows)

    top_k = min(20, len(kept_tokens))
    top_tokens = [str(row["token_id"]) for row in market_rows_sorted[:top_k]]
    top_indices = [kept_tokens.index(token) for token in top_tokens]
    top_returns = eval_asset_returns[:, top_indices]
    corr_matrix = np.atleast_2d(np.corrcoef(top_returns, rowvar=False))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr_labels = [
        f"{token_to_meta.get(token, {}).get('domain', 'other')}::{token}" for token in top_tokens
    ]
    corr_path = out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_top_market_correlation.csv"
    _write_matrix_csv(corr_path, corr_labels, corr_matrix)

    corr_pairs: list[dict[str, float | str]] = []
    for i in range(len(top_tokens)):
        for j in range(i + 1, len(top_tokens)):
            corr = float(corr_matrix[i, j])
            corr_pairs.append(
                {
                    "token_a": top_tokens[i],
                    "token_b": top_tokens[j],
                    "corr": corr,
                    "abs_corr": abs(corr),
                }
            )
    corr_pairs_sorted = sorted(corr_pairs, key=lambda row: float(row["abs_corr"]), reverse=True)
    corr_pairs_path = (
        out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_top_market_correlation_pairs.csv"
    )
    with corr_pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["token_a", "token_b", "corr", "abs_corr"],
        )
        writer.writeheader()
        writer.writerows(corr_pairs_sorted)

    summary = {
        "steps": int(contrib_matrix.shape[0]),
        "asset_count": int(contrib_matrix.shape[1]),
        "total_portfolio_return": total_portfolio_return,
        "top_market_by_abs_contribution": market_rows_sorted[0] if market_rows_sorted else {},
        "top_domain_by_abs_contribution": domain_rows[0] if domain_rows else {},
        "top_abs_corr_pair": corr_pairs_sorted[0] if corr_pairs_sorted else {},
    }
    summary_path = out_dir / f"{artifact_prefix}_constrained_best{mode_suffix}_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "market_return_contributions": market_path,
        "domain_return_contributions": domain_path,
        "step_top_contributors": step_path,
        "top_market_correlation": corr_path,
        "top_market_correlation_pairs": corr_pairs_path,
        "attribution_summary": summary_path,
    }


def _compute_metrics(returns: np.ndarray) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Compute return/risk metrics and equity/drawdown series."""
    if returns.size == 0:
        empty = np.array([], dtype=float)
        return 0.0, 0.0, 0.0, 0.0, empty, empty
    cumulative = np.cumprod(1.0 + returns)
    drawdown, max_dd = _max_drawdown(cumulative)
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(np.mean(returns) / (downside_dev + 1e-8))
    mean_return = float(np.mean(returns))
    volatility = float(np.std(returns))
    return sortino, float(max_dd), mean_return, volatility, cumulative, drawdown


def _coarse_subset_float(values: tuple[float, ...]) -> tuple[float, ...]:
    """Pick a compact representative subset for stage-1 coarse search."""
    if len(values) <= 2:
        return values
    idx = sorted({0, len(values) // 2, len(values) - 1})
    return tuple(values[i] for i in idx)


def _coarse_subset_int(values: tuple[int, ...]) -> tuple[int, ...]:
    if len(values) <= 2:
        return values
    idx = sorted({0, len(values) // 2, len(values) - 1})
    return tuple(values[i] for i in idx)


def _build_candidates(
    *,
    learning_rates: tuple[float, ...],
    penalties_lambda: tuple[float, ...],
    rolling_windows: tuple[int, ...],
    domain_limits: tuple[float, ...],
    max_weights: tuple[float, ...],
    concentration_penalty_lambdas: tuple[float, ...],
    covariance_penalty_lambdas: tuple[float, ...],
    covariance_shrinkages: tuple[float, ...],
    entropy_lambdas: tuple[float, ...],
    uniform_mixes: tuple[float, ...],
    modes: tuple[str, ...],
) -> list[CandidateConfig]:
    candidates: list[CandidateConfig] = []
    for (
        lr,
        penalty_lambda,
        window,
        domain_limit,
        max_weight,
        concentration_penalty_lambda,
        covariance_penalty_lambda,
        covariance_shrinkage,
        entropy_lambda,
        uniform_mix,
        evaluation_mode,
    ) in itertools.product(
        learning_rates,
        penalties_lambda,
        rolling_windows,
        domain_limits,
        max_weights,
        concentration_penalty_lambdas,
        covariance_penalty_lambdas,
        covariance_shrinkages,
        entropy_lambdas,
        uniform_mixes,
        modes,
    ):
        candidates.append(
            CandidateConfig(
                lr=float(lr),
                penalty_lambda=float(penalty_lambda),
                rolling_window=int(window),
                domain_limit=float(domain_limit),
                max_weight=float(max_weight),
                concentration_penalty_lambda=float(concentration_penalty_lambda),
                covariance_penalty_lambda=float(covariance_penalty_lambda),
                covariance_shrinkage=float(covariance_shrinkage),
                entropy_lambda=float(entropy_lambda),
                uniform_mix=float(uniform_mix),
                evaluation_mode=str(evaluation_mode),
            )
        )
    return candidates


def _neighbor_values_float(current: float, values: tuple[float, ...]) -> list[float]:
    if current not in values:
        return []
    idx = values.index(current)
    out: list[float] = []
    if idx > 0:
        out.append(float(values[idx - 1]))
    if idx + 1 < len(values):
        out.append(float(values[idx + 1]))
    return out


def _neighbor_values_int(current: int, values: tuple[int, ...]) -> list[int]:
    if current not in values:
        return []
    idx = values.index(current)
    out: list[int] = []
    if idx > 0:
        out.append(int(values[idx - 1]))
    if idx + 1 < len(values):
        out.append(int(values[idx + 1]))
    return out


def _run_walkforward_validation(
    tuning_returns: np.ndarray,
    domains: list[str],
    cfg: ExperimentConfig,
    lr: float,
    penalty_lambda: float,
    rolling_window: int,
    domain_limit: float,
    max_weight: float,
    concentration_penalty_lambda: float,
    covariance_penalty_lambda: float,
    covariance_shrinkage: float,
    entropy_lambda: float,
    uniform_mix: float,
    evaluation_mode: str,
    risk_on_z: np.ndarray | None = None,
    energy_z: np.ndarray | None = None,
    rates_z: np.ndarray | None = None,
    exog_is_stale: np.ndarray | None = None,
    is_equity_open: np.ndarray | None = None,
    regime_k: float = 0.0,
    macro_integration: str = "rescale",
    lambda_macro_explicit: float = 0.0,
    etf_step_returns: np.ndarray | None = None,
    lambda_etf_tracking: float = 0.0,
) -> OnlinePassPayload:
    """Walk-forward tune: fit on past block, evaluate on next unseen block."""
    train_steps = max(cfg.walkforward_train_steps, rolling_window + 1)
    test_steps = max(cfg.walkforward_test_steps, 1)
    fold_returns: list[float] = []
    fold_weights: list[np.ndarray] = []
    start = 0

    while start + train_steps + test_steps <= tuning_returns.shape[0]:
        segment = tuning_returns[start : start + train_steps + test_steps]
        fold_payload = _run_online_pass(
            returns_matrix=segment,
            domains=domains,
            lr=lr,
            penalty_lambda=penalty_lambda,
            rolling_window=rolling_window,
            steps_per_window=cfg.steps_per_window,
            domain_limit=domain_limit,
            max_weight=max_weight,
            concentration_penalty_lambda=concentration_penalty_lambda,
            covariance_penalty_lambda=covariance_penalty_lambda,
            covariance_shrinkage=covariance_shrinkage,
            entropy_lambda=entropy_lambda,
            uniform_mix=uniform_mix,
            seed=cfg.seed,
            evaluation_start_t=train_steps,
            update_after_eval_start=(evaluation_mode == "online"),
            objective=cfg.objective,
            variance_penalty=cfg.variance_penalty,
            downside_penalty=cfg.downside_penalty,
            optimizer_type=cfg.optimizer_type,
            weight_parameterization=cfg.weight_parameterization,
            regime_k=regime_k,
            risk_on_z=risk_on_z,
            energy_z=energy_z,
            rates_z=rates_z,
            exog_is_stale=exog_is_stale,
            is_equity_open=is_equity_open,
            time_offset=start,
            macro_integration=macro_integration,
            lambda_macro_explicit=lambda_macro_explicit,
            etf_step_returns=etf_step_returns,
            lambda_etf_tracking=lambda_etf_tracking,
        )
        if (
            cfg.early_prune_enabled
            and cfg.max_domain_exposure_threshold > 0.0
            and fold_payload["avg_weights"].size == len(domains)
        ):
            fold_domain_exposure: dict[str, float] = {}
            for idx, domain in enumerate(domains):
                fold_domain_exposure[domain] = fold_domain_exposure.get(domain, 0.0) + float(
                    fold_payload["avg_weights"][idx]
                )
            fold_max_exposure = max(fold_domain_exposure.values()) if fold_domain_exposure else 0.0
            if fold_max_exposure > (cfg.max_domain_exposure_threshold * cfg.early_prune_exposure_factor):
                return OnlinePassPayload(
                    portfolio_returns=np.array([], dtype=float),
                    avg_weights=np.array([], dtype=float),
                    eval_weights=np.zeros((0, 0), dtype=float),
                    eval_asset_returns=np.zeros((0, 0), dtype=float),
                )
        fold_returns.extend(fold_payload["portfolio_returns"].tolist())
        fold_weights.append(fold_payload["avg_weights"])
        start += test_steps

    if not fold_returns:
        # Fallback for short datasets.
        return _run_online_pass(
            returns_matrix=tuning_returns,
            domains=domains,
            lr=lr,
            penalty_lambda=penalty_lambda,
            rolling_window=rolling_window,
            steps_per_window=cfg.steps_per_window,
            domain_limit=domain_limit,
            max_weight=max_weight,
            concentration_penalty_lambda=concentration_penalty_lambda,
            covariance_penalty_lambda=covariance_penalty_lambda,
            covariance_shrinkage=covariance_shrinkage,
            entropy_lambda=entropy_lambda,
            uniform_mix=uniform_mix,
            seed=cfg.seed,
            update_after_eval_start=(evaluation_mode == "online"),
            objective=cfg.objective,
            variance_penalty=cfg.variance_penalty,
            downside_penalty=cfg.downside_penalty,
            optimizer_type=cfg.optimizer_type,
            weight_parameterization=cfg.weight_parameterization,
            regime_k=regime_k,
            risk_on_z=risk_on_z,
            energy_z=energy_z,
            rates_z=rates_z,
            exog_is_stale=exog_is_stale,
            is_equity_open=is_equity_open,
            time_offset=0,
            macro_integration=macro_integration,
            lambda_macro_explicit=lambda_macro_explicit,
            etf_step_returns=etf_step_returns,
            lambda_etf_tracking=lambda_etf_tracking,
        )

    avg_weights = (
        np.mean(np.stack(fold_weights, axis=0), axis=0)
        if fold_weights
        else np.array([], dtype=float)
    )
    return OnlinePassPayload(
        portfolio_returns=np.array(fold_returns, dtype=float),
        avg_weights=avg_weights,
        eval_weights=np.zeros((0, 0), dtype=float),
        eval_asset_returns=np.zeros((0, 0), dtype=float),
    )


def _evaluate_candidate(
    *,
    candidate: CandidateConfig,
    tuning_returns: np.ndarray,
    domains: list[str],
    cfg: ExperimentConfig,
    risk_on_z: np.ndarray | None = None,
    energy_z: np.ndarray | None = None,
    rates_z: np.ndarray | None = None,
    exog_is_stale: np.ndarray | None = None,
    is_equity_open: np.ndarray | None = None,
    etf_step_returns: np.ndarray | None = None,
    lambda_etf_tracking: float = 0.0,
) -> tuple[dict[str, Any], SelectedExperimentPayload] | None:
    payload = _run_walkforward_validation(
        tuning_returns=tuning_returns,
        domains=domains,
        cfg=cfg,
        lr=candidate.lr,
        penalty_lambda=candidate.penalty_lambda,
        rolling_window=candidate.rolling_window,
        domain_limit=candidate.domain_limit,
        max_weight=candidate.max_weight,
        concentration_penalty_lambda=candidate.concentration_penalty_lambda,
        covariance_penalty_lambda=candidate.covariance_penalty_lambda,
        covariance_shrinkage=candidate.covariance_shrinkage,
        entropy_lambda=candidate.entropy_lambda,
        uniform_mix=candidate.uniform_mix,
        evaluation_mode=candidate.evaluation_mode,
        risk_on_z=risk_on_z,
        energy_z=energy_z,
        rates_z=rates_z,
        exog_is_stale=exog_is_stale,
        is_equity_open=is_equity_open,
        regime_k=cfg.regime_k,
        macro_integration=cfg.macro_integration,
        lambda_macro_explicit=cfg.lambda_macro_explicit,
        etf_step_returns=etf_step_returns,
        lambda_etf_tracking=lambda_etf_tracking,
    )
    portfolio_returns = payload["portfolio_returns"]
    if not isinstance(portfolio_returns, np.ndarray) or portfolio_returns.size == 0:
        return None

    sortino, max_dd, mean_return, volatility, cumulative, drawdown = _compute_metrics(portfolio_returns)
    avg_weights = payload["avg_weights"]
    if avg_weights.size == 0:
        return None

    domain_exposure: dict[str, float] = {}
    for idx, domain in enumerate(domains):
        domain_exposure[domain] = domain_exposure.get(domain, 0.0) + float(avg_weights[idx])

    row: dict[str, Any] = {
        "learning_rate": candidate.lr,
        "lambda_penalty": candidate.penalty_lambda,
        "rolling_window": candidate.rolling_window,
        "domain_limit": candidate.domain_limit,
        "max_weight": candidate.max_weight,
        "uniform_mix": candidate.uniform_mix,
        "evaluation_mode": candidate.evaluation_mode,
        "weight_parameterization": cfg.weight_parameterization,
        "sortino_ratio": sortino,
        "max_drawdown": float(max_dd),
        "mean_return": mean_return,
        "volatility": volatility,
        "concentration_penalty_lambda": candidate.concentration_penalty_lambda,
        "covariance_penalty_lambda": candidate.covariance_penalty_lambda,
        "covariance_shrinkage": candidate.covariance_shrinkage,
        "entropy_lambda": candidate.entropy_lambda,
        "regime_k": cfg.regime_k,
        "macro_integration": cfg.macro_integration,
        "lambda_macro_explicit": cfg.lambda_macro_explicit,
        "max_domain_exposure": float(max(domain_exposure.values())) if domain_exposure else 0.0,
        "max_domain_exposure_threshold": cfg.max_domain_exposure_threshold,
        "domain_exposure_json": json.dumps(domain_exposure),
    }
    selected_payload = SelectedExperimentPayload(
        params=row,
        portfolio_returns=portfolio_returns,
        cumulative=cumulative,
        drawdown=drawdown,
        domain_exposure=domain_exposure,
    )
    return row, selected_payload


def run_experiment_grid(
    project_root: pathlib.Path,
    config: ExperimentConfig | None = None,
    artifact_prefix: str = "week8",
) -> dict[str, pathlib.Path]:
    """Run rudimentary constrained experiments and persist results."""
    cfg = config or ExperimentConfig()
    valid_modes = {"online", "frozen"}
    modes = tuple(dict.fromkeys(cfg.evaluation_modes))
    if not modes:
        raise RuntimeError("evaluation_modes must contain at least one mode.")
    if any(mode not in valid_modes for mode in modes):
        raise RuntimeError(f"Unsupported evaluation mode in {modes}. Valid modes: {sorted(valid_modes)}.")
    primary_mode = cfg.primary_evaluation_mode
    if primary_mode not in modes:
        raise RuntimeError("primary_evaluation_mode must be included in evaluation_modes.")
    returns_matrix, kept_tokens, domains, _, token_to_meta = _load_returns_and_domains(
        project_root, artifact_prefix=artifact_prefix
    )
    if returns_matrix.size == 0:
        raise RuntimeError("No returns data available. Run data build first.")
    split_idx = max(int((1.0 - cfg.holdout_fraction) * returns_matrix.shape[0]), cfg.walkforward_train_steps)
    split_idx = min(split_idx, returns_matrix.shape[0] - 1)
    tuning_returns = returns_matrix[:split_idx]
    rz, ez, ratesz, st, op = _load_exogenous_regime(
        project_root, artifact_prefix, returns_matrix.shape[0]
    )
    etf_matrix_full_grid: np.ndarray | None = None
    lambda_etf_grid_holdout = 0.0
    if cfg.use_etf_tracking:
        etf_matrix_full_grid = _load_etf_tracking_matrix(
            project_root, artifact_prefix, returns_matrix.shape[0]
        )
        if etf_matrix_full_grid is not None and cfg.etf_tracking_lambdas:
            mid = len(cfg.etf_tracking_lambdas) // 2
            lambda_etf_grid_holdout = float(cfg.etf_tracking_lambdas[mid])

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    evaluated_candidates: set[CandidateConfig] = set()
    best_payload_by_mode: dict[str, SelectedExperimentPayload | None] = {mode: None for mode in modes}
    best_sortino_by_mode: dict[str, float] = {mode: -np.inf for mode in modes}
    feasible_payload_by_mode: dict[str, SelectedExperimentPayload | None] = {mode: None for mode in modes}
    feasible_best_sortino_by_mode: dict[str, float] = {mode: -np.inf for mode in modes}
    def _consume_results(
        results: list[tuple[CandidateConfig, tuple[dict[str, Any], SelectedExperimentPayload] | None]],
    ) -> None:
        for candidate, evaluated in results:
            evaluated_candidates.add(candidate)
            if evaluated is None:
                continue
            row, selected_payload = evaluated
            summary_rows.append(row)
            sortino = float(row["sortino_ratio"])
            mode = candidate.evaluation_mode
            if sortino > best_sortino_by_mode[mode]:
                best_sortino_by_mode[mode] = sortino
                best_payload_by_mode[mode] = selected_payload
            if (
                float(row["max_domain_exposure"]) <= cfg.max_domain_exposure_threshold
                and sortino > feasible_best_sortino_by_mode[mode]
            ):
                feasible_best_sortino_by_mode[mode] = sortino
                feasible_payload_by_mode[mode] = selected_payload

    def _evaluate_candidate_list(
        candidates: list[CandidateConfig],
        stage_label: str = "Grid",
    ) -> list[tuple[CandidateConfig, tuple[dict[str, Any], SelectedExperimentPayload] | None]]:
        if not candidates:
            return []
        total = len(candidates)
        counter_lock = threading.Lock()
        completed_count = [0]
        batch_start = time.perf_counter()

        def _log_progress(candidate: CandidateConfig) -> None:
            with counter_lock:
                completed_count[0] += 1
                done = completed_count[0]
            elapsed = time.perf_counter() - batch_start
            avg = elapsed / done
            remaining = avg * (total - done)
            mins_left = remaining / 60.0
            print(
                f"  [{stage_label}] {done}/{total} candidates done "
                f"({elapsed / 60:.1f}m elapsed, ~{mins_left:.1f}m remaining)",
                flush=True,
            )

        if cfg.max_parallel_workers <= 1:
            outputs: list[tuple[CandidateConfig, tuple[dict[str, Any], SelectedExperimentPayload] | None]] = []
            for candidate in candidates:
                result = _evaluate_candidate(
                    candidate=candidate,
                    tuning_returns=tuning_returns,
                    domains=domains,
                    cfg=cfg,
                    risk_on_z=rz,
                    energy_z=ez,
                    rates_z=ratesz,
                    exog_is_stale=st,
                    is_equity_open=op,
                    etf_step_returns=etf_matrix_full_grid,
                    lambda_etf_tracking=lambda_etf_grid_holdout,
                )
                outputs.append((candidate, result))
                _log_progress(candidate)
            return outputs
        outputs = []
        with ThreadPoolExecutor(max_workers=cfg.max_parallel_workers) as executor:
            future_map = {
                executor.submit(
                    _evaluate_candidate,
                    candidate=candidate,
                    tuning_returns=tuning_returns,
                    domains=domains,
                    cfg=cfg,
                    risk_on_z=rz,
                    energy_z=ez,
                    rates_z=ratesz,
                    exog_is_stale=st,
                    is_equity_open=op,
                    etf_step_returns=etf_matrix_full_grid,
                    lambda_etf_tracking=lambda_etf_grid_holdout,
                ): candidate
                for candidate in candidates
            }
            for future in as_completed(future_map):
                candidate = future_map[future]
                outputs.append((candidate, future.result()))
                _log_progress(candidate)
        return outputs

    grid_start = time.perf_counter()
    if cfg.enable_two_stage_search:
        stage1_candidates = _build_candidates(
            learning_rates=_coarse_subset_float(cfg.learning_rates),
            penalties_lambda=_coarse_subset_float(cfg.penalties_lambda),
            rolling_windows=_coarse_subset_int(cfg.rolling_windows),
            domain_limits=_coarse_subset_float(cfg.domain_limits),
            max_weights=_coarse_subset_float(cfg.max_weights),
            concentration_penalty_lambdas=_coarse_subset_float(cfg.concentration_penalty_lambdas),
            covariance_penalty_lambdas=_coarse_subset_float(cfg.covariance_penalty_lambdas),
            covariance_shrinkages=_coarse_subset_float(cfg.covariance_shrinkages),
            entropy_lambdas=_coarse_subset_float(cfg.entropy_lambdas),
            uniform_mixes=_coarse_subset_float(cfg.uniform_mixes),
            modes=modes,
        )
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE 1 (coarse): {len(stage1_candidates)} candidates", flush=True)
        print(f"{'='*60}", flush=True)
        _consume_results(_evaluate_candidate_list(stage1_candidates, stage_label="Stage 1"))

        top_rows = sorted(summary_rows, key=lambda row: float(row["sortino_ratio"]), reverse=True)[
            : max(1, cfg.stage2_top_k)
        ]
        top_candidates: list[CandidateConfig] = []
        for row in top_rows:
            top_candidates.append(
                CandidateConfig(
                    lr=float(row["learning_rate"]),
                    penalty_lambda=float(row["lambda_penalty"]),
                    rolling_window=int(row["rolling_window"]),
                    domain_limit=float(row["domain_limit"]),
                    max_weight=float(row["max_weight"]),
                    concentration_penalty_lambda=float(row["concentration_penalty_lambda"]),
                    covariance_penalty_lambda=float(row["covariance_penalty_lambda"]),
                    covariance_shrinkage=float(row["covariance_shrinkage"]),
                    entropy_lambda=float(row["entropy_lambda"]),
                    uniform_mix=float(row["uniform_mix"]),
                    evaluation_mode=str(row["evaluation_mode"]),
                )
            )

        stage2_set: set[CandidateConfig] = set()
        for candidate in top_candidates:
            stage2_set.add(candidate)
            for value in _neighbor_values_float(candidate.lr, cfg.learning_rates):
                stage2_set.add(replace(candidate, lr=value))
            for value in _neighbor_values_float(candidate.penalty_lambda, cfg.penalties_lambda):
                stage2_set.add(replace(candidate, penalty_lambda=value))
            for value in _neighbor_values_int(candidate.rolling_window, cfg.rolling_windows):
                stage2_set.add(replace(candidate, rolling_window=value))
            for value in _neighbor_values_float(candidate.domain_limit, cfg.domain_limits):
                stage2_set.add(replace(candidate, domain_limit=value))
            for value in _neighbor_values_float(candidate.max_weight, cfg.max_weights):
                stage2_set.add(replace(candidate, max_weight=value))
            for value in _neighbor_values_float(
                candidate.concentration_penalty_lambda, cfg.concentration_penalty_lambdas
            ):
                stage2_set.add(replace(candidate, concentration_penalty_lambda=value))
            for value in _neighbor_values_float(candidate.covariance_penalty_lambda, cfg.covariance_penalty_lambdas):
                stage2_set.add(replace(candidate, covariance_penalty_lambda=value))
            for value in _neighbor_values_float(candidate.covariance_shrinkage, cfg.covariance_shrinkages):
                stage2_set.add(replace(candidate, covariance_shrinkage=value))
            for value in _neighbor_values_float(candidate.entropy_lambda, cfg.entropy_lambdas):
                stage2_set.add(replace(candidate, entropy_lambda=value))
            for value in _neighbor_values_float(candidate.uniform_mix, cfg.uniform_mixes):
                stage2_set.add(replace(candidate, uniform_mix=value))
        stage2_candidates = [candidate for candidate in stage2_set if candidate not in evaluated_candidates]
        stage1_elapsed = time.perf_counter() - grid_start
        print(f"\n{'='*60}", flush=True)
        print(
            f"STAGE 2 (fine-tune): {len(stage2_candidates)} new candidates "
            f"(stage 1 took {stage1_elapsed / 60:.1f}m)",
            flush=True,
        )
        print(f"{'='*60}", flush=True)
        _consume_results(_evaluate_candidate_list(stage2_candidates, stage_label="Stage 2"))
    else:
        full_candidates = _build_candidates(
            learning_rates=cfg.learning_rates,
            penalties_lambda=cfg.penalties_lambda,
            rolling_windows=cfg.rolling_windows,
            domain_limits=cfg.domain_limits,
            max_weights=cfg.max_weights,
            concentration_penalty_lambdas=cfg.concentration_penalty_lambdas,
            covariance_penalty_lambdas=cfg.covariance_penalty_lambdas,
            covariance_shrinkages=cfg.covariance_shrinkages,
            entropy_lambdas=cfg.entropy_lambdas,
            uniform_mixes=cfg.uniform_mixes,
            modes=modes,
        )
        print(f"\n{'='*60}", flush=True)
        print(f"GRID SEARCH: {len(full_candidates)} candidates", flush=True)
        print(f"{'='*60}", flush=True)
        _consume_results(_evaluate_candidate_list(full_candidates, stage_label="Grid"))

    grid_elapsed = time.perf_counter() - grid_start
    print(f"\n{'='*60}", flush=True)
    print(
        f"Grid search complete — {len(summary_rows)} candidates evaluated "
        f"in {grid_elapsed / 60:.1f}m ({grid_elapsed / 3600:.1f}h)",
        flush=True,
    )
    print(f"{'='*60}\n", flush=True)

    summary_path = out_dir / f"{artifact_prefix}_constrained_experiment_grid.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            handle.write("")

    if best_payload_by_mode[primary_mode] is None:
        raise RuntimeError("No valid constrained experiment produced returns.")
    artifacts: dict[str, pathlib.Path] = {"constrained_grid": summary_path}
    for mode in modes:
        mode_best = best_payload_by_mode[mode]
        mode_feasible = feasible_payload_by_mode[mode]
        if mode_best is None:
            continue
        selected_payload_train: SelectedExperimentPayload = (
            mode_feasible if mode_feasible is not None else mode_best
        )
        selected_params = selected_payload_train["params"]
        lr_value = selected_params.get("learning_rate", cfg.learning_rates[0])
        penalty_value = selected_params.get("lambda_penalty", cfg.penalties_lambda[0])
        window_value = selected_params.get("rolling_window", cfg.rolling_windows[0])
        domain_limit_value = selected_params.get("domain_limit", cfg.domain_limits[0])
        max_weight_value = selected_params.get("max_weight", cfg.max_weights[0])
        concentration_lambda_value = selected_params.get(
            "concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0]
        )
        covariance_lambda_value = selected_params.get(
            "covariance_penalty_lambda", cfg.covariance_penalty_lambdas[0]
        )
        covariance_shrinkage_value = selected_params.get(
            "covariance_shrinkage", cfg.covariance_shrinkages[0]
        )
        entropy_lambda_value = selected_params.get("entropy_lambda", cfg.entropy_lambdas[0])
        uniform_mix_value = selected_params.get("uniform_mix", cfg.uniform_mixes[0])
        selected_lr = (
            float(lr_value) if isinstance(lr_value, (int, float)) else float(cfg.learning_rates[0])
        )
        selected_penalty = (
            float(penalty_value) if isinstance(penalty_value, (int, float)) else float(cfg.penalties_lambda[0])
        )
        selected_window = (
            int(window_value) if isinstance(window_value, (int, float)) else int(cfg.rolling_windows[0])
        )
        selected_domain_limit = (
            float(domain_limit_value)
            if isinstance(domain_limit_value, (int, float))
            else float(cfg.domain_limits[0])
        )
        selected_max_weight = (
            float(max_weight_value)
            if isinstance(max_weight_value, (int, float))
            else float(cfg.max_weights[0])
        )
        selected_concentration_lambda = (
            float(concentration_lambda_value)
            if isinstance(concentration_lambda_value, (int, float))
            else float(cfg.concentration_penalty_lambdas[0])
        )
        selected_covariance_lambda = (
            float(covariance_lambda_value)
            if isinstance(covariance_lambda_value, (int, float))
            else float(cfg.covariance_penalty_lambdas[0])
        )
        selected_covariance_shrinkage = (
            float(covariance_shrinkage_value)
            if isinstance(covariance_shrinkage_value, (int, float))
            else float(cfg.covariance_shrinkages[0])
        )
        selected_entropy_lambda = (
            float(entropy_lambda_value)
            if isinstance(entropy_lambda_value, (int, float))
            else float(cfg.entropy_lambdas[0])
        )
        selected_uniform_mix = (
            float(uniform_mix_value)
            if isinstance(uniform_mix_value, (int, float))
            else float(cfg.uniform_mixes[0])
        )

        holdout_payload = _run_online_pass(
            returns_matrix=returns_matrix,
            domains=domains,
            lr=selected_lr,
            penalty_lambda=selected_penalty,
            rolling_window=selected_window,
            steps_per_window=cfg.steps_per_window,
            domain_limit=selected_domain_limit,
            max_weight=selected_max_weight,
            concentration_penalty_lambda=selected_concentration_lambda,
            covariance_penalty_lambda=selected_covariance_lambda,
            covariance_shrinkage=selected_covariance_shrinkage,
            entropy_lambda=selected_entropy_lambda,
            uniform_mix=selected_uniform_mix,
            seed=cfg.seed,
            evaluation_start_t=split_idx,
            update_after_eval_start=(mode == "online"),
            capture_diagnostics=True,
            objective=cfg.objective,
            variance_penalty=cfg.variance_penalty,
            downside_penalty=cfg.downside_penalty,
            optimizer_type=cfg.optimizer_type,
            weight_parameterization=cfg.weight_parameterization,
            regime_k=cfg.regime_k,
            risk_on_z=rz,
            energy_z=ez,
            rates_z=ratesz,
            exog_is_stale=st,
            is_equity_open=op,
            time_offset=0,
            macro_integration=cfg.macro_integration,
            lambda_macro_explicit=cfg.lambda_macro_explicit,
            etf_step_returns=etf_matrix_full_grid,
            lambda_etf_tracking=lambda_etf_grid_holdout,
        )
        holdout_returns_realized = holdout_payload["portfolio_returns"]
        (
            holdout_sortino,
            holdout_max_dd,
            holdout_mean,
            holdout_vol,
            holdout_cumulative,
            holdout_drawdown,
        ) = _compute_metrics(holdout_returns_realized)
        holdout_domain_exposure: dict[str, float] = {}
        avg_holdout_weights = holdout_payload["avg_weights"]
        if avg_holdout_weights.size == 0:
            raise RuntimeError(
                "Holdout window too short for selected rolling_window. "
                "Increase holdout_fraction, reduce rolling_windows, or reduce walkforward_train_steps."
            )
        for idx, domain in enumerate(domains):
            holdout_domain_exposure[domain] = holdout_domain_exposure.get(domain, 0.0) + float(
                avg_holdout_weights[idx]
            )
        selected_payload: SelectedExperimentPayload = SelectedExperimentPayload(
            params={
                **selected_params,
                "selection_source": "walkforward_tuning",
                "evaluation_mode": mode,
                "holdout_sortino_ratio": holdout_sortino,
                "holdout_max_drawdown": holdout_max_dd,
                "holdout_mean_return": holdout_mean,
                "holdout_volatility": holdout_vol,
                "max_domain_exposure": float(max(holdout_domain_exposure.values()))
                if holdout_domain_exposure
                else 0.0,
            },
            portfolio_returns=holdout_returns_realized,
            cumulative=holdout_cumulative,
            drawdown=holdout_drawdown,
            domain_exposure=holdout_domain_exposure,
        )

        suffix = "" if mode == primary_mode else f"_{mode}"
        best_metrics_path = out_dir / f"{artifact_prefix}_constrained_best{suffix}_metrics.json"
        best_metrics_path.write_text(
            json.dumps(
                {
                    "strategy": f"constrained_ogd_{cfg.objective}",
                    "best_params": selected_payload["params"],
                    "domain_exposure": selected_payload["domain_exposure"],
                    "feasibility_filter": {
                        "applied_threshold": cfg.max_domain_exposure_threshold,
                        "feasible_solution_found": mode_feasible is not None,
                    },
                    "data_split": {
                        "tuning_steps": int(tuning_returns.shape[0]),
                        "holdout_steps_total": int(returns_matrix.shape[0] - split_idx),
                        "holdout_fraction": cfg.holdout_fraction,
                        "walkforward_train_steps": cfg.walkforward_train_steps,
                        "walkforward_test_steps": cfg.walkforward_test_steps,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        best_series_path = out_dir / f"{artifact_prefix}_constrained_best{suffix}_timeseries.csv"
        with best_series_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["step", "portfolio_return", "cumulative_return", "drawdown"],
            )
            writer.writeheader()
            for idx, (ret, cum, dd) in enumerate(
                zip(
                    selected_payload["portfolio_returns"].tolist(),
                    selected_payload["cumulative"].tolist(),
                    selected_payload["drawdown"].tolist(),
                    strict=False,
                )
            ):
                writer.writerow(
                    {
                        "step": idx,
                        "portfolio_return": float(ret),
                        "cumulative_return": float(cum),
                        "drawdown": float(dd),
                    }
                )

        if mode == primary_mode:
            artifacts["constrained_best_metrics"] = best_metrics_path
            artifacts["constrained_best_timeseries"] = best_series_path
        else:
            artifacts[f"constrained_best_{mode}_metrics"] = best_metrics_path
            artifacts[f"constrained_best_{mode}_timeseries"] = best_series_path

        attribution_artifacts = _build_attribution_artifacts(
            out_dir=out_dir,
            artifact_prefix=artifact_prefix,
            mode_suffix=suffix,
            holdout_payload=holdout_payload,
            kept_tokens=kept_tokens,
            token_to_meta=token_to_meta,
        )
        for key, value in attribution_artifacts.items():
            artifact_key = f"constrained_best_{key}" if mode == primary_mode else f"constrained_best_{mode}_{key}"
            artifacts[artifact_key] = value

    return artifacts


def run_optuna_search(
    project_root: pathlib.Path,
    config: ExperimentConfig | None = None,
    artifact_prefix: str = "week8",
    n_trials: int = 100,
    timeout_sec: int | None = None,
    *,
    joint_macro_mode_search: bool = False,
    output_artifact_suffix: str = "",
) -> dict[str, pathlib.Path]:
    """Run quasi-random hyperparameter search via Optuna QMCSampler (Sobol + scramble).

    Drop-in replacement for ``run_experiment_grid`` that uses Optuna's quasi-Monte Carlo
    sampler (low-discrepancy Sobol sequence) with MedianPruner for early stopping of
    unpromising trials during walk-forward validation folds. This avoids TPE/Bayesian
    overhead in very high-dimensional conditional spaces.

    Macro integration: by default ``cfg.macro_integration`` is fixed and Optuna
    conditionally tunes ``regime_k`` (rescale/both) and ``lambda_macro_explicit``
    (explicit/both). Set ``joint_macro_mode_search=True`` for one study with a
    categorical ``macro_mode`` and the same conditional hyperparameters (larger
    joint search space, single CSV for cross-mode comparison).

    Use ``output_artifact_suffix`` (e.g. ``_macro_explicit``) when running multiple
    Optuna jobs with the same ``artifact_prefix`` so constrained CSV/JSON outputs
    do not overwrite each other.

    Parallel trials: set ``cfg.optuna_n_jobs`` > 1 (or ``-1`` for auto). Optuna runs
    objectives concurrently via threads; speedup depends on how much time is spent
    in NumPy/PyTorch (GIL-released) versus pure Python. Cap ``n_jobs`` and set
    ``OMP_NUM_THREADS=1`` / ``TORCH_NUM_THREADS=1`` on big boxes to avoid oversubscription.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = config or ExperimentConfig()
    mode = cfg.primary_evaluation_mode

    returns_matrix, kept_tokens, domains, _, token_to_meta = _load_returns_and_domains(
        project_root, artifact_prefix=artifact_prefix
    )
    if returns_matrix.size == 0:
        raise RuntimeError("No returns data available. Run data build first.")

    split_idx = max(
        int((1.0 - cfg.holdout_fraction) * returns_matrix.shape[0]),
        cfg.walkforward_train_steps,
    )
    split_idx = min(split_idx, returns_matrix.shape[0] - 1)
    tuning_returns = returns_matrix[:split_idx]
    rz, ez, ratesz, st, op = _load_exogenous_regime(
        project_root, artifact_prefix, returns_matrix.shape[0]
    )
    etf_tracking_matrix = _load_etf_tracking_matrix(
        project_root, artifact_prefix, returns_matrix.shape[0]
    )
    use_etf_effective = bool(cfg.use_etf_tracking and etf_tracking_matrix is not None)
    if cfg.use_etf_tracking and etf_tracking_matrix is None:
        print(
            "WARNING: use_etf_tracking=True but ETF *_ret_1 columns missing or exogenous CSV "
            f"row count != return steps ({returns_matrix.shape[0]}); continuing without ETF term.",
            flush=True,
        )

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    constrained_stem = f"{artifact_prefix}{output_artifact_suffix}"

    def _suggest(
        trial: optuna.Trial,
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

    def objective(trial: optuna.Trial) -> float:
        lr = _suggest(trial, "lr", cfg.learning_rates, log=True)
        penalty_lambda = _suggest(trial, "penalty_lambda", cfg.penalties_lambda)
        rolling_window = int(
            trial.suggest_categorical("rolling_window", list(cfg.rolling_windows))
        )
        domain_limit = _suggest(trial, "domain_limit", cfg.domain_limits)
        max_weight = _suggest(trial, "max_weight", cfg.max_weights)
        conc_lambda = _suggest(
            trial, "concentration_penalty_lambda", cfg.concentration_penalty_lambdas, log=True
        )
        cov_lambda = _suggest(
            trial, "covariance_penalty_lambda", cfg.covariance_penalty_lambdas, log=True
        )
        cov_shrinkage = _suggest(trial, "covariance_shrinkage", cfg.covariance_shrinkages)
        ent_lambda = _suggest(trial, "entropy_lambda", cfg.entropy_lambdas)
        uniform_mix = _suggest(trial, "uniform_mix", cfg.uniform_mixes)
        var_pen = (
            _suggest(trial, "variance_penalty", cfg.variance_penalties)
            if cfg.variance_penalties
            else cfg.variance_penalty
        )
        down_pen = (
            _suggest(trial, "downside_penalty", cfg.downside_penalties)
            if cfg.downside_penalties
            else cfg.downside_penalty
        )
        if joint_macro_mode_search:
            macro_int = str(
                trial.suggest_categorical("macro_mode", ["rescale", "explicit", "both"])
            )
            # QMCSampler needs a static search space: always draw macro knobs, then mask
            # for the objective (and holdout/CSV use the same effective values).
            regime_k_draw = trial.suggest_float("regime_k", 0.0, 0.8)
            lambda_macro_draw = trial.suggest_float("lambda_macro_explicit", 0.0, 5.0)
            regime_k_val = regime_k_draw if macro_int in ("rescale", "both") else float(cfg.regime_k)
            lambda_macro_val = (
                lambda_macro_draw if macro_int in ("explicit", "both") else 0.0
            )
        else:
            macro_int = str(cfg.macro_integration)
            if macro_int in ("rescale", "both"):
                regime_k_val = trial.suggest_float("regime_k", 0.0, 0.8)
            else:
                regime_k_val = float(cfg.regime_k)
            if macro_int in ("explicit", "both"):
                lambda_macro_val = trial.suggest_float("lambda_macro_explicit", 0.0, 5.0)
            else:
                lambda_macro_val = float(cfg.lambda_macro_explicit)

        if use_etf_effective:
            lambda_etf_val = float(
                trial.suggest_categorical("lambda_etf_tracking", list(cfg.etf_tracking_lambdas))
            )
        else:
            lambda_etf_val = 0.0

        train_steps = max(cfg.walkforward_train_steps, rolling_window + 1)
        test_steps = max(cfg.walkforward_test_steps, 1)
        all_fold_returns: list[float] = []
        fold_weights_list: list[np.ndarray] = []
        fold_idx = 0
        start = 0

        while start + train_steps + test_steps <= tuning_returns.shape[0]:
            segment = tuning_returns[start : start + train_steps + test_steps]
            fold_payload = _run_online_pass(
                returns_matrix=segment,
                domains=domains,
                lr=lr,
                penalty_lambda=penalty_lambda,
                rolling_window=rolling_window,
                steps_per_window=cfg.steps_per_window,
                domain_limit=domain_limit,
                max_weight=max_weight,
                concentration_penalty_lambda=conc_lambda,
                covariance_penalty_lambda=cov_lambda,
                covariance_shrinkage=cov_shrinkage,
                entropy_lambda=ent_lambda,
                uniform_mix=uniform_mix,
                seed=cfg.seed,
                evaluation_start_t=train_steps,
                update_after_eval_start=(mode == "online"),
                objective=cfg.objective,
                variance_penalty=var_pen,
                downside_penalty=down_pen,
                optimizer_type=cfg.optimizer_type,
                weight_parameterization=cfg.weight_parameterization,
                regime_k=regime_k_val,
                risk_on_z=rz,
                energy_z=ez,
                rates_z=ratesz,
                exog_is_stale=st,
                is_equity_open=op,
                time_offset=start,
                macro_integration=macro_int,
                lambda_macro_explicit=lambda_macro_val,
                etf_step_returns=etf_tracking_matrix,
                lambda_etf_tracking=lambda_etf_val,
            )

            if (
                cfg.early_prune_enabled
                and cfg.max_domain_exposure_threshold > 0.0
                and fold_payload["avg_weights"].size == len(domains)
            ):
                fold_exposure: dict[str, float] = {}
                for idx, domain in enumerate(domains):
                    fold_exposure[domain] = fold_exposure.get(domain, 0.0) + float(
                        fold_payload["avg_weights"][idx]
                    )
                fold_max_exp = max(fold_exposure.values()) if fold_exposure else 0.0
                if fold_max_exp > cfg.max_domain_exposure_threshold * cfg.early_prune_exposure_factor:
                    raise optuna.TrialPruned()

            fold_rets = fold_payload["portfolio_returns"]
            if isinstance(fold_rets, np.ndarray) and fold_rets.size > 0:
                all_fold_returns.extend(fold_rets.tolist())
                if fold_payload["avg_weights"].size > 0:
                    fold_weights_list.append(fold_payload["avg_weights"])

            if all_fold_returns:
                intermediate_sortino = _compute_metrics(
                    np.array(all_fold_returns, dtype=float)
                )[0]
                trial.report(intermediate_sortino, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            fold_idx += 1
            start += test_steps

        if not all_fold_returns:
            return float("-inf")

        returns_arr = np.array(all_fold_returns, dtype=float)
        sortino, max_dd, mean_return, volatility, _, _ = _compute_metrics(returns_arr)

        trial.set_user_attr("max_drawdown", float(max_dd))
        trial.set_user_attr("mean_return", float(mean_return))
        trial.set_user_attr("volatility", float(volatility))

        if fold_weights_list:
            avg_weights = np.mean(np.stack(fold_weights_list), axis=0)
            domain_exposure: dict[str, float] = {}
            for idx, domain in enumerate(domains):
                domain_exposure[domain] = domain_exposure.get(domain, 0.0) + float(
                    avg_weights[idx]
                )
            trial.set_user_attr(
                "max_domain_exposure",
                float(max(domain_exposure.values())) if domain_exposure else 0.0,
            )
            trial.set_user_attr("domain_exposure_json", json.dumps(domain_exposure))

        return sortino

    # Pruner: only prune after many folds so intermediate Sortino has signal.
    # With n_warmup_steps=2 we were pruning after ~1% of folds (noise) and killed 80%+ of trials.
    _train = max(cfg.walkforward_train_steps, 96 + 1)
    _test = max(cfg.walkforward_test_steps, 1)
    _n_folds = max(1, (tuning_returns.shape[0] - _train) // _test)
    n_warmup_steps = min(25, max(5, _n_folds // 10))  # warmup ~10% of folds, at least 5, cap 25
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.QMCSampler(seed=cfg.seed, qmc_type="sobol"),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=n_warmup_steps),
    )

    trial_start = time.perf_counter()

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = time.perf_counter() - trial_start
        n_complete = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )
        if trial.state == optuna.trial.TrialState.PRUNED:
            status = "PRUNED"
        elif trial.value is not None:
            status = f"sortino={trial.value:.4f}"
        else:
            status = "FAILED"
        best_val = study.best_value if n_complete > 0 else float("nan")
        print(
            f"  Trial {trial.number}: {status} "
            f"| best={best_val:.4f} | {n_complete} complete, {n_pruned} pruned "
            f"| {elapsed / 60:.1f}m elapsed",
            flush=True,
        )

    n_jobs = int(cfg.optuna_n_jobs)
    if n_jobs == 0:
        raise ValueError("ExperimentConfig.optuna_n_jobs must be non-zero (use 1 for sequential, -1 for auto).")

    print(f"\n{'='*60}", flush=True)
    print(
        f"OPTUNA QUASI-RANDOM SEARCH: {n_trials} trials (QMCSampler Sobol + MedianPruner), "
        f"n_jobs={n_jobs}",
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    trial_log_lock = threading.Lock()

    def _log_trial_locked(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with trial_log_lock:
            _log_trial(study, trial)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        callbacks=[_log_trial_locked],
        show_progress_bar=False,
        n_jobs=n_jobs,
    )

    search_elapsed = time.perf_counter() - search_start
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]

    print(f"\n{'='*60}", flush=True)
    print(
        f"Optuna search complete — {len(completed_trials)} completed, "
        f"{len(pruned_trials)} pruned in {search_elapsed / 60:.1f}m ({search_elapsed / 3600:.1f}h)",
        flush=True,
    )
    print(f"{'='*60}\n", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        p = trial.params
        if joint_macro_mode_search:
            macro_mode = str(p.get("macro_mode", cfg.macro_integration))
            eff_regime_k = (
                float(p["regime_k"]) if macro_mode in ("rescale", "both") else float(cfg.regime_k)
            )
            eff_lambda_macro = (
                float(p["lambda_macro_explicit"])
                if macro_mode in ("explicit", "both")
                else 0.0
            )
            macro_integration_out = macro_mode
        else:
            eff_regime_k = float(p["regime_k"]) if "regime_k" in p else float(cfg.regime_k)
            eff_lambda_macro = (
                float(p["lambda_macro_explicit"])
                if "lambda_macro_explicit" in p
                else float(cfg.lambda_macro_explicit)
            )
            macro_integration_out = str(cfg.macro_integration)

        summary_rows.append(
            {
                "learning_rate": p.get("lr", cfg.learning_rates[0]),
                "lambda_penalty": p.get("penalty_lambda", cfg.penalties_lambda[0]),
                "rolling_window": p.get("rolling_window", cfg.rolling_windows[0]),
                "domain_limit": p.get("domain_limit", cfg.domain_limits[0]),
                "max_weight": p.get("max_weight", cfg.max_weights[0]),
                "uniform_mix": p.get("uniform_mix", cfg.uniform_mixes[0]),
                "evaluation_mode": mode,
                "sortino_ratio": trial.value,
                "max_drawdown": trial.user_attrs.get("max_drawdown", 0.0),
                "mean_return": trial.user_attrs.get("mean_return", 0.0),
                "volatility": trial.user_attrs.get("volatility", 0.0),
                "concentration_penalty_lambda": p.get(
                    "concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0]
                ),
                "covariance_penalty_lambda": p.get(
                    "covariance_penalty_lambda", cfg.covariance_penalty_lambdas[0]
                ),
                "covariance_shrinkage": p.get(
                    "covariance_shrinkage", cfg.covariance_shrinkages[0]
                ),
                "entropy_lambda": p.get("entropy_lambda", cfg.entropy_lambdas[0]),
                "variance_penalty": p.get("variance_penalty", cfg.variance_penalty),
                "downside_penalty": p.get("downside_penalty", cfg.downside_penalty),
                "regime_k": eff_regime_k,
                "macro_integration": macro_integration_out,
                "lambda_macro_explicit": eff_lambda_macro,
                "lambda_etf_tracking": float(p.get("lambda_etf_tracking", 0.0)),
                "max_domain_exposure": trial.user_attrs.get("max_domain_exposure", 0.0),
                "max_domain_exposure_threshold": cfg.max_domain_exposure_threshold,
                "domain_exposure_json": trial.user_attrs.get("domain_exposure_json", "{}"),
            }
        )

    summary_path = out_dir / f"{constrained_stem}_constrained_experiment_grid.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            handle.write("")

    if not completed_trials:
        raise RuntimeError("No valid Optuna trial completed.")

    best_trial = study.best_trial
    feasible_trials = [
        t
        for t in completed_trials
        if t.user_attrs.get("max_domain_exposure", 1.0) <= cfg.max_domain_exposure_threshold
    ]
    if feasible_trials:
        best_trial = max(
            feasible_trials,
            key=lambda t: t.value if t.value is not None else float("-inf"),
        )

    bp = best_trial.params

    if joint_macro_mode_search:
        holdout_macro = str(bp.get("macro_mode", cfg.macro_integration))
        holdout_regime_k = (
            float(bp["regime_k"])
            if holdout_macro in ("rescale", "both") and "regime_k" in bp
            else float(cfg.regime_k)
        )
        holdout_lambda_macro = (
            float(bp["lambda_macro_explicit"])
            if holdout_macro in ("explicit", "both") and "lambda_macro_explicit" in bp
            else 0.0
        )
    else:
        holdout_macro = str(cfg.macro_integration)
        holdout_lambda_macro = (
            float(bp["lambda_macro_explicit"])
            if "lambda_macro_explicit" in bp
            else float(cfg.lambda_macro_explicit)
        )
        holdout_regime_k = (
            float(bp["regime_k"]) if "regime_k" in bp else float(cfg.regime_k)
        )

    holdout_lambda_etf = float(bp.get("lambda_etf_tracking", 0.0)) if use_etf_effective else 0.0

    holdout_payload = _run_online_pass(
        returns_matrix=returns_matrix,
        domains=domains,
        lr=float(bp.get("lr", cfg.learning_rates[0])),
        penalty_lambda=float(bp.get("penalty_lambda", cfg.penalties_lambda[0])),
        rolling_window=int(bp.get("rolling_window", cfg.rolling_windows[0])),
        steps_per_window=cfg.steps_per_window,
        domain_limit=float(bp.get("domain_limit", cfg.domain_limits[0])),
        max_weight=float(bp.get("max_weight", cfg.max_weights[0])),
        concentration_penalty_lambda=float(
            bp.get("concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0])
        ),
        covariance_penalty_lambda=float(
            bp.get("covariance_penalty_lambda", cfg.covariance_penalty_lambdas[0])
        ),
        covariance_shrinkage=float(
            bp.get("covariance_shrinkage", cfg.covariance_shrinkages[0])
        ),
        entropy_lambda=float(bp.get("entropy_lambda", cfg.entropy_lambdas[0])),
        uniform_mix=float(bp.get("uniform_mix", cfg.uniform_mixes[0])),
        seed=cfg.seed,
        evaluation_start_t=split_idx,
        update_after_eval_start=(mode == "online"),
        capture_diagnostics=True,
        objective=cfg.objective,
        variance_penalty=float(bp.get("variance_penalty", cfg.variance_penalty)),
        downside_penalty=float(bp.get("downside_penalty", cfg.downside_penalty)),
        optimizer_type=cfg.optimizer_type,
        weight_parameterization=cfg.weight_parameterization,
        regime_k=holdout_regime_k,
        risk_on_z=rz,
        energy_z=ez,
        rates_z=ratesz,
        exog_is_stale=st,
        is_equity_open=op,
        time_offset=0,
        macro_integration=holdout_macro,
        lambda_macro_explicit=holdout_lambda_macro,
        etf_step_returns=etf_tracking_matrix,
        lambda_etf_tracking=holdout_lambda_etf,
    )

    holdout_returns = holdout_payload["portfolio_returns"]
    (
        holdout_sortino,
        holdout_max_dd,
        holdout_mean,
        holdout_vol,
        holdout_cumulative,
        holdout_drawdown,
    ) = _compute_metrics(holdout_returns)

    avg_holdout_weights = holdout_payload["avg_weights"]
    if avg_holdout_weights.size == 0:
        raise RuntimeError(
            "Holdout window too short for selected rolling_window. "
            "Increase holdout_fraction or reduce rolling_windows."
        )

    holdout_domain_exposure: dict[str, float] = {}
    for idx, domain in enumerate(domains):
        holdout_domain_exposure[domain] = holdout_domain_exposure.get(domain, 0.0) + float(
            avg_holdout_weights[idx]
        )

    selected_params: dict[str, Any] = {
        "learning_rate": float(bp.get("lr", cfg.learning_rates[0])),
        "lambda_penalty": float(bp.get("penalty_lambda", cfg.penalties_lambda[0])),
        "rolling_window": int(bp.get("rolling_window", cfg.rolling_windows[0])),
        "domain_limit": float(bp.get("domain_limit", cfg.domain_limits[0])),
        "max_weight": float(bp.get("max_weight", cfg.max_weights[0])),
        "uniform_mix": float(bp.get("uniform_mix", cfg.uniform_mixes[0])),
        "evaluation_mode": mode,
        "weight_parameterization": cfg.weight_parameterization,
        "concentration_penalty_lambda": float(
            bp.get("concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0])
        ),
        "covariance_penalty_lambda": float(
            bp.get("covariance_penalty_lambda", cfg.covariance_penalty_lambdas[0])
        ),
        "covariance_shrinkage": float(
            bp.get("covariance_shrinkage", cfg.covariance_shrinkages[0])
        ),
        "entropy_lambda": float(bp.get("entropy_lambda", cfg.entropy_lambdas[0])),
        "variance_penalty": float(bp.get("variance_penalty", cfg.variance_penalty)),
        "downside_penalty": float(bp.get("downside_penalty", cfg.downside_penalty)),
        "regime_k": holdout_regime_k,
        "macro_integration": holdout_macro,
        "lambda_macro_explicit": holdout_lambda_macro,
        "lambda_etf_tracking": holdout_lambda_etf,
        "selection_source": "optuna_qmc",
        "holdout_sortino_ratio": holdout_sortino,
        "holdout_max_drawdown": holdout_max_dd,
        "holdout_mean_return": holdout_mean,
        "holdout_volatility": holdout_vol,
        "max_domain_exposure": float(max(holdout_domain_exposure.values()))
        if holdout_domain_exposure
        else 0.0,
    }

    best_metrics_path = out_dir / f"{constrained_stem}_constrained_best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(
            {
                "strategy": (
                    f"constrained_optuna_{cfg.objective}_{holdout_macro}"
                    f"{'_joint_macro' if joint_macro_mode_search else ''}"
                ),
                "best_params": selected_params,
                "domain_exposure": holdout_domain_exposure,
                "feasibility_filter": {
                    "applied_threshold": cfg.max_domain_exposure_threshold,
                    "feasible_solution_found": len(feasible_trials) > 0,
                },
                "data_split": {
                    "tuning_steps": int(tuning_returns.shape[0]),
                    "holdout_steps_total": int(returns_matrix.shape[0] - split_idx),
                    "holdout_fraction": cfg.holdout_fraction,
                    "walkforward_train_steps": cfg.walkforward_train_steps,
                    "walkforward_test_steps": cfg.walkforward_test_steps,
                },
                "optuna_summary": {
                    "n_trials": n_trials,
                    "n_jobs": n_jobs,
                    "completed_trials": len(completed_trials),
                    "pruned_trials": len(pruned_trials),
                    "best_trial_number": best_trial.number,
                    "search_time_sec": search_elapsed,
                    "sampler": "QMCSampler(sobol)",
                    "joint_macro_mode_search": joint_macro_mode_search,
                    "output_artifact_suffix": output_artifact_suffix,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_series_path = out_dir / f"{constrained_stem}_constrained_best_timeseries.csv"
    with best_series_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "portfolio_return", "cumulative_return", "drawdown"],
        )
        writer.writeheader()
        for idx, (ret, cum, dd) in enumerate(
            zip(
                holdout_returns.tolist(),
                holdout_cumulative.tolist(),
                holdout_drawdown.tolist(),
                strict=False,
            )
        ):
            writer.writerow(
                {
                    "step": idx,
                    "portfolio_return": float(ret),
                    "cumulative_return": float(cum),
                    "drawdown": float(dd),
                }
            )

    artifacts: dict[str, pathlib.Path] = {
        "constrained_grid": summary_path,
        "constrained_best_metrics": best_metrics_path,
        "constrained_best_timeseries": best_series_path,
    }

    attribution_artifacts = _build_attribution_artifacts(
        out_dir=out_dir,
        artifact_prefix=constrained_stem,
        mode_suffix="",
        holdout_payload=holdout_payload,
        kept_tokens=kept_tokens,
        token_to_meta=token_to_meta,
    )
    for key, value in attribution_artifacts.items():
        artifacts[f"constrained_best_{key}"] = value

    return artifacts
