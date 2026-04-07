"""First-pass constrained OGD/SGD experiments for domain-aware allocation."""

from __future__ import annotations

import csv
import itertools
import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, TypedDict

import numpy as np
import torch

from src.baseline import _build_price_matrix, _compute_returns, _max_drawdown, _read_csv


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for rudimentary constrained optimizer runs."""

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
    enable_equity_signal: bool = False
    equity_signal_lambda: float = 0.0
    equity_signal_lambdas: tuple[float, ...] = ()
    equity_signal_mapping_csv: str | None = None
    optimizer_type: str = "sgd"
    evaluation_modes: tuple[str, ...] = ("online",)
    primary_evaluation_mode: str = "online"
    enable_two_stage_search: bool = True
    stage2_top_k: int = 8
    max_parallel_workers: int = 1
    early_prune_enabled: bool = True
    early_prune_exposure_factor: float = 1.5
    max_domain_exposure_threshold: float = 1.0
    holdout_fraction: float = 0.2
    walkforward_train_steps: int = 240
    walkforward_test_steps: int = 48
    seed: int = 7


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
    equity_signal_matrix: np.ndarray | None = None,
    equity_signal_lambda: float = 0.0,
    optimizer_type: str = "sgd",
) -> OnlinePassPayload:
    torch.manual_seed(seed)
    if returns_matrix.shape[0] <= rolling_window:
        return OnlinePassPayload(
            portfolio_returns=np.array([], dtype=float),
            avg_weights=np.array([], dtype=float),
            eval_weights=np.zeros((0, 0), dtype=float),
            eval_asset_returns=np.zeros((0, 0), dtype=float),
        )

    n_assets = returns_matrix.shape[1]
    logits = torch.zeros(n_assets, dtype=torch.float32, requires_grad=True)
    optimizer = (
        torch.optim.Adam([logits], lr=lr)
        if optimizer_type == "adam"
        else torch.optim.SGD([logits], lr=lr)
    )

    realized_returns: list[float] = []
    weight_snapshots: list[np.ndarray] = []
    eval_asset_returns: list[np.ndarray] = []

    eval_start = evaluation_start_t if evaluation_start_t is not None else rolling_window
    for t in range(rolling_window, returns_matrix.shape[0]):
        step_returns_np = np.array(returns_matrix[t], dtype=float)
        available_mask_np = np.isfinite(step_returns_np)
        if not np.any(available_mask_np):
            continue
        available_mask = torch.tensor(available_mask_np.astype(np.float32), dtype=torch.float32)
        window_np = np.array(returns_matrix[t - rolling_window : t], dtype=float)
        window_np = np.nan_to_num(window_np, nan=0.0)
        window = torch.tensor(window_np, dtype=torch.float32)

        should_update = update_after_eval_start or (t < eval_start)
        if should_update:
            for _ in range(steps_per_window):
                optimizer.zero_grad()
                weights_raw = torch.softmax(logits, dim=0)
                masked_raw = weights_raw * available_mask
                masked_raw_sum = torch.sum(masked_raw)
                if float(masked_raw_sum.detach().cpu().item()) <= 0.0:
                    masked_raw = available_mask
                    masked_raw_sum = torch.sum(masked_raw)
                masked_raw = masked_raw / torch.clamp(masked_raw_sum, min=1e-8)
                uniform_weights = available_mask / torch.clamp(torch.sum(available_mask), min=1e-8)
                # Guaranteed diversification floor to prevent single-domain collapse.
                weights = (1.0 - uniform_mix) * masked_raw + uniform_mix * uniform_weights
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
                        variance_penalty=variance_penalty,
                        downside_penalty=downside_penalty,
                    )
                else:
                    return_term = _sortino_torch(portfolio)
                if equity_signal_matrix is not None and t < equity_signal_matrix.shape[0]:
                    signal_step_np = np.nan_to_num(equity_signal_matrix[t], nan=0.0)
                    signal_step = torch.tensor(signal_step_np, dtype=torch.float32)
                    signal_reward = torch.sum(weights * signal_step)
                else:
                    signal_reward = torch.tensor(0.0, dtype=torch.float32)
                obj = (
                    return_term
                    - penalty_lambda
                    * _domain_penalty(
                        weights=weights,
                        domains=domains,
                        domain_limit=domain_limit,
                    )
                    - concentration_penalty_lambda * concentration_penalty
                    - covariance_penalty_lambda * covariance_penalty
                    + entropy_lambda * entropy_bonus
                    + equity_signal_lambda * signal_reward
                )
                loss = -obj
                loss.backward()
                optimizer.step()

        weights_raw_eval = torch.softmax(logits, dim=0)
        masked_eval = weights_raw_eval * available_mask
        masked_eval_sum = torch.sum(masked_eval)
        if float(masked_eval_sum.detach().cpu().item()) <= 0.0:
            masked_eval = available_mask
            masked_eval_sum = torch.sum(masked_eval)
        masked_eval = masked_eval / torch.clamp(masked_eval_sum, min=1e-8)
        uniform_weights_eval = available_mask / torch.clamp(torch.sum(available_mask), min=1e-8)
        current_weights = (
            (1.0 - uniform_mix) * masked_eval + uniform_mix * uniform_weights_eval
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
        "sortino_ratio": sortino,
        "max_drawdown": float(max_dd),
        "mean_return": mean_return,
        "volatility": volatility,
        "concentration_penalty_lambda": candidate.concentration_penalty_lambda,
        "covariance_penalty_lambda": candidate.covariance_penalty_lambda,
        "covariance_shrinkage": candidate.covariance_shrinkage,
        "entropy_lambda": candidate.entropy_lambda,
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
) -> dict[str, pathlib.Path]:
    """Run Bayesian hyperparameter optimization via Optuna TPE sampler.

    Drop-in replacement for ``run_experiment_grid`` that uses Optuna's
    Tree-structured Parzen Estimator with MedianPruner for early stopping
    of unpromising trials during walk-forward validation folds.
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

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_signal_matrix: np.ndarray | None = None
    equity_signal_artifacts: dict[str, pathlib.Path] = {}
    if cfg.enable_equity_signal:
        from src.equity_signal import EquitySignalConfig, build_asset_equity_signal_matrix

        mapping_path = (
            pathlib.Path(cfg.equity_signal_mapping_csv)
            if cfg.equity_signal_mapping_csv is not None
            else None
        )
        equity_signal_matrix, equity_signal_artifacts = build_asset_equity_signal_matrix(
            project_root=project_root,
            artifact_prefix=artifact_prefix,
            asset_domains=domains,
            config=EquitySignalConfig(mapping_csv=mapping_path),
        )
        if equity_signal_matrix.shape != returns_matrix.shape:
            raise RuntimeError(
                f"Equity signal shape mismatch: expected {returns_matrix.shape}, got {equity_signal_matrix.shape}"
            )
    tuning_equity_signal = equity_signal_matrix[:split_idx] if equity_signal_matrix is not None else None

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
        eq_signal_lambda = (
            _suggest(trial, "equity_signal_lambda", cfg.equity_signal_lambdas)
            if cfg.equity_signal_lambdas
            else cfg.equity_signal_lambda
        )

        train_steps = max(cfg.walkforward_train_steps, rolling_window + 1)
        test_steps = max(cfg.walkforward_test_steps, 1)
        all_fold_returns: list[float] = []
        fold_weights_list: list[np.ndarray] = []
        fold_idx = 0
        start = 0

        while start + train_steps + test_steps <= tuning_returns.shape[0]:
            segment = tuning_returns[start : start + train_steps + test_steps]
            segment_signal = (
                tuning_equity_signal[start : start + train_steps + test_steps]
                if tuning_equity_signal is not None
                else None
            )
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
                equity_signal_matrix=segment_signal,
                equity_signal_lambda=eq_signal_lambda,
                optimizer_type=cfg.optimizer_type,
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
        trial.set_user_attr("equity_signal_lambda", float(eq_signal_lambda))

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
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
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

    print(f"\n{'='*60}", flush=True)
    print(f"OPTUNA BAYESIAN SEARCH: {n_trials} trials (TPE + MedianPruner)", flush=True)
    print(f"{'='*60}", flush=True)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        callbacks=[_log_trial],
        show_progress_bar=False,
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
        summary_rows.append(
            {
                "learning_rate": trial.params.get("lr", cfg.learning_rates[0]),
                "lambda_penalty": trial.params.get("penalty_lambda", cfg.penalties_lambda[0]),
                "rolling_window": trial.params.get("rolling_window", cfg.rolling_windows[0]),
                "domain_limit": trial.params.get("domain_limit", cfg.domain_limits[0]),
                "max_weight": trial.params.get("max_weight", cfg.max_weights[0]),
                "uniform_mix": trial.params.get("uniform_mix", cfg.uniform_mixes[0]),
                "evaluation_mode": mode,
                "sortino_ratio": trial.value,
                "max_drawdown": trial.user_attrs.get("max_drawdown", 0.0),
                "mean_return": trial.user_attrs.get("mean_return", 0.0),
                "volatility": trial.user_attrs.get("volatility", 0.0),
                "concentration_penalty_lambda": trial.params.get(
                    "concentration_penalty_lambda", cfg.concentration_penalty_lambdas[0]
                ),
                "covariance_penalty_lambda": trial.params.get(
                    "covariance_penalty_lambda", cfg.covariance_penalty_lambdas[0]
                ),
                "covariance_shrinkage": trial.params.get(
                    "covariance_shrinkage", cfg.covariance_shrinkages[0]
                ),
                "entropy_lambda": trial.params.get("entropy_lambda", cfg.entropy_lambdas[0]),
                "variance_penalty": trial.params.get("variance_penalty", cfg.variance_penalty),
                "downside_penalty": trial.params.get("downside_penalty", cfg.downside_penalty),
                "equity_signal_lambda": trial.params.get("equity_signal_lambda", cfg.equity_signal_lambda),
                "max_domain_exposure": trial.user_attrs.get("max_domain_exposure", 0.0),
                "max_domain_exposure_threshold": cfg.max_domain_exposure_threshold,
                "domain_exposure_json": trial.user_attrs.get("domain_exposure_json", "{}"),
            }
        )

    summary_path = out_dir / f"{artifact_prefix}_constrained_experiment_grid.csv"
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
        equity_signal_matrix=equity_signal_matrix,
        equity_signal_lambda=float(bp.get("equity_signal_lambda", cfg.equity_signal_lambda)),
        optimizer_type=cfg.optimizer_type,
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
        "equity_signal_lambda": float(bp.get("equity_signal_lambda", cfg.equity_signal_lambda)),
        "selection_source": "optuna_bayesian",
        "holdout_sortino_ratio": holdout_sortino,
        "holdout_max_drawdown": holdout_max_dd,
        "holdout_mean_return": holdout_mean,
        "holdout_volatility": holdout_vol,
        "max_domain_exposure": float(max(holdout_domain_exposure.values()))
        if holdout_domain_exposure
        else 0.0,
    }

    best_metrics_path = out_dir / f"{artifact_prefix}_constrained_best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(
            {
                "strategy": f"constrained_optuna_{cfg.objective}",
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
                    "completed_trials": len(completed_trials),
                    "pruned_trials": len(pruned_trials),
                    "best_trial_number": best_trial.number,
                    "search_time_sec": search_elapsed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_series_path = out_dir / f"{artifact_prefix}_constrained_best_timeseries.csv"
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
    for key, value in equity_signal_artifacts.items():
        artifacts[key] = value

    attribution_artifacts = _build_attribution_artifacts(
        out_dir=out_dir,
        artifact_prefix=artifact_prefix,
        mode_suffix="",
        holdout_payload=holdout_payload,
        kept_tokens=kept_tokens,
        token_to_meta=token_to_meta,
    )
    for key, value in attribution_artifacts.items():
        artifacts[f"constrained_best_{key}"] = value

    return artifacts
