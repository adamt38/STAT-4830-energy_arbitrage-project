"""First-pass constrained OGD/SGD experiments for domain-aware allocation."""

from __future__ import annotations

import csv
import itertools
import json
import pathlib
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch

from src.baseline import _build_price_matrix, _compute_returns, _max_drawdown, _read_csv


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for rudimentary constrained optimizer runs."""

    learning_rates: tuple[float, ...] = (0.02, 0.05, 0.1)
    penalties_lambda: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    rolling_windows: tuple[int, ...] = (24, 48)
    steps_per_window: int = 1
    domain_limit: float = 0.35
    max_weight: float = 0.25
    concentration_penalty_lambda: float = 50.0
    entropy_lambda: float = 0.02
    uniform_mix: float = 0.8
    max_domain_exposure_threshold: float = 1.0
    holdout_fraction: float = 0.2
    walkforward_train_steps: int = 240
    walkforward_test_steps: int = 48
    seed: int = 7


class OnlinePassPayload(TypedDict):
    """Typed payload returned by one online optimization pass."""

    portfolio_returns: np.ndarray
    avg_weights: np.ndarray


class SelectedExperimentPayload(TypedDict):
    """Typed payload for a selected experiment candidate."""

    params: dict[str, object]
    portfolio_returns: np.ndarray
    cumulative: np.ndarray
    drawdown: np.ndarray
    domain_exposure: dict[str, float]


def _sortino_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean_return = torch.mean(returns)
    downside = torch.clamp(-returns, min=0.0)
    downside_semivar = torch.mean(downside * downside)
    downside_dev = torch.sqrt(downside_semivar + eps)
    return mean_return / downside_dev


def _load_returns_and_domains(project_root: pathlib.Path) -> tuple[np.ndarray, list[str], dict[str, str]]:
    markets_rows = _read_csv(project_root / "data" / "processed" / "week8_markets_filtered.csv")
    history_rows = _read_csv(project_root / "data" / "processed" / "week8_price_history.csv")
    _, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)
    token_to_domain = {row["yes_token_id"]: row["domain"] for row in markets_rows}
    kept_domains = [token_to_domain.get(token, "other") for token in kept_tokens]
    return returns_matrix, kept_domains, token_to_domain


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
    entropy_lambda: float,
    uniform_mix: float,
    seed: int,
    evaluation_start_t: int | None = None,
) -> OnlinePassPayload:
    torch.manual_seed(seed)
    if returns_matrix.shape[0] <= rolling_window:
        return OnlinePassPayload(
            portfolio_returns=np.array([], dtype=float),
            avg_weights=np.array([], dtype=float),
        )

    n_assets = returns_matrix.shape[1]
    logits = torch.zeros(n_assets, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=lr)

    realized_returns: list[float] = []
    weight_snapshots: list[np.ndarray] = []

    eval_start = evaluation_start_t if evaluation_start_t is not None else rolling_window
    for t in range(rolling_window, returns_matrix.shape[0]):
        window = torch.tensor(
            returns_matrix[t - rolling_window : t],
            dtype=torch.float32,
        )

        for _ in range(steps_per_window):
            optimizer.zero_grad()
            weights_raw = torch.softmax(logits, dim=0)
            uniform_weights = torch.ones_like(weights_raw) / float(n_assets)
            # Guaranteed diversification floor to prevent single-domain collapse.
            weights = (1.0 - uniform_mix) * weights_raw + uniform_mix * uniform_weights
            portfolio = window @ weights
            # Penalize excessive concentration and reward spread.
            concentration_penalty = torch.sum(torch.clamp(weights - max_weight, min=0.0).pow(2))
            entropy_bonus = -torch.sum(weights * torch.log(weights + 1e-8))
            objective = _sortino_torch(portfolio) - penalty_lambda * _domain_penalty(
                weights=weights,
                domains=domains,
                domain_limit=domain_limit,
            ) - concentration_penalty_lambda * concentration_penalty + entropy_lambda * entropy_bonus
            loss = -objective
            loss.backward()
            optimizer.step()

        weights_raw_eval = torch.softmax(logits, dim=0)
        uniform_weights_eval = torch.ones_like(weights_raw_eval) / float(n_assets)
        current_weights = (
            (1.0 - uniform_mix) * weights_raw_eval + uniform_mix * uniform_weights_eval
        ).detach().cpu().numpy()
        if t >= eval_start:
            weight_snapshots.append(current_weights)
            realized_returns.append(float(returns_matrix[t] @ current_weights))

    avg_weights = (
        np.mean(np.array(weight_snapshots), axis=0)
        if weight_snapshots
        else np.zeros(n_assets, dtype=float)
    )
    return OnlinePassPayload(
        portfolio_returns=np.array(realized_returns, dtype=float),
        avg_weights=avg_weights,
    )


def _compute_metrics(returns: np.ndarray) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Compute return/risk metrics and equity/drawdown series."""
    if returns.size == 0:
        empty = np.array([], dtype=float)
        return 0.0, 0.0, 0.0, 0.0, empty, empty
    cumulative = np.cumprod(1.0 + returns)
    drawdown, max_dd = _max_drawdown(cumulative)
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
    sortino = float(np.mean(returns) / (downside_std + 1e-8))
    mean_return = float(np.mean(returns))
    volatility = float(np.std(returns))
    return sortino, float(max_dd), mean_return, volatility, cumulative, drawdown


def _run_walkforward_validation(
    tuning_returns: np.ndarray,
    domains: list[str],
    cfg: ExperimentConfig,
    lr: float,
    penalty_lambda: float,
    rolling_window: int,
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
            domain_limit=cfg.domain_limit,
            max_weight=cfg.max_weight,
            concentration_penalty_lambda=cfg.concentration_penalty_lambda,
            entropy_lambda=cfg.entropy_lambda,
            uniform_mix=cfg.uniform_mix,
            seed=cfg.seed,
            evaluation_start_t=train_steps,
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
            domain_limit=cfg.domain_limit,
            max_weight=cfg.max_weight,
            concentration_penalty_lambda=cfg.concentration_penalty_lambda,
            entropy_lambda=cfg.entropy_lambda,
            uniform_mix=cfg.uniform_mix,
            seed=cfg.seed,
        )

    avg_weights = np.mean(np.array(fold_weights), axis=0) if fold_weights else np.array([], dtype=float)
    return OnlinePassPayload(
        portfolio_returns=np.array(fold_returns, dtype=float),
        avg_weights=avg_weights,
    )


def run_experiment_grid(
    project_root: pathlib.Path,
    config: ExperimentConfig | None = None,
    artifact_prefix: str = "week8",
) -> dict[str, pathlib.Path]:
    """Run rudimentary constrained experiments and persist results."""
    cfg = config or ExperimentConfig()
    returns_matrix, domains, _ = _load_returns_and_domains(project_root)
    if returns_matrix.size == 0:
        raise RuntimeError("No returns data available. Run data build first.")
    split_idx = max(int((1.0 - cfg.holdout_fraction) * returns_matrix.shape[0]), cfg.walkforward_train_steps)
    split_idx = min(split_idx, returns_matrix.shape[0] - 1)
    min_window = min(cfg.rolling_windows) if cfg.rolling_windows else 1
    holdout_start = max(split_idx - min_window, 0)
    tuning_returns = returns_matrix[:split_idx]
    holdout_returns = returns_matrix[holdout_start:]

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    best_payload: SelectedExperimentPayload | None = None
    best_sortino = -np.inf
    feasible_payload: SelectedExperimentPayload | None = None
    feasible_best_sortino = -np.inf

    for lr, penalty_lambda, window in itertools.product(
        cfg.learning_rates,
        cfg.penalties_lambda,
        cfg.rolling_windows,
    ):
        payload = _run_walkforward_validation(
            tuning_returns=tuning_returns,
            domains=domains,
            cfg=cfg,
            lr=lr,
            penalty_lambda=penalty_lambda,
            rolling_window=window,
        )
        portfolio_returns = payload["portfolio_returns"]
        if not isinstance(portfolio_returns, np.ndarray) or portfolio_returns.size == 0:
            continue

        sortino, max_dd, mean_return, volatility, cumulative, drawdown = _compute_metrics(portfolio_returns)

        avg_weights = payload["avg_weights"]
        domain_exposure: dict[str, float] = {}
        for idx, domain in enumerate(domains):
            domain_exposure[domain] = domain_exposure.get(domain, 0.0) + float(avg_weights[idx])

        row = {
            "learning_rate": lr,
            "lambda_penalty": penalty_lambda,
            "rolling_window": window,
            "max_weight": cfg.max_weight,
            "uniform_mix": cfg.uniform_mix,
            "sortino_ratio": sortino,
            "max_drawdown": float(max_dd),
            "mean_return": mean_return,
            "volatility": volatility,
            "concentration_penalty_lambda": cfg.concentration_penalty_lambda,
            "entropy_lambda": cfg.entropy_lambda,
            "max_domain_exposure": float(max(domain_exposure.values())) if domain_exposure else 0.0,
            "max_domain_exposure_threshold": cfg.max_domain_exposure_threshold,
            "domain_exposure_json": json.dumps(domain_exposure),
        }
        summary_rows.append(row)

        if sortino > best_sortino:
            best_sortino = sortino
            best_payload = SelectedExperimentPayload(
                params=row,
                portfolio_returns=portfolio_returns,
                cumulative=cumulative,
                drawdown=drawdown,
                domain_exposure=domain_exposure,
            )

        if (
            float(row["max_domain_exposure"]) <= cfg.max_domain_exposure_threshold
            and sortino > feasible_best_sortino
        ):
            feasible_best_sortino = sortino
            feasible_payload = SelectedExperimentPayload(
                params=row,
                portfolio_returns=portfolio_returns,
                cumulative=cumulative,
                drawdown=drawdown,
                domain_exposure=domain_exposure,
            )

    summary_path = out_dir / f"{artifact_prefix}_constrained_experiment_grid.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            handle.write("")

    if best_payload is None:
        raise RuntimeError("No valid constrained experiment produced returns.")
    selected_payload_train: SelectedExperimentPayload = (
        feasible_payload if feasible_payload is not None else best_payload
    )
    selected_params = selected_payload_train["params"]
    lr_value = selected_params.get("learning_rate", cfg.learning_rates[0])
    penalty_value = selected_params.get("lambda_penalty", cfg.penalties_lambda[0])
    window_value = selected_params.get("rolling_window", cfg.rolling_windows[0])
    selected_lr = float(lr_value) if isinstance(lr_value, (int, float)) else float(cfg.learning_rates[0])
    selected_penalty = (
        float(penalty_value) if isinstance(penalty_value, (int, float)) else float(cfg.penalties_lambda[0])
    )
    selected_window = (
        int(window_value) if isinstance(window_value, (int, float)) else int(cfg.rolling_windows[0])
    )
    holdout_payload = _run_online_pass(
        returns_matrix=holdout_returns,
        domains=domains,
        lr=selected_lr,
        penalty_lambda=selected_penalty,
        rolling_window=selected_window,
        steps_per_window=cfg.steps_per_window,
        domain_limit=cfg.domain_limit,
        max_weight=cfg.max_weight,
        concentration_penalty_lambda=cfg.concentration_penalty_lambda,
        entropy_lambda=cfg.entropy_lambda,
        uniform_mix=cfg.uniform_mix,
        seed=cfg.seed,
    )
    holdout_returns_realized = holdout_payload["portfolio_returns"]
    holdout_sortino, holdout_max_dd, holdout_mean, holdout_vol, holdout_cumulative, holdout_drawdown = (
        _compute_metrics(holdout_returns_realized)
    )
    holdout_domain_exposure: dict[str, float] = {}
    avg_holdout_weights = holdout_payload["avg_weights"]
    for idx, domain in enumerate(domains):
        holdout_domain_exposure[domain] = holdout_domain_exposure.get(domain, 0.0) + float(
            avg_holdout_weights[idx]
        )
    selected_payload: SelectedExperimentPayload = SelectedExperimentPayload(
        params={
            **selected_params,
            "selection_source": "walkforward_tuning",
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

    best_metrics_path = out_dir / f"{artifact_prefix}_constrained_best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(
            {
                "strategy": "constrained_ogd_sgd",
                "best_params": selected_payload["params"],
                "domain_exposure": selected_payload["domain_exposure"],
                "feasibility_filter": {
                    "applied_threshold": cfg.max_domain_exposure_threshold,
                    "feasible_solution_found": feasible_payload is not None,
                },
                "data_split": {
                    "tuning_steps": int(tuning_returns.shape[0]),
                    "holdout_steps_total": int(holdout_returns.shape[0]),
                    "holdout_fraction": cfg.holdout_fraction,
                    "walkforward_train_steps": cfg.walkforward_train_steps,
                    "walkforward_test_steps": cfg.walkforward_test_steps,
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

    return {
        "constrained_grid": summary_path,
        "constrained_best_metrics": best_metrics_path,
        "constrained_best_timeseries": best_series_path,
    }

