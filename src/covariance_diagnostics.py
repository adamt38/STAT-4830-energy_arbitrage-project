"""Covariance and correlation diagnostics for concentration risk checks."""

from __future__ import annotations

import csv
import json
import pathlib
from itertools import combinations

import numpy as np

from src.baseline import _build_price_matrix, _compute_returns, _read_csv


def _write_matrix_csv(path: pathlib.Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *matrix[idx, :].tolist()])


def _build_category_returns(
    project_root: pathlib.Path,
    artifact_prefix: str,
) -> tuple[list[str], np.ndarray]:
    markets_rows = _read_csv(project_root / "data" / "processed" / f"{artifact_prefix}_markets_filtered.csv")
    history_rows = _read_csv(project_root / "data" / "processed" / f"{artifact_prefix}_price_history.csv")
    _, price_matrix, kept_tokens = _build_price_matrix(markets_rows, history_rows)
    returns_matrix = _compute_returns(price_matrix)

    token_to_category = {row["yes_token_id"]: row["domain"] for row in markets_rows}
    categories = sorted({token_to_category[token] for token in kept_tokens})
    category_to_indices: dict[str, list[int]] = {category: [] for category in categories}
    for idx, token in enumerate(kept_tokens):
        category_to_indices[token_to_category[token]].append(idx)

    category_returns = np.zeros((returns_matrix.shape[0], len(categories)), dtype=float)
    for j, category in enumerate(categories):
        indices = category_to_indices[category]
        category_returns[:, j] = np.nanmean(returns_matrix[:, indices], axis=1)
    category_returns = np.nan_to_num(category_returns, nan=0.0)
    return categories, category_returns


def run_covariance_diagnostics(
    project_root: pathlib.Path,
    artifact_prefix: str = "week8",
) -> dict[str, pathlib.Path]:
    """Compute covariance diagnostics and persist summary artifacts."""
    categories, category_returns = _build_category_returns(project_root, artifact_prefix=artifact_prefix)
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(categories) == 0 or category_returns.size == 0:
        raise RuntimeError("No category returns available for covariance diagnostics.")

    cov_matrix = np.atleast_2d(np.cov(category_returns, rowvar=False))
    if cov_matrix.shape != (len(categories), len(categories)):
        if len(categories) == 1:
            cov_matrix = np.array([[float(np.var(category_returns[:, 0]))]], dtype=float)
        else:
            raise RuntimeError("Unexpected covariance matrix shape for category returns.")
    std = np.sqrt(np.clip(np.diag(cov_matrix), 0.0, None))
    denom = np.outer(std, std)
    corr_matrix = np.divide(
        cov_matrix,
        denom,
        out=np.zeros_like(cov_matrix),
        where=denom > 1e-12,
    )
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    cov_path = out_dir / f"{artifact_prefix}_category_covariance.csv"
    corr_path = out_dir / f"{artifact_prefix}_category_correlation.csv"
    _write_matrix_csv(cov_path, categories, cov_matrix)
    _write_matrix_csv(corr_path, categories, corr_matrix)

    pair_rows: list[dict[str, float | str]] = []
    for i, j in combinations(range(len(categories)), 2):
        corr = float(corr_matrix[i, j])
        pair_rows.append(
            {
                "category_a": categories[i],
                "category_b": categories[j],
                "correlation": corr,
                "abs_correlation": abs(corr),
            }
        )
    pair_rows_sorted = sorted(pair_rows, key=lambda row: float(row["abs_correlation"]), reverse=True)

    top_pairs_path = out_dir / f"{artifact_prefix}_top_correlation_pairs.csv"
    with top_pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category_a", "category_b", "correlation", "abs_correlation"],
        )
        writer.writeheader()
        writer.writerows(pair_rows_sorted[:200])

    baseline_metrics = json.loads(
        (out_dir / f"{artifact_prefix}_baseline_metrics.json").read_text(encoding="utf-8")
    )
    constrained_metrics = json.loads(
        (out_dir / f"{artifact_prefix}_constrained_best_metrics.json").read_text(encoding="utf-8")
    )
    baseline_exp = baseline_metrics.get("exposure_by_domain", {})
    constrained_exp = constrained_metrics.get("domain_exposure", {})
    baseline_w = np.array([float(baseline_exp.get(cat, 0.0)) for cat in categories], dtype=float)
    constrained_w = np.array([float(constrained_exp.get(cat, 0.0)) for cat in categories], dtype=float)
    if baseline_w.sum() > 0:
        baseline_w = baseline_w / baseline_w.sum()
    if constrained_w.sum() > 0:
        constrained_w = constrained_w / constrained_w.sum()

    baseline_var = float(baseline_w @ cov_matrix @ baseline_w)
    constrained_var = float(constrained_w @ cov_matrix @ constrained_w)

    eigvals = np.linalg.eigvalsh(cov_matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals_sum = float(np.sum(eigvals))
    top_eig_share = float(eigvals[-1] / eigvals_sum) if eigvals_sum > 0 else 0.0

    pairwise_upper = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    summary = {
        "category_count": len(categories),
        "avg_abs_correlation": float(np.mean(pairwise_upper)) if pairwise_upper.size else 0.0,
        "max_abs_correlation": float(np.max(pairwise_upper)) if pairwise_upper.size else 0.0,
        "pairs_abs_corr_ge_0p8": int(
            sum(1 for row in pair_rows_sorted if float(row["abs_correlation"]) >= 0.8)
        ),
        "pairs_abs_corr_ge_0p6": int(
            sum(1 for row in pair_rows_sorted if float(row["abs_correlation"]) >= 0.6)
        ),
        "top_eigenvalue_share": top_eig_share,
        "baseline_portfolio_variance": baseline_var,
        "constrained_portfolio_variance": constrained_var,
        "variance_ratio_constrained_vs_baseline": (
            constrained_var / baseline_var if baseline_var > 0 else 0.0
        ),
    }
    summary_path = out_dir / f"{artifact_prefix}_covariance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "category_covariance": cov_path,
        "category_correlation": corr_path,
        "top_correlation_pairs": top_pairs_path,
        "covariance_summary": summary_path,
    }

