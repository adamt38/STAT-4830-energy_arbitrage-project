"""Circular-block bootstrap confidence interval on Kelly vs baseline log-wealth delta.

For time-series returns, IID bootstrap destroys autocorrelation structure and
produces artificially narrow CIs. The circular-block bootstrap (Politis &
Romano 1992) resamples contiguous blocks of length ``L`` from the concatenated
series (wrapping around at the boundary), preserving short-range dependence.
This script:

1. Loads the aligned Kelly + baseline holdout returns.
2. For each of ``B`` replicates, draws ``ceil(N/L)`` blocks of length ``L``
   from random circular offsets, truncates back to ``N``, and computes the
   delta total log-wealth between the Kelly and baseline paths (using the
   SAME block offsets so the two series stay aligned step-by-step).
3. Reports mean / std / 95% CI / Pr(delta > 0) over the ``B`` replicates.

Why circular-block and not stationary-block: with ~11k hourly steps, a fixed
block length of L=50 (~2 days) matches the visible autocorrelation horizon in
Polymarket returns and keeps implementation simple.

Usage:
    python script/posthoc_bootstrap_ci.py \\
        --cache-dir data/round7_cache \\
        --runs K10A,K10B,K10C \\
        --n-boot 1000 \\
        --block-length 50 \\
        --seed 7 \\
        --output-stem round7_bootstrap_ci
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS: tuple[str, ...] = ("K10A", "K10B", "K10C")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _read_column(path: Path, column: str) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"[posthoc_bootstrap_ci] missing file: {_rel(path)}")
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"[posthoc_bootstrap_ci] {_rel(path)} missing column "
                f"{column!r}; found: {reader.fieldnames}"
            )
        for row in reader:
            raw = row.get(column, "")
            try:
                values.append(float(raw) if raw != "" else 0.0)
            except ValueError:
                values.append(0.0)
    return np.asarray(values, dtype=float)


def _circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of length ``n`` of indices sampled via circular-block
    bootstrap with block length ``block_length``."""
    if block_length <= 0:
        raise SystemExit(f"block_length must be positive, got {block_length}")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    # For each block, generate contiguous indices [start, start+1, ..., start+L-1] mod n.
    offsets = np.arange(block_length)
    block_indices = (starts[:, None] + offsets[None, :]) % n
    flat = block_indices.reshape(-1)
    return flat[:n]


def _log_wealth(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.sum(np.log(np.clip(1.0 + returns, 1e-8, None))))


def _bootstrap_delta(
    kelly: np.ndarray,
    baseline: np.ndarray,
    *,
    n_boot: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    if kelly.size != baseline.size:
        raise SystemExit(
            f"[posthoc_bootstrap_ci] kelly size {kelly.size} != baseline "
            f"size {baseline.size}; cannot run paired bootstrap."
        )
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _circular_block_indices(kelly.size, block_length, rng)
        deltas[b] = _log_wealth(kelly[idx]) - _log_wealth(baseline[idx])
    observed = _log_wealth(kelly) - _log_wealth(baseline)
    return {
        "observed_delta": observed,
        "bootstrap_mean": float(np.mean(deltas)),
        "bootstrap_std": float(np.std(deltas, ddof=1)),
        "ci_lo_95": float(np.percentile(deltas, 2.5)),
        "ci_hi_95": float(np.percentile(deltas, 97.5)),
        "ci_lo_99": float(np.percentile(deltas, 0.5)),
        "ci_hi_99": float(np.percentile(deltas, 99.5)),
        "pr_delta_gt_0": float(np.mean(deltas > 0)),
        "z_score_vs_zero": observed / (float(np.std(deltas, ddof=1)) + 1e-12),
    }


def _evaluate_run(
    run: str,
    cache_dir: Path,
    *,
    n_boot: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    kelly_path = cache_dir / f"{run}_kelly_best_timeseries.csv"
    baseline_path = cache_dir / f"{run}_baseline_timeseries.csv"
    r_kelly = _read_column(kelly_path, "portfolio_return")
    r_baseline_full = _read_column(baseline_path, "portfolio_return")
    if r_baseline_full.size < r_kelly.size:
        raise SystemExit(
            f"[posthoc_bootstrap_ci] {run}: baseline size {r_baseline_full.size} < "
            f"kelly size {r_kelly.size}"
        )
    r_baseline = r_baseline_full[-r_kelly.size:]
    result = _bootstrap_delta(
        r_kelly,
        r_baseline,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
    )
    result["run"] = run
    result["n_steps"] = int(r_kelly.size)
    return result


def _write_markdown(
    path: Path,
    rows: Sequence[dict[str, float]],
    *,
    n_boot: int,
    block_length: int,
    seed: int,
) -> None:
    lines: list[str] = []
    lines.append("# Round 7 circular-block bootstrap CI on Kelly log-wealth delta")
    lines.append("")
    lines.append(
        f"- **replicates:** {n_boot}; **block length:** {block_length} steps "
        f"(~{block_length / 24:.1f} days on 60-min bars); **seed:** {seed}."
    )
    lines.append(
        "- Delta = `total_log_wealth(kelly) - total_log_wealth(baseline)` on paired "
        "circular-block resamples (same block offsets applied to both series so "
        "step-by-step alignment is preserved)."
    )
    lines.append("")
    lines.append(
        "| Run | n_steps | Observed Δ | Bootstrap mean | Std | 95% CI | 99% CI | Pr(Δ>0) | z |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['run']} "
            f"| {row['n_steps']} "
            f"| {row['observed_delta']:+.4f} "
            f"| {row['bootstrap_mean']:+.4f} "
            f"| {row['bootstrap_std']:.4f} "
            f"| [{row['ci_lo_95']:+.4f}, {row['ci_hi_95']:+.4f}] "
            f"| [{row['ci_lo_99']:+.4f}, {row['ci_hi_99']:+.4f}] "
            f"| {row['pr_delta_gt_0']:.3f} "
            f"| {row['z_score_vs_zero']:+.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **Pr(Δ>0)** = fraction of bootstrap replicates where Kelly beat "
        "baseline on resampled blocks. A value near 1.0 with a 95% CI that "
        "excludes zero is strong evidence of a real (as opposed to noise-driven) "
        "gross edge. A CI that straddles zero means the apparent +0.46 log-"
        "wealth is within the range of luck given the autocorrelation structure."
    )
    lines.append(
        "- **z-score** = observed_delta / bootstrap_std. Rule of thumb: |z|>2 is "
        "marginal evidence, |z|>3 is strong, |z|>5 is overwhelming."
    )
    lines.append(
        "- **Caveat:** these CIs are gross-of-fees. The fee-ranking post-hoc shows "
        "break-even fees of 3.8 / 10.9 / 5.6 bps for K10C / K10B / K10A; the CI "
        "here only tells you whether the **gross** edge is statistically real."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Put 'run' first, then numeric columns
    ordered_keys = ["run", "n_steps"] + [
        k for k in rows[0].keys() if k not in ("run", "n_steps")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_csv_strs(raw: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/round7_cache")
    parser.add_argument("--runs", default=None,
                        help=f"Comma-separated run ids (default: {','.join(DEFAULT_RUNS)}).")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-stem", default="round7_bootstrap_ci")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    if not cache_dir.exists():
        raise SystemExit(f"[posthoc_bootstrap_ci] cache dir not found: {_rel(cache_dir)}")
    runs = _parse_csv_strs(args.runs, DEFAULT_RUNS)

    rows: list[dict[str, float]] = []
    for run in runs:
        rows.append(
            _evaluate_run(
                run,
                cache_dir,
                n_boot=args.n_boot,
                block_length=args.block_length,
                seed=args.seed,
            )
        )

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_stem}.csv"
    md_path = output_dir / f"{args.output_stem}.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, n_boot=args.n_boot,
                    block_length=args.block_length, seed=args.seed)

    print(f"[posthoc_bootstrap_ci] wrote {_rel(csv_path)} ({len(rows)} rows)")
    print(f"[posthoc_bootstrap_ci] wrote {_rel(md_path)}")
    for row in rows:
        print(
            f"  {row['run']}: observed Δ={row['observed_delta']:+.4f}, "
            f"95% CI [{row['ci_lo_95']:+.4f}, {row['ci_hi_95']:+.4f}], "
            f"Pr(Δ>0)={row['pr_delta_gt_0']:.3f}, z={row['z_score_vs_zero']:+.2f}"
        )


if __name__ == "__main__":
    main()
