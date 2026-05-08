"""Post-hoc alpha-blend evaluator for constrained week8 / week9 / week11 runs.

Takes a single pod's already-optimized constrained weights and its equal-weight
baseline, then evaluates the per-step blended return

    r_blend(t; alpha) = alpha * r_constrained(t) + (1 - alpha) * r_baseline(t)

on the held-out walk-forward steps for a grid of alpha values. Writes:

    data/processed/{output_stem}_alpha_blend.csv
        one row per alpha with Sortino / total-return / max-drawdown /
        volatility / cumulative-log-wealth columns.
    data/processed/{output_stem}_alpha_blend_summary.md
        markdown summary table plus a short narrative block.
    data/processed/{output_stem}_alpha_blend_sortino.png
        Sortino-vs-alpha frontier plot.

Why post-hoc (rather than as an Optuna dimension):

    --baseline-shrinkage already exists as a search-space lever, but every Round 2
    pod that opened that dimension collapsed to alpha=0 because the training
    objective (Sortino or Sortino-vs-baseline) never rewards dilution -- the
    volatility reduction from blending never offsets the mean-return hit inside
    the walk-forward fold. A post-hoc sweep bypasses that problem: we take the
    weights the pipeline DID converge on and re-score them at alpha > 0 to see
    whether a blended portfolio would have had a better out-of-sample Sortino
    (or drawdown profile) than the pure constrained portfolio. This is a risk-
    adjusted sensitivity test, not a fit.

Usage:

    python script/posthoc_alpha_blend.py \
        --artifact-prefix week11_I \
        --constrained-stem week11_I_macro_both \
        --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --output-stem week11_I_alpha_blend

If --output-stem is omitted it defaults to "{constrained-stem}_alpha_blend".

Alignment: the baseline_timeseries.csv covers the full history; the
constrained_best_timeseries.csv covers only the holdout walk-forward steps. We
align by taking the last N rows of the baseline where N == len(constrained),
matching the convention used in _make_week9_diagnostics_report().
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _read_returns(path: Path, column: str = "portfolio_return") -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"[posthoc_alpha_blend] missing file: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"[posthoc_alpha_blend] {path} missing column {column!r}; "
                f"found columns: {reader.fieldnames}"
            )
        for row in reader:
            raw = row.get(column, "")
            try:
                values.append(float(raw) if raw != "" else 0.0)
            except ValueError:
                values.append(0.0)
    return np.asarray(values, dtype=float)


def _sortino(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(returns) / (downside_dev + 1e-8))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = cum / np.clip(peak, 1e-8, None) - 1.0
    return float(np.min(dd))


def _total_return(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def _cumulative_log_wealth(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    clipped = np.clip(1.0 + returns, 1e-8, None)
    return float(np.sum(np.log(clipped)))


def _metrics_for_alpha(
    r_constrained: np.ndarray,
    r_baseline: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    blended = alpha * r_constrained + (1.0 - alpha) * r_baseline
    return {
        "alpha": float(alpha),
        "n_steps": int(blended.size),
        "mean_return": float(np.mean(blended)) if blended.size else 0.0,
        "volatility": float(np.std(blended)) if blended.size else 0.0,
        "sortino": _sortino(blended),
        "max_drawdown": _max_drawdown(blended),
        "total_return": _total_return(blended),
        "cumulative_log_wealth": _cumulative_log_wealth(blended),
    }


def _parse_alphas(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_ALPHAS
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise SystemExit("--alphas requires at least one value.")
    try:
        values = tuple(float(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(
            f"--alphas must be comma-separated floats, got {raw!r}: {exc}"
        ) from None
    for v in values:
        if v < 0.0 or v > 1.0:
            raise SystemExit(
                f"--alphas entries must lie in [0.0, 1.0], got {values}."
            )
    return values


def _write_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    rows: Sequence[dict[str, float]],
    artifact_prefix: str,
    constrained_stem: str,
    constrained_suffix: str,
    baseline_rows: int,
    constrained_rows: int,
    aligned_rows: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Post-hoc alpha-blend evaluation: `{constrained_stem}`")
    lines.append("")
    lines.append(
        f"- **baseline series** `{artifact_prefix}_baseline_timeseries.csv` "
        f"(full rows: {baseline_rows})"
    )
    lines.append(
        f"- **optimized series** "
        f"`{constrained_stem}_{constrained_suffix}_timeseries.csv` "
        f"(holdout rows: {constrained_rows})"
    )
    lines.append(f"- **aligned holdout rows** used: {aligned_rows}")
    lines.append("")
    lines.append(
        "Blend definition: `r_blend(t; alpha) = alpha * r_constrained(t) + "
        "(1 - alpha) * r_baseline(t)`. `alpha = 1.0` is the pure constrained "
        "portfolio (what the pipeline reports); `alpha = 0.0` is pure equal-"
        "weight baseline."
    )
    lines.append("")
    lines.append(
        "| alpha | Sortino | mean_ret | volatility | max_dd | total_ret | "
        "cum_log_wealth |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['alpha']:.2f} "
            f"| {row['sortino']:+.4f} "
            f"| {row['mean_return']:+.6f} "
            f"| {row['volatility']:.6f} "
            f"| {row['max_drawdown']:+.4%} "
            f"| {row['total_return']:+.4%} "
            f"| {row['cumulative_log_wealth']:+.6f} |"
        )
    lines.append("")
    best_sortino = max(rows, key=lambda r: r["sortino"])
    best_total = max(rows, key=lambda r: r["total_return"])
    best_dd = max(rows, key=lambda r: r["max_drawdown"])  # least negative
    lines.append("## Argmax summary")
    lines.append("")
    lines.append(
        f"- **best Sortino** at alpha = {best_sortino['alpha']:.2f} "
        f"(Sortino = {best_sortino['sortino']:+.4f})"
    )
    lines.append(
        f"- **best total return** at alpha = {best_total['alpha']:.2f} "
        f"(total = {best_total['total_return']:+.4%})"
    )
    lines.append(
        f"- **least-negative max drawdown** at alpha = {best_dd['alpha']:.2f} "
        f"(max_dd = {best_dd['max_drawdown']:+.4%})"
    )
    lines.append("")
    lines.append(
        "If the Sortino argmax is strictly interior (0 < alpha < 1) and beats the "
        "alpha=1 row, the optimized constrained portfolio is *over*-exposed on a "
        "risk-adjusted basis — a blended allocation would have dominated on this "
        "holdout. If the argmax is at alpha = 1.0, the constrained portfolio is "
        "already efficient relative to the baseline and there is no free lunch in "
        "post-hoc dilution."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(
    path: Path,
    rows: Sequence[dict[str, float]],
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[posthoc_alpha_blend] matplotlib not available, skipping plot.",
            flush=True,
        )
        return

    alphas = np.array([r["alpha"] for r in rows])
    sortinos = np.array([r["sortino"] for r in rows])
    totals = np.array([r["total_return"] for r in rows])
    drawdowns = np.array([r["max_drawdown"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(alphas, sortinos, marker="o")
    axes[0].axvline(1.0, linestyle=":", color="grey", alpha=0.6)
    axes[0].set_xlabel("alpha (constrained weight)")
    axes[0].set_ylabel("Sortino (holdout)")
    axes[0].set_title("Sortino frontier")
    axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, totals * 100.0, marker="o", color="tab:green")
    axes[1].axvline(1.0, linestyle=":", color="grey", alpha=0.6)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("total return (%)")
    axes[1].set_title("Total-return frontier")
    axes[1].grid(alpha=0.3)

    axes[2].plot(alphas, drawdowns * 100.0, marker="o", color="tab:red")
    axes[2].axvline(1.0, linestyle=":", color="grey", alpha=0.6)
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("max drawdown (%)")
    axes[2].set_title("Drawdown frontier")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc alpha-blend evaluator for constrained vs equal-weight "
            "baseline holdout returns. Does NOT re-run Optuna -- only re-scores "
            "already-written timeseries."
        )
    )
    parser.add_argument(
        "--artifact-prefix",
        required=True,
        help="Prefix used when the pipeline wrote the baseline timeseries "
        "(e.g. 'week11_I'). The script reads "
        "data/processed/{artifact_prefix}_baseline_timeseries.csv.",
    )
    parser.add_argument(
        "--constrained-stem",
        required=True,
        help="Stem used for the optimized portfolio's best-weights artifacts "
        "(e.g. 'week11_I_macro_both' for constrained, 'week10_kelly_B' for Kelly). "
        "The script reads "
        "data/processed/{constrained_stem}_{constrained_suffix}_timeseries.csv.",
    )
    parser.add_argument(
        "--constrained-suffix",
        default="constrained_best",
        help="Suffix between {constrained_stem} and _timeseries.csv. Defaults to "
        "'constrained_best' for week8/week9/week11 pods. Use 'kelly_best' for "
        "week10 Kelly pods. Output file names use {output_stem} unchanged.",
    )
    parser.add_argument(
        "--alphas",
        default=None,
        metavar="A1,A2,...",
        help="Comma-separated alpha grid in [0.0, 1.0]. "
        f"Defaults to {','.join(f'{a:.1f}' for a in DEFAULT_ALPHAS)}.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Stem for written artifacts (csv / md / png). "
        "Defaults to '{constrained-stem}_alpha_blend'.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override the data/processed/ directory. Defaults to "
        f"{REPO_ROOT / 'data' / 'processed'}.",
    )
    args = parser.parse_args()

    processed = (
        Path(args.processed_dir)
        if args.processed_dir is not None
        else REPO_ROOT / "data" / "processed"
    )
    processed.mkdir(parents=True, exist_ok=True)

    baseline_path = processed / f"{args.artifact_prefix}_baseline_timeseries.csv"
    constrained_path = (
        processed
        / f"{args.constrained_stem}_{args.constrained_suffix}_timeseries.csv"
    )

    baseline_rets_full = _read_returns(baseline_path)
    constrained_rets = _read_returns(constrained_path)

    if constrained_rets.size == 0:
        raise SystemExit(
            f"[posthoc_alpha_blend] {constrained_path} has zero rows; "
            f"run the pipeline first."
        )
    if baseline_rets_full.size < constrained_rets.size:
        raise SystemExit(
            f"[posthoc_alpha_blend] baseline rows ({baseline_rets_full.size}) "
            f"< constrained rows ({constrained_rets.size}); alignment impossible."
        )

    baseline_holdout = baseline_rets_full[-constrained_rets.size :]
    aligned_rows = int(constrained_rets.size)

    alphas = _parse_alphas(args.alphas)
    rows = [_metrics_for_alpha(constrained_rets, baseline_holdout, a) for a in alphas]

    output_stem = args.output_stem or f"{args.constrained_stem}_alpha_blend"
    csv_path = processed / f"{output_stem}.csv"
    md_path = processed / f"{output_stem}_summary.md"
    png_path = processed / f"{output_stem}_sortino.png"

    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        rows,
        artifact_prefix=args.artifact_prefix,
        constrained_stem=args.constrained_stem,
        constrained_suffix=args.constrained_suffix,
        baseline_rows=int(baseline_rets_full.size),
        constrained_rows=int(constrained_rets.size),
        aligned_rows=aligned_rows,
    )
    _write_plot(
        png_path,
        rows,
        title=f"Post-hoc alpha-blend: {args.constrained_stem}",
    )

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(
        json.dumps(
            {
                "artifact_prefix": args.artifact_prefix,
                "constrained_stem": args.constrained_stem,
                "constrained_suffix": args.constrained_suffix,
                "aligned_rows": aligned_rows,
                "n_alphas": len(alphas),
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "plot": _rel(png_path),
                "best_sortino_alpha": max(rows, key=lambda r: r["sortino"])["alpha"],
                "best_sortino_value": max(rows, key=lambda r: r["sortino"])["sortino"],
                "alpha_1_sortino": next(r["sortino"] for r in rows if r["alpha"] == 1.0)
                if any(r["alpha"] == 1.0 for r in rows)
                else None,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
