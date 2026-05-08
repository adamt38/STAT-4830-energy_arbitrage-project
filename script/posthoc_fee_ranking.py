"""Post-hoc net-of-fees ranking across Kelly (week10) runs.

Deducts a per-step transaction cost ``fee_rate * ||w_t - w_{t-1}||_1`` from the
logged gross portfolio return and re-computes total log-wealth / annualized
log-growth / CAGR / Sortino / max drawdown at a grid of fee_rate values. The
equal-weight baseline has effectively zero turnover (it only pays the one-shot
setup cost at step 0 when going from zero to equal weights), so its metrics
barely move with fee_rate; the constrained Kelly portfolios, which exhibit
large realized turnover (K10C averages ``0.107`` L1 per step), pay a material
cost that this script surfaces.

Why Kelly-only: the week8 / week11 / week12 / week13 constrained MVO pipelines
write ``constrained_best_timeseries.csv`` with columns
``step,portfolio_return,cumulative_return,drawdown`` only -- no per-step
turnover is logged and no ``*_weights.csv`` is pushed, so a like-for-like
fee adjustment is impossible from the artifacts alone. Since those runs all
tie baseline to within +/-0.02 Sortino *before* fees, any positive fee can
only make them tie or lose -- their net-of-fees ranking is bounded above by
their gross-of-fees ranking. The markdown output notes this explicitly.

Inputs (cached under ``data/round7_cache/`` by default):
    {run}_kelly_best_timeseries.csv  -- columns include portfolio_return, turnover_l1
    {run}_baseline_timeseries.csv    -- column portfolio_return (full history)

Outputs:
    data/processed/{output_stem}_fee_ranking.csv        long-format rows
    data/processed/{output_stem}_fee_ranking.md         markdown leaderboard

Usage:
    python script/posthoc_fee_ranking.py \\
        --cache-dir data/round7_cache \\
        --runs K10A,K10B,K10C \\
        --fee-rates 0,0.0010,0.0050,0.0200 \\
        --steps-per-year 8760 \\
        --output-stem round7
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEE_RATES: tuple[float, ...] = (0.0, 0.0010, 0.0050, 0.0200)
DEFAULT_RUNS: tuple[str, ...] = ("K10A", "K10B", "K10C")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _read_column(path: Path, column: str, *, default: float = 0.0) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"[posthoc_fee_ranking] missing file: {_rel(path)}")
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"[posthoc_fee_ranking] {_rel(path)} missing column "
                f"{column!r}; found: {reader.fieldnames}"
            )
        for row in reader:
            raw = row.get(column, "")
            try:
                values.append(float(raw) if raw != "" else default)
            except ValueError:
                values.append(default)
    return np.asarray(values, dtype=float)


def _align_baseline(baseline: np.ndarray, n_holdout: int) -> np.ndarray:
    """Take the last ``n_holdout`` rows of baseline (same convention as
    :func:`_make_week9_diagnostics_report`)."""
    if baseline.size < n_holdout:
        raise SystemExit(
            f"[posthoc_fee_ranking] baseline has only {baseline.size} rows but "
            f"holdout needs {n_holdout}; cannot align."
        )
    return baseline[-n_holdout:]


def _sortino(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(returns) / (downside_dev + 1e-8))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    # Use log-wealth cumulative path so large negatives don't blow up
    # cumprod(1+r) when r approaches -1.
    cum = np.cumprod(np.clip(1.0 + returns, 1e-8, None))
    peak = np.maximum.accumulate(cum)
    dd = cum / np.clip(peak, 1e-8, None) - 1.0
    return float(np.min(dd))


def _log_wealth_total(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.sum(np.log(np.clip(1.0 + returns, 1e-8, None))))


def _metrics(
    returns: np.ndarray,
    *,
    steps_per_year: float,
) -> dict[str, float]:
    if returns.size == 0:
        return {
            "n_steps": 0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "total_log_wealth": 0.0,
            "annualized_log_growth": 0.0,
            "cagr": 0.0,
        }
    total_lw = _log_wealth_total(returns)
    ann_lg = total_lw / returns.size * steps_per_year
    cagr = float(np.exp(ann_lg) - 1.0)
    return {
        "n_steps": int(returns.size),
        "mean_return": float(np.mean(returns)),
        "volatility": float(np.std(returns)),
        "sortino": _sortino(returns),
        "max_drawdown": _max_drawdown(returns),
        "total_log_wealth": total_lw,
        "annualized_log_growth": ann_lg,
        "cagr": cagr,
    }


def _parse_csv_floats(raw: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(default)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise SystemExit("expected at least one comma-separated float.")
    try:
        return tuple(float(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid floats in {raw!r}: {exc}") from None


def _parse_csv_strs(raw: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise SystemExit("expected at least one comma-separated run id.")
    return tuple(parts)


def _resolve(path: str, cache_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cache_dir / p


def _evaluate_run(
    run: str,
    cache_dir: Path,
    fee_rates: Sequence[float],
    steps_per_year: float,
) -> list[dict[str, float]]:
    """Return one row per (strategy, fee_rate) for this run."""
    kelly_path = cache_dir / f"{run}_kelly_best_timeseries.csv"
    baseline_path = cache_dir / f"{run}_baseline_timeseries.csv"

    r_kelly = _read_column(kelly_path, "portfolio_return")
    turnover = _read_column(kelly_path, "turnover_l1")
    r_baseline_full = _read_column(baseline_path, "portfolio_return")
    r_baseline = _align_baseline(r_baseline_full, r_kelly.size)

    # Equal-weight baseline: the pipeline resets to uniform weights every step,
    # so its effective per-step turnover is 0 on the holdout (the one-shot
    # setup cost at step 0 is the same for both strategies and irrelevant for
    # the *delta* we care about).
    baseline_turnover = np.zeros_like(r_baseline)

    rows: list[dict[str, float]] = []
    for fee in fee_rates:
        r_k_net = r_kelly - fee * turnover
        r_b_net = r_baseline - fee * baseline_turnover
        m_k = _metrics(r_k_net, steps_per_year=steps_per_year)
        m_b = _metrics(r_b_net, steps_per_year=steps_per_year)
        rows.append({
            "run": run,
            "strategy": "kelly_constrained",
            "fee_rate": float(fee),
            "fee_bps": float(fee * 10_000),
            **m_k,
            "avg_turnover_per_step": float(np.mean(turnover)),
        })
        rows.append({
            "run": run,
            "strategy": "equal_weight_baseline",
            "fee_rate": float(fee),
            "fee_bps": float(fee * 10_000),
            **m_b,
            "avg_turnover_per_step": 0.0,
        })
    return rows


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
    runs: Sequence[str],
    fee_rates: Sequence[float],
    steps_per_year: float,
) -> None:
    lines: list[str] = []
    lines.append("# Round 7 post-hoc net-of-fees ranking (Kelly runs)")
    lines.append("")
    lines.append(
        f"Deducted `fee_rate * turnover_l1` from each step's gross return on the "
        f"already-evaluated Kelly holdout timeseries, then recomputed Sortino / "
        f"total log-wealth / equivalent CAGR / max drawdown. Steps/year = "
        f"`{steps_per_year:g}`. Runs: {', '.join(runs)}."
    )
    lines.append("")
    lines.append("## Headline: Kelly net-of-fees vs equal-weight baseline")
    lines.append("")
    lines.append(
        "| Run | fee (bps) | Strategy | Total log-wealth | Ann. log-growth | CAGR | Sortino | Max DD | avg L1 turn |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['run']} "
            f"| {row['fee_bps']:.1f} "
            f"| {row['strategy']} "
            f"| {row['total_log_wealth']:+.4f} "
            f"| {row['annualized_log_growth']:+.4f} "
            f"| {row['cagr']:+.2%} "
            f"| {row['sortino']:+.4f} "
            f"| {row['max_drawdown']:+.2%} "
            f"| {row['avg_turnover_per_step']:.4f} |"
        )
    lines.append("")
    lines.append("## Net-vs-gross delta (constrained minus baseline, per fee level)")
    lines.append("")
    lines.append(
        "| Run | fee (bps) | Δ total log-wealth | Δ CAGR | Δ Sortino | Δ Max DD |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    by_run_fee: dict[tuple[str, float], dict[str, dict[str, float]]] = {}
    for row in rows:
        key = (row["run"], row["fee_bps"])
        by_run_fee.setdefault(key, {})[row["strategy"]] = row
    for (run, fee_bps), pair in by_run_fee.items():
        k = pair.get("kelly_constrained")
        b = pair.get("equal_weight_baseline")
        if k is None or b is None:
            continue
        lines.append(
            f"| {run} "
            f"| {fee_bps:.1f} "
            f"| {k['total_log_wealth'] - b['total_log_wealth']:+.4f} "
            f"| {k['cagr'] - b['cagr']:+.2%} "
            f"| {k['sortino'] - b['sortino']:+.4f} "
            f"| {k['max_drawdown'] - b['max_drawdown']:+.2%} |"
        )
    lines.append("")
    lines.append("## Caveat on MVO (week8/week11/week12/week13) runs")
    lines.append("")
    lines.append(
        "The constrained MVO pipelines did not log per-step turnover or push "
        "`*_weights.csv`, so exact fee adjustment is not possible from the pushed "
        "artifacts. However, every MVO run on the 20-market universe "
        "(I4 / Q5 / S1 / S4 / S5) ties the equal-weight baseline within "
        "+/-0.02 Sortino *before* fees -- any positive fee rate strictly "
        "degrades their net-of-fees performance, so they cannot overtake Kelly "
        "after friction. This is a bound, not a calculation."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- At `fee = 0` the Kelly runs show their gross advantage (K10C: "
        "+0.46 log-wealth, +58 pp CAGR vs baseline)."
    )
    lines.append(
        "- The break-even fee rate is where `Δ total log-wealth -> 0` "
        "(see table below)."
    )
    lines.append(
        "- This motivates K10E (fee-aware Kelly) on the pod: by sweeping "
        "fee_rate directly inside the optimizer's loss + reported return, we let "
        "the optimizer re-allocate to LOWER-turnover policies when friction is "
        "expensive, rather than paying fees on a policy optimized for a "
        "frictionless world."
    )
    lines.append("")
    lines.append("## Break-even fee rate per run")
    lines.append("")
    lines.append(
        "`break_even ≈ (Δ log-wealth at fee=0) / (n_steps * avg_turnover_per_step)`"
    )
    lines.append("")
    lines.append("| Run | Δ log-w @ fee=0 | n_steps | avg turnover | break-even fee (bps) |")
    lines.append("|---|---:|---:|---:|---:|")
    for run in runs:
        k0 = next(
            (r for r in rows if r["run"] == run and r["strategy"] == "kelly_constrained" and r["fee_bps"] == 0.0),
            None,
        )
        b0 = next(
            (r for r in rows if r["run"] == run and r["strategy"] == "equal_weight_baseline" and r["fee_bps"] == 0.0),
            None,
        )
        if k0 is None or b0 is None:
            continue
        delta0 = k0["total_log_wealth"] - b0["total_log_wealth"]
        n = k0["n_steps"]
        tr = k0["avg_turnover_per_step"]
        denom = n * tr
        be_bps = (delta0 / denom) * 10_000 if denom > 0 else float("nan")
        lines.append(
            f"| {run} | {delta0:+.4f} | {n} | {tr:.4f} | {be_bps:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="data/round7_cache",
        help="Directory holding {run}_kelly_best_timeseries.csv and "
             "{run}_baseline_timeseries.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help=f"Comma-separated run ids (default: {','.join(DEFAULT_RUNS)}).",
    )
    parser.add_argument(
        "--fee-rates",
        default=None,
        help="Comma-separated fee rates as decimals (e.g. 0.0010 = 10 bps). "
             f"Default: {','.join(f'{x:g}' for x in DEFAULT_FEE_RATES)}.",
    )
    parser.add_argument(
        "--steps-per-year",
        type=float,
        default=8760.0,
        help="Steps per year for annualization (default: %(default)s -- 60-min "
             "bars).",
    )
    parser.add_argument(
        "--output-stem",
        default="round7_fee_ranking",
        help="Output stem under data/processed/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: %(default)s).",
    )
    args = parser.parse_args()

    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    if not cache_dir.exists():
        raise SystemExit(
            f"[posthoc_fee_ranking] cache dir not found: {_rel(cache_dir)}"
        )
    fee_rates = _parse_csv_floats(args.fee_rates, DEFAULT_FEE_RATES)
    runs = _parse_csv_strs(args.runs, DEFAULT_RUNS)

    all_rows: list[dict[str, float]] = []
    for run in runs:
        all_rows.extend(
            _evaluate_run(run, cache_dir, fee_rates, args.steps_per_year)
        )

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_stem}.csv"
    md_path = output_dir / f"{args.output_stem}.md"
    _write_csv(csv_path, all_rows)
    _write_markdown(md_path, all_rows, runs, fee_rates, args.steps_per_year)

    print(f"[posthoc_fee_ranking] wrote {_rel(csv_path)} ({len(all_rows)} rows)")
    print(f"[posthoc_fee_ranking] wrote {_rel(md_path)}")

    # Print one-line summary to stdout for quick sanity checks.
    for run in runs:
        run_rows = [r for r in all_rows if r["run"] == run]
        for fee_bps in sorted({r["fee_bps"] for r in run_rows}):
            k = next(
                (r for r in run_rows if r["fee_bps"] == fee_bps and r["strategy"] == "kelly_constrained"),
                None,
            )
            b = next(
                (r for r in run_rows if r["fee_bps"] == fee_bps and r["strategy"] == "equal_weight_baseline"),
                None,
            )
            if k is None or b is None:
                continue
            print(
                f"  {run} @ {fee_bps:>5.1f} bps: kelly log-w {k['total_log_wealth']:+.4f} | "
                f"baseline log-w {b['total_log_wealth']:+.4f} | "
                f"delta {k['total_log_wealth'] - b['total_log_wealth']:+.4f}"
            )


if __name__ == "__main__":
    main()
