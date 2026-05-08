"""Generate docs/option_A_results.md — 3-seed Option A (momentum × shrinkage × macro=both) analysis.

Directly tests a combination Adam's R2/R4/R5 matrices never exercised jointly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_series(p: Path) -> list[dict]:
    with p.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sortino(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    d = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(d * d)))
    return float(np.mean(r) / (dd + 1e-8))


def _max_dd(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak - 1.0))


def load_A(seed: int) -> dict | None:
    p = REPO / "data" / "processed"
    prefix = f"week13_A_seed{seed}"
    stem = f"{prefix}_macro_both"
    co_path = p / f"{stem}_constrained_best_timeseries.csv"
    bl_path = p / f"{prefix}_baseline_timeseries.csv"
    metrics_path = p / f"{stem}_constrained_best_metrics.json"
    ci_path = p / f"{prefix}_bootstrap_sortino_ci.json"
    if not (co_path.exists() and bl_path.exists()):
        return None
    bl_ts = _read_series(bl_path)
    co_ts = _read_series(co_path)
    N = len(co_ts)
    bl_r = np.array([float(r["portfolio_return"]) for r in bl_ts[-N:]])
    co_r = np.array([float(r["portfolio_return"]) for r in co_ts])
    co_m = _read_json(metrics_path) if metrics_path.exists() else {}
    ci = _read_json(ci_path) if ci_path.exists() else {}
    bp = co_m.get("best_params", {})
    exp = co_m.get("domain_exposure", {})
    return {
        "seed": seed,
        "bl_sortino": _sortino(bl_r),
        "co_sortino": _sortino(co_r),
        "delta": _sortino(co_r) - _sortino(bl_r),
        "bl_maxdd": _max_dd(bl_r),
        "co_maxdd": _max_dd(co_r),
        "max_domain_exp": max(exp.values()) if exp else 0.0,
        "best_params": bp,
        "ci": ci,
    }


def _fmt_ci(ci: dict) -> str:
    if not ci:
        return "—"
    return (
        f"[{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}] "
        f"(frac+ {ci.get('frac_positive', 0):.1%})"
    )


def main() -> None:
    A_runs = [load_A(s) for s in (7, 42, 123)]
    A_runs = [r for r in A_runs if r is not None]

    lines: list[str] = []
    lines.append("# Option A Results — Momentum × Baseline-Shrinkage × macro=both")
    lines.append("")
    lines.append(
        "A novel three-lever combination that Adam's Round-2/4/5 pods never tested "
        "jointly. Round 2's B2 tested baseline-shrinkage alone with `macro=rescale` "
        "(Δ = 0.000 — Optuna collapsed α to 1.0). Round 2's G/Round 4's I4 tested "
        "momentum alone with `macro=both` (Δ ∈ {+0.034, +0.008}). Nothing combines "
        "all three: shrinkage + momentum + macro=both."
    )
    lines.append("")
    lines.append("**Hypothesis**: shrinkage reduces selection overfit (α < 1 blends toward equal-weight at eval time, attenuating any one Optuna best-trial's idiosyncrasies). Momentum removes dead-weight markets (reduces parameter count). macro=both is the Round 4 winner. All three are orthogonal risk-reduction mechanisms; combining them should reduce the tuning→holdout gap that killed Round 5 G seeds.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Lever | Value |")
    lines.append("|---|---|")
    lines.append("| Macro mode | `both` (rescale path + additive `J_macro` term) |")
    lines.append("| Momentum screening | top-20 markets, 5-day lookback |")
    lines.append("| Baseline shrinkage | enabled (Optuna tunes α ∈ [0, 1]) |")
    lines.append("| Reduced search | enabled (denser Sobol on the meaningful subspace) |")
    lines.append("| Top-K bagging | 5 |")
    lines.append("| Optuna trials | 100 per seed |")
    lines.append("| Seeds | {7, 42, 123} |")
    lines.append("")
    lines.append("## Results by seed")
    lines.append("")
    lines.append("| Seed | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD | Max dom exp |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in A_runs:
        lines.append(
            f"| {r['seed']} | {r['bl_sortino']:+.4f} | {r['co_sortino']:+.4f} | "
            f"**{r['delta']:+.4f}** | {_fmt_ci(r['ci'])} | "
            f"{r['bl_maxdd']*100:+.2f}% | {r['co_maxdd']*100:+.2f}% | {r['max_domain_exp']:.4f} |"
        )

    if A_runs:
        deltas = np.array([r["delta"] for r in A_runs])
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1) if deltas.size > 1 else 0.0)
        frac_pos = float(np.mean(deltas > 0))
        lines.append("")
        lines.append(f"**Ensemble stats ({len(A_runs)} seeds):**")
        lines.append(f"- Mean Δ: **{mean_d:+.4f} ± {std_d:.4f}** (std)")
        lines.append(
            f"- Δ range: [{float(np.min(deltas)):+.4f}, {float(np.max(deltas)):+.4f}]"
        )
        lines.append(f"- Fraction of seeds with positive Δ: **{frac_pos:.0%}**")
        lines.append("")
        lines.append("**Benchmarks for comparison:**")
        lines.append("- Seed noise floor (Adam §15.9): **~0.036** Sortino (from G2 ↔ G-seed42 |ΔΔ|)")
        lines.append("- Adam's I4 single-seed Δ: **+0.0077**")
        lines.append("- Teammate's week17 Δ: **+0.1018** (on `origin/stock-PM-combined-strategy`)")
        lines.append("")
        if mean_d > 0.036 and frac_pos >= 0.66:
            lines.append(
                "**Verdict**: novel combo **robustly beats baseline**. "
                "Mean Δ clears the seed-noise floor and the majority of seeds are "
                "positive. Momentum + shrinkage + macro=both compound favorably."
            )
        elif mean_d > 0.01 and frac_pos >= 0.66:
            lines.append(
                "**Verdict**: novel combo is **directionally positive** across seeds "
                "but the mean is within the noise floor. Suggestive but not decisive; "
                "would need 5+ seeds to pin down."
            )
        elif frac_pos >= 0.5:
            lines.append(
                "**Verdict**: novel combo is **neutral / noise-dominated**. "
                "Momentum × shrinkage × macro=both does not compound robustly."
            )
        else:
            lines.append(
                "**Verdict**: novel combo **does not beat baseline**. "
                "Most seeds show negative delta; the added shrinkage lever likely "
                "over-regularizes the momentum-screened universe."
            )

    lines.append("")
    lines.append("## Selected hyperparameters per seed")
    lines.append("")
    for r in A_runs:
        bp = r.get("best_params", {})
        lines.append(f"**seed={r['seed']}**:")
        for k in (
            "learning_rate", "rolling_window", "domain_limit", "max_weight",
            "uniform_mix", "concentration_penalty_lambda", "covariance_penalty_lambda",
            "variance_penalty", "downside_penalty", "baseline_alpha",
        ):
            v = bp.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"  - `{k}`: {v:.4f}")
            else:
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    out = REPO / "docs" / "option_A_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
