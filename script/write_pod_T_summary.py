"""Generate docs/pod_T_results.md — 3-seed Pod T analysis vs teammate baseline.

Reads Pod T seeds 7/42/123 (week13_T_*) artifacts plus Pod G seed=123 from the
earlier run, and writes a comparison doc targeting Adam's §15.9 noise floor
analysis and the teammate's week17 +0.1018 benchmark.
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


def load_T(seed: int) -> dict | None:
    p = REPO / "data" / "processed"
    prefix = f"week13_T_seed{seed}"
    stem = f"{prefix}_macro_both"
    co_path = p / f"{stem}_constrained_best_timeseries.csv"
    bl_path = p / f"{prefix}_baseline_timeseries.csv"
    co_metrics_path = p / f"{stem}_constrained_best_metrics.json"
    ci_path = p / f"{prefix}_bootstrap_sortino_ci.json"
    if not (co_path.exists() and bl_path.exists()):
        return None
    bl_ts = _read_series(bl_path)
    co_ts = _read_series(co_path)
    N = len(co_ts)
    bl_r = np.array([float(r["portfolio_return"]) for r in bl_ts[-N:]])
    co_r = np.array([float(r["portfolio_return"]) for r in co_ts])
    co_m = _read_json(co_metrics_path) if co_metrics_path.exists() else {}
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
        "weight_std": float(np.std(list(exp.values()))) if exp else 0.0,
        "best_params": bp,
        "ci": ci,
    }


def _fmt_ci(ci: dict) -> str:
    if not ci:
        return "—"
    return f"[{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}] (frac+ {ci.get('frac_positive', 0):.1%})"


def main() -> None:
    T_runs = [load_T(s) for s in (7, 42, 123)]
    T_runs = [r for r in T_runs if r is not None]

    lines: list[str] = []
    lines.append("# Pod T Results — Teammate Config + Momentum + 3-Seed Robustness")
    lines.append("")
    lines.append(
        "**Goal**: verify whether the teammate's week17 winning configuration (Sortino "
        "delta +0.1018 per `origin/stock-PM-combined-strategy`) replicates across seeds "
        "and stacks with our merged momentum screening lever. Addresses the ~0.036 "
        "seed-noise floor Adam documented in `docs/cloud_runbook.md` §15.9 (Round 5 "
        "closeout)."
    )
    lines.append("")
    lines.append("## Pod T configuration")
    lines.append("")
    lines.append("Centered tightly on teammate's week17 winners, with our merged momentum lever added.")
    lines.append("")
    lines.append("| Hyperparameter | Teammate (week17) | Pod T search |")
    lines.append("|---|---|---|")
    lines.append("| `lr` | 0.005 | 0.005, 0.01 |")
    lines.append("| `max_weight` | 0.04 | 0.04 (pinned) |")
    lines.append("| `domain_limit` | 0.08 | 0.08 (pinned) |")
    lines.append("| `rolling_window` | 288 | 96, 288 |")
    lines.append("| `variance_penalty` | 0.5 | 0.25, 0.5, 1.0 |")
    lines.append("| `downside_penalty` | 1.0 | 0.5, 1.0, 2.0 |")
    lines.append("| `covariance_penalty_lambda` | 0.5 | 0.25, 0.5, 1.0 |")
    lines.append("| `concentration_penalty_lambda` | 2.0 | 1.0, 2.0, 3.0 |")
    lines.append("| Momentum screening | — | top-20, 5d lookback (added) |")
    lines.append("| Macro mode | rescale (implicit) | both (Adam's R4 winner) |")
    lines.append("| top-K bagging | n/a | 5 |")
    lines.append("| Optuna trials | n/a | 100 per seed |")
    lines.append("| Seeds | one | {7, 42, 123} |")
    lines.append("")

    lines.append("## Results by seed")
    lines.append("")
    lines.append("| Seed | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD | Max dom exp |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in T_runs:
        lines.append(
            f"| {r['seed']} | {r['bl_sortino']:+.4f} | {r['co_sortino']:+.4f} | "
            f"**{r['delta']:+.4f}** | {_fmt_ci(r['ci'])} | "
            f"{r['bl_maxdd']*100:+.2f}% | {r['co_maxdd']*100:+.2f}% | {r['max_domain_exp']:.4f} |"
        )
    if T_runs:
        deltas = np.array([r["delta"] for r in T_runs])
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1) if deltas.size > 1 else 0.0)
        frac_pos = float(np.mean(deltas > 0))
        lines.append("")
        lines.append(f"**Ensemble stats ({len(T_runs)} seeds):**")
        lines.append(f"- Mean Δ: **{mean_d:+.4f} ± {std_d:.4f}** (std)")
        lines.append(f"- Δ range: [{float(np.min(deltas)):+.4f}, {float(np.max(deltas)):+.4f}]")
        lines.append(f"- Fraction of seeds with positive Δ: **{frac_pos:.0%}**")
        lines.append("")
        lines.append("**Comparison to benchmarks:**")
        lines.append(f"- Teammate's week17 Δ: +0.1018 (single seed, `origin/stock-PM-combined-strategy`)")
        lines.append("- Round 5 closeout noise floor (σ across seeds on G recipe): ~0.036")
        lines.append(f"- Adam's I4 Δ: +0.0077 (single seed, `origin/cloud-runs-I4`)")
        lines.append("")
        if mean_d > 0.036 and frac_pos >= 0.66:
            lines.append(
                "**Verdict**: Pod T's mean Δ exceeds the seed-noise floor AND the majority "
                "of seeds show positive delta. This is a **statistically defensible beat** of baseline. "
                "Adding the momentum lever to teammate's tight-sizing recipe stacks favorably."
            )
        elif mean_d > 0 and frac_pos >= 0.5:
            lines.append(
                "**Verdict**: Pod T is directionally positive across seeds but the mean is "
                "within the noise floor. Cannot yet claim robust beat. More seeds needed, or "
                "the teammate's exact recipe should be replicated without the momentum lever "
                "to isolate sources of uncertainty."
            )
        elif mean_d <= 0:
            lines.append(
                "**Verdict**: Pod T does NOT replicate a positive delta across seeds. "
                "Teammate's recipe × our momentum lever does not compound cleanly. "
                "The teammate's +0.1018 may be seed-specific to their exact run."
            )
    lines.append("")
    lines.append("## Selected hyperparameters (per seed)")
    lines.append("")
    for r in T_runs:
        bp = r.get("best_params", {})
        lines.append(f"**seed={r['seed']}**:")
        for k in (
            "learning_rate", "rolling_window", "domain_limit", "max_weight", "uniform_mix",
            "variance_penalty", "downside_penalty", "covariance_penalty_lambda",
            "covariance_shrinkage", "concentration_penalty_lambda",
        ):
            v = bp.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"  - `{k}`: {v:.4f}")
            else:
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    out = REPO / "docs" / "pod_T_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
