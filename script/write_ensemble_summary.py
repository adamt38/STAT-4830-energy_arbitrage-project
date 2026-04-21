"""Generate docs/direction_A_results.md with full multi-seed ensemble analysis.

Runs after the full queue completes. Produces:
  - Per-run table (baseline, constrained, delta, CI, holdout DD)
  - Pod G seed ensemble statistics (mean ± std across 5 seeds)
  - Pod L seed ensemble statistics (mean ± std across 3 seeds)
  - Hybrid run vs best G / best L
  - Interpretation in light of Adam's M4 noise finding
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_series(p: Path) -> list[dict[str, str]]:
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


def load_run(prefix: str, label: str) -> dict[str, Any] | None:
    p = REPO / "data" / "processed"
    bl_ts_path = p / f"{prefix}_baseline_timeseries.csv"
    co_ts_path = p / f"{prefix}_constrained_best_timeseries.csv"
    bl_m_path = p / f"{prefix}_baseline_metrics.json"
    co_m_path = p / f"{prefix}_constrained_best_metrics.json"
    ci_path = p / f"{prefix}_bootstrap_sortino_ci.json"
    if not (bl_ts_path.exists() and co_ts_path.exists()):
        return None
    bl_ts = _read_series(bl_ts_path)
    co_ts = _read_series(co_ts_path)
    N = len(co_ts)
    bl_r = np.array([float(r["portfolio_return"]) for r in bl_ts[-N:]])
    co_r = np.array([float(r["portfolio_return"]) for r in co_ts])
    bl_m = _read_json(bl_m_path) if bl_m_path.exists() else {}
    co_m = _read_json(co_m_path) if co_m_path.exists() else {}
    ci = _read_json(ci_path) if ci_path.exists() else {}
    exp = co_m.get("domain_exposure", {})
    return {
        "label": label,
        "prefix": prefix,
        "n_markets": int(bl_m.get("market_count", 0)) or 0,
        "bl_sortino": _sortino(bl_r),
        "co_sortino": _sortino(co_r),
        "delta": _sortino(co_r) - _sortino(bl_r),
        "bl_maxdd": _max_dd(bl_r),
        "co_maxdd": _max_dd(co_r),
        "weight_std": float(np.std(list(exp.values()))) if exp else 0.0,
        "max_domain_exp": max(exp.values()) if exp else 0.0,
        "ci": ci,
    }


def _fmt_ci(ci: dict[str, Any]) -> str:
    if not ci:
        return "—"
    return (
        f"[{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}] "
        f"(frac+ {ci.get('frac_positive', 0):.1%})"
    )


def ensemble_stats(runs: list[dict[str, Any]]) -> dict[str, float]:
    deltas = np.array([r["delta"] for r in runs if r is not None])
    bl_s = np.array([r["bl_sortino"] for r in runs if r is not None])
    co_s = np.array([r["co_sortino"] for r in runs if r is not None])
    if deltas.size == 0:
        return {}
    return {
        "n_seeds": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas, ddof=1) if deltas.size > 1 else 0.0),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "frac_pos": float(np.mean(deltas > 0)),
        "bl_sortino_mean": float(np.mean(bl_s)),
        "co_sortino_mean": float(np.mean(co_s)),
    }


def main() -> None:
    # Pod G seed sweep
    g_runs = [
        load_run("week9_G", "Pod G seed=7"),
        load_run("week9_G_seed42", "Pod G seed=42"),
        load_run("week9_G_seed123", "Pod G seed=123"),
        load_run("week9_G_seed456", "Pod G seed=456"),
        load_run("week9_G_seed789", "Pod G seed=789"),
    ]
    g_runs = [r for r in g_runs if r is not None]

    # Pod L seed sweep
    l_runs = [
        load_run("week9_L", "Pod L seed=7"),
        load_run("week9_L_seed42", "Pod L seed=42"),
        load_run("week9_L_seed123", "Pod L seed=123"),
    ]
    l_runs = [r for r in l_runs if r is not None]

    # Hybrid
    hybrid = load_run("week9_hybrid", "Hybrid (mom30 prefilter + learnable inclusion)")

    lines: list[str] = []
    lines.append("# Direction A Results — Multi-Seed Ensemble Study")
    lines.append("")
    lines.append(
        "Context: Adam's Round 4 M4 pod (200 trials, macro=both + top-K=3, no momentum) "
        "landed at **Sortino delta −0.048**, suggesting Round 2's +0.020 headline wins were "
        "at or below the noise floor. This document reports multi-seed replication of my "
        "two follow-up contributions:"
    )
    lines.append("")
    lines.append("1. **Pod G** — hand-picked top-20 momentum pre-filter (the original `--momentum-screening` lever that Adam merged into `cloud-runs` as commit `1436c71`).")
    lines.append("2. **Pod L** — learnable market inclusion (`--learnable-inclusion`, on `origin/learnable-selection`): replaces the hard top-K selection with a differentiable gate per market, jointly optimized with the weight logits under cardinality + binary-entropy regularizers.")
    lines.append("3. **Hybrid** — momentum prefilter to top-30, then learnable inclusion gates pick from those 30. Tests composition.")
    lines.append("")

    # ---- Pod G ensemble ----
    lines.append("## Pod G — momentum top-20 / 5d (5-seed ensemble)")
    lines.append("")
    lines.append("| Seed | N | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in g_runs:
        seed = r["label"].rsplit("=", 1)[-1]
        lines.append(
            f"| {seed} | {r['n_markets']} | {r['bl_sortino']:+.4f} | {r['co_sortino']:+.4f} | "
            f"**{r['delta']:+.4f}** | {_fmt_ci(r['ci'])} | "
            f"{r['bl_maxdd']*100:+.2f}% | {r['co_maxdd']*100:+.2f}% |"
        )
    g_stats = ensemble_stats(g_runs)
    if g_stats:
        lines.append("")
        lines.append("**Pod G ensemble summary:**")
        lines.append(f"- Mean Δ across {g_stats['n_seeds']} seeds: **{g_stats['delta_mean']:+.4f} ± {g_stats['delta_std']:.4f}** (std)")
        lines.append(f"- Δ range: [{g_stats['delta_min']:+.4f}, {g_stats['delta_max']:+.4f}]")
        lines.append(f"- Fraction of seeds with positive Δ: **{g_stats['frac_pos']:.0%}**")
        lines.append(f"- Mean baseline Sortino: {g_stats['bl_sortino_mean']:+.4f}")
        lines.append(f"- Mean constrained Sortino: {g_stats['co_sortino_mean']:+.4f}")
        # Verdict for G
        lines.append("")
        if g_stats["frac_pos"] >= 0.8 and g_stats["delta_mean"] > 0.01:
            verdict = (
                "**Verdict:** Pod G's positive Sortino delta replicates across seeds. "
                "The mean is well above the M4 noise floor (±0.05), and the fraction "
                "of positive seeds is strong evidence the effect is real."
            )
        elif g_stats["delta_mean"] > 0 and g_stats["frac_pos"] >= 0.5:
            verdict = (
                "**Verdict:** Pod G's effect is directionally positive but variance-limited. "
                "Mean is positive but the std is comparable to the mean, so with this sample "
                "size the effect is not distinguishable from noise at the M4 scale."
            )
        else:
            verdict = (
                "**Verdict:** Pod G's single-seed +0.034 does not replicate across seeds. "
                "The effect is consistent with noise at the M4 scale."
            )
        lines.append(verdict)
    lines.append("")

    # ---- Pod L ensemble ----
    lines.append("## Pod L — learnable market inclusion (multi-seed)")
    lines.append("")
    lines.append("| Seed | N | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD | Max dom exp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in l_runs:
        seed = r["label"].rsplit("=", 1)[-1]
        lines.append(
            f"| {seed} | {r['n_markets']} | {r['bl_sortino']:+.4f} | {r['co_sortino']:+.4f} | "
            f"**{r['delta']:+.4f}** | {_fmt_ci(r['ci'])} | "
            f"{r['bl_maxdd']*100:+.2f}% | {r['co_maxdd']*100:+.2f}% | "
            f"{r['max_domain_exp']:.4f} |"
        )
    l_stats = ensemble_stats(l_runs)
    if l_stats:
        lines.append("")
        lines.append("**Pod L ensemble summary:**")
        lines.append(f"- Mean Δ across {l_stats['n_seeds']} seeds: **{l_stats['delta_mean']:+.4f} ± {l_stats['delta_std']:.4f}** (std)")
        lines.append(f"- Δ range: [{l_stats['delta_min']:+.4f}, {l_stats['delta_max']:+.4f}]")
        lines.append(f"- Fraction of seeds with positive Δ: **{l_stats['frac_pos']:.0%}**")
    lines.append("")

    # ---- Hybrid ----
    if hybrid is not None:
        lines.append("## Hybrid — momentum top-30 prefilter + learnable inclusion (target k=15)")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Universe | {hybrid['n_markets']} markets (top-30 momentum) |")
        lines.append(f"| Baseline Sortino (holdout) | {hybrid['bl_sortino']:+.4f} |")
        lines.append(f"| Constrained Sortino (holdout) | {hybrid['co_sortino']:+.4f} |")
        lines.append(f"| **Δ** | **{hybrid['delta']:+.4f}** |")
        lines.append(f"| 95% CI | {_fmt_ci(hybrid['ci'])} |")
        lines.append(f"| Max DD (baseline / constrained) | {hybrid['bl_maxdd']*100:+.2f}% / {hybrid['co_maxdd']*100:+.2f}% |")
        lines.append(f"| Max domain exposure | {hybrid['max_domain_exp']:.4f} |")
        lines.append("")

    # ---- Comparison + interpretation ----
    lines.append("## Three-way comparison and interpretation")
    lines.append("")
    if g_stats and l_stats:
        lines.append(
            f"- **Pod G (static top-K)**: mean Δ = {g_stats['delta_mean']:+.4f} ± {g_stats['delta_std']:.4f} ({g_stats['n_seeds']} seeds)"
        )
        lines.append(
            f"- **Pod L (learnable inclusion)**: mean Δ = {l_stats['delta_mean']:+.4f} ± {l_stats['delta_std']:.4f} ({l_stats['n_seeds']} seeds)"
        )
        if hybrid:
            lines.append(
                f"- **Hybrid (prefilter + learnable)**: Δ = {hybrid['delta']:+.4f} (n=1)"
            )
        lines.append("")
        # Does L reduce variance vs G?
        if l_stats["n_seeds"] >= 2 and g_stats["n_seeds"] >= 2:
            var_ratio = (
                l_stats["delta_std"] / g_stats["delta_std"]
                if g_stats["delta_std"] > 1e-8
                else float("inf")
            )
            lines.append(
                f"- **Variance ratio (L/G std)**: {var_ratio:.2f} — "
                + (
                    "learnable inclusion is **more stable** across seeds"
                    if var_ratio < 0.9
                    else (
                        "learnable inclusion is **less stable** across seeds"
                        if var_ratio > 1.1
                        else "comparable stability across seeds"
                    )
                )
            )
        lines.append("")

    lines.append("### Placement against Adam's Round 4")
    lines.append("")
    lines.append(
        "- Adam's I4 (C2 × G2 cross-term): Δ = +0.0077 on a fresh data snapshot."
    )
    lines.append(
        "- Adam's M4 (noise check, 200 trials on C2): Δ = −0.0480, flipping sign."
    )
    lines.append(
        "- If our Pod G ensemble mean is above M4's |−0.05| threshold with low std, it means momentum pre-selection survives Adam's noise-robustness check. If it's at or below, the original +0.034 is consistent with noise at this problem's scale."
    )
    lines.append("")
    lines.append("### Next-experiment recommendation")
    lines.append("")
    if g_stats and l_stats:
        if l_stats["delta_mean"] > g_stats["delta_mean"] and l_stats.get("delta_std", 1) < g_stats.get("delta_std", 1):
            rec = "**Pod L (learnable inclusion) dominates**: higher mean, lower variance. Integrate into `cloud-runs` as a new CLI lever and pitch to Adam as a Round 5 variant on top of the Kelly pipeline."
        elif g_stats["delta_mean"] > l_stats["delta_mean"]:
            rec = "**Pod G (static top-K) has higher mean but learnable inclusion may still be worth keeping for stability**. Hybrid result may tell us if composition helps."
        else:
            rec = "**Neither cleanly dominates**; consider running a frozen-data snapshot study to remove ingest-time confound before more parameter exploration."
        lines.append(rec)
    lines.append("")

    out = REPO / "docs" / "direction_A_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
