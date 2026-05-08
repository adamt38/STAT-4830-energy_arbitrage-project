"""Generate docs/direction_A_results.md summarizing Pod G / seed42 / Pod L.

Reads three sets of artifacts from data/processed/ and produces a single
markdown file ready to commit.
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


def _load_pod(prefix: str) -> dict[str, Any] | None:
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

    bp = co_m.get("best_params", {})
    exp = co_m.get("domain_exposure", {})
    return {
        "prefix": prefix,
        "n_markets": int(bl_m.get("market_count", 0)),
        "bl_sortino": _sortino(bl_r),
        "co_sortino": _sortino(co_r),
        "delta": _sortino(co_r) - _sortino(bl_r),
        "bl_maxdd": _max_dd(bl_r),
        "co_maxdd": _max_dd(co_r),
        "holdout_mean": float(bp.get("holdout_mean_return", 0.0)),
        "holdout_vol": float(bp.get("holdout_volatility", 0.0)),
        "weight_std": float(np.std(list(exp.values()))) if exp else 0.0,
        "max_domain_exp": max(exp.values()) if exp else 0.0,
        "ci": ci,
        "best_params": bp,
    }


def _fmt_ci(ci: dict[str, Any]) -> str:
    if not ci:
        return "—"
    return (
        f"[{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}] "
        f"(frac+ {ci.get('frac_positive', 0):.1%})"
    )


def main() -> None:
    pods = {
        "G (seed=7)": _load_pod("week9_G"),
        "G (seed=42)": _load_pod("week9_G_seed42"),
        "L (learnable-inclusion)": _load_pod("week9_L"),
    }

    lines: list[str] = []
    lines.append("# Direction A Results — Learnable Market Inclusion")
    lines.append("")
    lines.append(
        "Summary of three runs: Pod G (momentum screening, seed=7), Pod G replication "
        "(same config, seed=42), and Pod L (learnable inclusion on full universe)."
    )
    lines.append("")
    lines.append("## Head-to-head Sortino (holdout)")
    lines.append("")
    lines.append("| Run | N markets | BL Sortino | CO Sortino | Δ | 95% CI | BL DD | CO DD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, pod in pods.items():
        if pod is None:
            lines.append(f"| {name} | — | — | — | — | *not available* | — | — |")
            continue
        lines.append(
            f"| {name} | {pod['n_markets']} | {pod['bl_sortino']:+.4f} | "
            f"{pod['co_sortino']:+.4f} | **{pod['delta']:+.4f}** | {_fmt_ci(pod['ci'])} | "
            f"{pod['bl_maxdd']*100:+.2f}% | {pod['co_maxdd']*100:+.2f}% |"
        )
    lines.append("")

    # Replication analysis
    g7 = pods.get("G (seed=7)")
    g42 = pods.get("G (seed=42)")
    if g7 is not None and g42 is not None:
        lines.append("## Replication check (Pod G)")
        lines.append("")
        lines.append(
            f"- seed=7 observed delta: **{g7['delta']:+.4f}**  "
            f"(95% CI {_fmt_ci(g7['ci'])})"
        )
        lines.append(
            f"- seed=42 observed delta: **{g42['delta']:+.4f}**  "
            f"(95% CI {_fmt_ci(g42['ci'])})"
        )
        same_sign = (g7["delta"] > 0) == (g42["delta"] > 0)
        diff = abs(g7["delta"] - g42["delta"])
        lines.append(f"- same sign across seeds: **{same_sign}**")
        lines.append(f"- absolute seed-to-seed spread: **{diff:.4f}**")
        lines.append("")
        if same_sign and diff < 0.02:
            lines.append(
                "**Interpretation:** positive Sortino delta replicates under a second seed "
                "with tight variation — momentum screening effect is robust to initialization."
            )
        elif same_sign:
            lines.append(
                "**Interpretation:** same-sign replication but meaningful spread; result is "
                "directionally stable but absolute magnitude is seed-sensitive."
            )
        else:
            lines.append(
                "**Interpretation:** seed-to-seed sign flip — Pod G's positive delta is not "
                "robust and is within noise."
            )
        lines.append("")

    # Learnable inclusion analysis
    L = pods.get("L (learnable-inclusion)")
    if L is not None:
        lines.append("## Learnable inclusion (Pod L)")
        lines.append("")
        lines.append(
            f"- Universe: {L['n_markets']} markets (full, no pre-filter)"
        )
        lines.append(
            f"- Holdout Sortino: baseline {L['bl_sortino']:+.4f}, "
            f"constrained {L['co_sortino']:+.4f}, delta **{L['delta']:+.4f}**"
        )
        lines.append(
            f"- 95% CI on delta: {_fmt_ci(L['ci'])}"
        )
        lines.append(
            f"- Weight std (domain): {L['weight_std']:.4f}  "
            f"(equal-weight = 0; higher = more concentrated)"
        )
        lines.append(
            f"- Max domain exposure: {L['max_domain_exp']:.4f}  "
            f"(equal-weight on {L['n_markets']} markets ≈ {1.0/max(L['n_markets'],1):.4f})"
        )
        bp = L["best_params"]
        if bp:
            lines.append("")
            lines.append("**Selected hyperparameters (best trial):**")
            keys = [
                "learning_rate", "rolling_window", "domain_limit", "max_weight",
                "concentration_penalty_lambda", "covariance_penalty_lambda",
                "uniform_mix", "inclusion_init_gain", "inclusion_target_k",
                "lambda_inclusion_cardinality", "lambda_inclusion_commitment",
            ]
            for k in keys:
                if k in bp:
                    v = bp[k]
                    if isinstance(v, float):
                        lines.append(f"  - `{k}`: {v:.4f}")
                    else:
                        lines.append(f"  - `{k}`: {v}")
        lines.append("")

    # Comparison
    if L is not None and g7 is not None:
        lines.append("## Pod L (learnable) vs Pod G (hand-picked top-K)")
        lines.append("")
        delta_of_deltas = L["delta"] - g7["delta"]
        lines.append(
            f"- Pod G delta:     **{g7['delta']:+.4f}**  "
            f"(universe: top-20 momentum, 5d lookback)"
        )
        lines.append(
            f"- Pod L delta:     **{L['delta']:+.4f}**  "
            f"(universe: full 40 markets, learned inclusion gates)"
        )
        lines.append(f"- Pod L − Pod G:  {delta_of_deltas:+.4f}")
        lines.append("")
        if L["delta"] > g7["delta"] and L["delta"] > 0:
            lines.append(
                "**Interpretation:** learnable inclusion outperforms hand-picked top-K "
                "momentum. The optimizer's online selection is strictly better than the "
                "static pre-filter."
            )
        elif L["delta"] > 0 and g7["delta"] > 0:
            lines.append(
                "**Interpretation:** both approaches beat baseline. Hand-picked is still "
                "ahead — the static momentum prior is competitive or slightly better than "
                "the learned gate under the current regularizer settings."
            )
        elif L["delta"] <= 0 and g7["delta"] > 0:
            lines.append(
                "**Interpretation:** learnable inclusion does NOT beat baseline in this "
                "run, while hand-picked top-K does. The added flexibility (4 new "
                "hyperparameters, joint optimization) likely overfits the tuning set. "
                "Worth trying: stronger commitment penalty, smaller target_k, or "
                "combining learnable inclusion with hand-picked pre-filter."
            )
        else:
            lines.append("**Interpretation:** neither approach beats baseline cleanly.")
        lines.append("")

    out = REPO / "docs" / "direction_A_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
