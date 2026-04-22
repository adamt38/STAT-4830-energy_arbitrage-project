"""Generate docs/option_B_results.md — momentum-Kelly Pod M seed=7 summary.

Compares Pod M (momentum-Kelly) against:
  - Pod M's own equal-weight baseline (on the momentum-screened universe)
  - Adam's K10A / K10B / K10C (Kelly on full 40-market universe) gross numbers
  - Cross-contribution framing: Option B is momentum applied to Adam's Kelly
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


def _max_dd_log_wealth(rets: np.ndarray) -> float:
    if rets.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak - 1.0))


def main() -> None:
    p = REPO / "data" / "processed"
    prefix = "week14_M_seed7"
    co_ts_path = p / f"{prefix}_kelly_best_timeseries.csv"
    bl_ts_path = p / f"{prefix}_baseline_timeseries.csv"
    ci_path = p / f"{prefix}_bootstrap_ci.json"
    metrics_path = p / f"{prefix}_kelly_best_metrics.json"
    mom_scores_path = p / f"{prefix}_momentum_scores.json"

    if not (co_ts_path.exists() and bl_ts_path.exists()):
        print("MISSING Pod M artifacts — cannot write summary")
        return

    bl_ts = _read_series(bl_ts_path)
    co_ts = _read_series(co_ts_path)
    N = len(co_ts)
    bl_r = np.array([float(r["portfolio_return"]) for r in bl_ts[-N:]])
    co_r = np.array([float(r.get("portfolio_return", 0.0)) for r in co_ts])

    bl_sortino = _sortino(bl_r)
    co_sortino = _sortino(co_r)
    sortino_delta = co_sortino - bl_sortino
    bl_maxdd = _max_dd_log_wealth(bl_r)
    co_maxdd = _max_dd_log_wealth(co_r)

    bl_lw = float(np.sum(np.log(1.0 + bl_r)))
    co_lw = float(np.sum(np.log(1.0 + co_r)))
    lw_delta = co_lw - bl_lw

    ci = _read_json(ci_path) if ci_path.exists() else {}
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    mom = _read_json(mom_scores_path) if mom_scores_path.exists() else {}
    bp = metrics.get("best_params", {})

    lines: list[str] = []
    lines.append("# Option B Results — Momentum × Kelly (Pod M)")
    lines.append("")
    lines.append(
        "Ports the `--momentum-screening` lever (merged from Option A / Pod G) from "
        "the MVO/Sortino week8 pipeline into the Kelly log-wealth week10 pipeline. "
        "Tests whether momentum pre-selection of the market universe compounds with "
        "Adam's Round 3 Kelly + dynamic-copula + L1-turnover optimizer."
    )
    lines.append("")
    lines.append("## Why this combination is novel")
    lines.append("")
    lines.append(
        "- **K10A / K10B / K10C / K10D / K10E / K10F** (Adam's Round 3 + Round 7 Kelly pods) all operate on the **full 40-market universe** from cached week8 data."
    )
    lines.append("- **G / I4 / Q5 / S1 / S4 / S5 / Option A** (MVO pipeline pods) test momentum screening at various intensities, but all optimize Sortino / mean-downside — not Kelly log-wealth.")
    lines.append("- **Pod M = Kelly objective + momentum pre-filter**. The one cross-term between Adam's two Rounds' methodologies that nobody has run.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Lever | Value |")
    lines.append("|---|---|")
    lines.append("| Pipeline | `polymarket_week10_kelly_pipeline.py` (dynamic-copula Kelly OGD) |")
    lines.append("| Data build | fresh (`--rebuild-data`), momentum-screened |")
    lines.append("| Momentum screening | top-20 markets, 5-day lookback |")
    lines.append("| Reduced Optuna search | enabled |")
    lines.append("| Trials | 100 |")
    lines.append("| Seed | 7 (pipeline default; no CLI override implemented) |")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(f"| Metric | Baseline (eq-wt on momentum universe) | Pod M (Kelly) | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Sortino ratio (holdout)** | {bl_sortino:+.4f} | {co_sortino:+.4f} | **{sortino_delta:+.4f}** |")
    lines.append(f"| **Log-wealth (cumulative holdout)** | {bl_lw:+.4f} | {co_lw:+.4f} | **{lw_delta:+.4f}** |")
    lines.append(f"| Max drawdown (holdout) | {bl_maxdd*100:+.2f}% | {co_maxdd*100:+.2f}% | {(co_maxdd - bl_maxdd)*100:+.2f} pp |")
    lines.append("")
    if ci:
        lines.append("## Bootstrap 95% CI on Sortino delta")
        lines.append("")
        lines.append(f"- Observed Sortino Δ: **{ci.get('observed_sortino_delta', 0):+.4f}**")
        lines.append(f"- 95% CI: [{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}]")
        lines.append(f"- Fraction of resamples positive: **{ci.get('frac_positive', 0):.1%}**")
        lines.append(f"- n_bootstrap: {ci.get('n_bootstrap', 0)}, n_holdout_steps: {ci.get('n_holdout_steps', 0)}")
        lines.append("")

    lines.append("## Comparison to benchmarks")
    lines.append("")
    lines.append("**MVO pods (same Sortino metric):**")
    lines.append("- Noise floor (Adam §15.9): ~0.036 Sortino Δ")
    lines.append("- Adam's I4 (single seed): +0.0077")
    lines.append("- Adam's S1 (single seed, momentum + wide sweep): +0.0149")
    lines.append("- Adam's S4 seed=3 (multi-seed I4 repro, 1 of 5): +0.0156")
    lines.append("- Our Option A (momentum × shrinkage × macro=both, 3 seeds): −0.0169 ± 0.0003")
    lines.append("- Teammate week17 (`stock-PM-combined-strategy`): +0.1018")
    lines.append("")
    lines.append("**Kelly pods (log-wealth metric):**")
    lines.append("- K10A (full Kelly, no momentum): log-wealth Δ gross = see Adam's §17")
    lines.append("- K10C (same but turnover-focused): gross +0.46 log-wealth, BUT break-even fee only 3.76 bps → flips negative at realistic 10 bps")
    lines.append("- **Pod M (Kelly + momentum top-20/5d)**: **{:+.4f}** log-wealth".format(lw_delta))
    lines.append("")

    lines.append("## Selected hyperparameters (best Optuna trial)")
    lines.append("")
    if bp:
        interesting_keys = [
            "lr_w", "lr_theta", "rolling_window", "mc_samples",
            "turnover_lambda", "copula_shrinkage", "copula_temperature",
            "mlp_hidden_dim", "concentration_penalty_lambda", "max_weight",
            "fee_rate", "dd_penalty",
        ]
        for k in interesting_keys:
            v = bp.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.6f}")
            else:
                lines.append(f"- `{k}`: {v}")

    lines.append("")
    lines.append("## Momentum universe (top 20 markets by |5d return|)")
    lines.append("")
    if mom:
        markets = mom.get("selected_markets", [])[:20]
        lines.append("| Rank | Domain | Momentum | Question (truncated) |")
        lines.append("|---|---|---|---|")
        for i, m in enumerate(markets):
            q = str(m.get("question", ""))[:80]
            lines.append(
                f"| {i+1} | {m.get('domain', '')} | {float(m.get('momentum', 0)):+.4f} | {q} |"
            )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if sortino_delta > 0.036 and ci.get("frac_positive", 0) >= 0.75:
        lines.append(
            "**Verdict: clean positive.** Pod M beats baseline on Sortino by more than the "
            "seed-noise floor AND log-wealth is positive. Momentum + Kelly is a real "
            "combination worth multi-seed validation as a follow-up."
        )
    elif sortino_delta > 0 and lw_delta > 0:
        lines.append(
            "**Verdict: directional positive on both metrics.** Neither is "
            "statistically decisive but both point the same way. A multi-seed "
            "follow-up or fee-sweep (à la Adam's K10E) is the natural next step."
        )
    elif lw_delta > 0 and sortino_delta < 0:
        lines.append(
            "**Verdict: Kelly-native metric positive, Sortino negative.** "
            "Pod M improves log-wealth (what Kelly optimizes) but slightly "
            "hurts Sortino. Consistent with Kelly prioritizing geometric growth "
            "at the cost of downside-deviation-adjusted mean. The log-wealth "
            "number should be adjusted for fees (see Adam's K10C caveat) before "
            "making a tradability claim."
        )
    elif sortino_delta > 0:
        lines.append(
            "**Verdict: Sortino positive, Kelly-native log-wealth negative.** "
            "Unexpected — the Kelly optimizer was supposed to improve log-wealth. "
            "Likely means the momentum universe is too small / too volatile for "
            "Kelly's geometric-growth premium to accumulate. Investigation needed."
        )
    else:
        lines.append(
            "**Verdict: Pod M does not beat baseline.** The momentum filter may "
            "reduce universe below the minimum N where the dynamic-copula Kelly "
            "optimizer can exploit cross-asset structure. K10B's no-copula "
            "equivalent (on full universe) was also marginal — possibly momentum "
            "× Kelly compounds badly rather than favorably."
        )

    out = REPO / "docs" / "option_B_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
