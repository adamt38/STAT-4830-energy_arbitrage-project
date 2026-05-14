"""Run the fee-aware Majority strategy against the equal-domain baseline."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
(REPO_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.majority_strategy import run_majority_vs_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fee-aware majority-side PM strategy vs baseline.")
    parser.add_argument(
        "--artifact-prefix",
        default="final_week",
        help="Input processed artifact prefix (default: final_week).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix (default: {artifact-prefix}_majority_strategy).",
    )
    parser.add_argument(
        "--strategy-name",
        default="Majority strategy",
        help="Display name written to summaries and plot labels.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Per-unit L1 turnover transaction fee (default: 0.001 = 10 bps).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Majority threshold for YES or synthetic NO side selection.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=None,
        help="Optional trailing window in calendar days for saved comparison metrics.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional deterministic downsample cap for the saved comparison window.",
    )
    args = parser.parse_args()

    output_prefix = args.output_prefix or f"{args.artifact_prefix}_majority_strategy"
    majority, baseline, paths = run_majority_vs_baseline(
        REPO_ROOT,
        artifact_prefix=args.artifact_prefix,
        output_prefix=output_prefix,
        strategy_name=args.strategy_name,
        fee_rate=args.fee_rate,
        threshold=args.threshold,
        window_days=args.window_days,
        max_steps=args.max_steps,
    )

    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = min(majority.returns.size, baseline.returns.size, len(majority.timestamps), len(baseline.timestamps))
    timestamps = np.array(majority.timestamps[-n:], dtype=np.int64)
    selected = np.arange(n, dtype=np.int64)
    if args.window_days is not None and selected.size:
        start_ts = int(timestamps[-1] - float(args.window_days) * 86400.0)
        selected = selected[timestamps[selected] >= start_ts]
    if args.max_steps is not None and int(args.max_steps) > 0 and selected.size > int(args.max_steps):
        sampled = np.linspace(0, selected.size - 1, int(args.max_steps), dtype=np.int64)
        selected = selected[sampled]
    majority_returns = majority.returns[-n:][selected]
    baseline_returns = baseline.returns[-n:][selected]
    cm = np.cumprod(1.0 + majority_returns)
    cb = np.cumprod(1.0 + baseline_returns)
    majority_gain = float(cm[-1] - 1.0) if cm.size else 0.0
    baseline_gain = float(cb[-1] - 1.0) if cb.size else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cb, label="Baseline (equal-domain YES)", color="tab:blue", linewidth=2.0)
    ax.plot(cm, label=f"{args.strategy_name} (fee-aware)", color="tab:orange", linewidth=2.0)
    ax.set_title(f"{args.strategy_name} vs fee-aware baseline")
    ax.set_xlabel("Step")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    comparison_plot = fig_dir / f"{output_prefix}_vs_baseline.png"
    fig.savefig(comparison_plot, dpi=120)
    plt.close(fig)

    fig2, axb = plt.subplots(figsize=(7, 4.5))
    names = ["Baseline\n(equal-domain YES)", f"{args.strategy_name}\n(fee-aware)"]
    values = [100.0 * baseline_gain, 100.0 * majority_gain]
    bars = axb.bar(names, values, color=["tab:blue", "tab:orange"], width=0.55)
    axb.axhline(0.0, color="black", linewidth=0.8)
    axb.set_ylabel("Compound return (%)")
    axb.set_title(
        f"{args.strategy_name} excess = {100.0 * (majority_gain - baseline_gain):.2f} pp "
        f"({100.0 * majority_gain:.2f}% vs {100.0 * baseline_gain:.2f}%)"
    )
    ymax = max(max(abs(v) for v in values), 1.0)
    for bar, value in zip(bars, values, strict=False):
        dy = 0.04 * ymax * (1.0 if value >= 0 else -1.0)
        axb.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            f"{value:.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    axb.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    bar_plot = fig_dir / f"{output_prefix}_returns_bar.png"
    fig2.savefig(bar_plot, dpi=120)
    plt.close(fig2)

    paths["comparison_plot"] = comparison_plot
    paths["returns_bar_plot"] = bar_plot
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(
        json.dumps(
            {
                "strategy": args.strategy_name,
                "fee_rate": float(args.fee_rate),
                "threshold": float(args.threshold),
                "growth_of_1_majority": float(cm[-1]) if cm.size else 1.0,
                "growth_of_1_baseline": float(cb[-1]) if cb.size else 1.0,
                "excess_gain_vs_baseline": majority_gain - baseline_gain,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
