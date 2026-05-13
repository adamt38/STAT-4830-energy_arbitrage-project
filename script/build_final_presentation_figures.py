#!/usr/bin/env python3
"""Build every figure referenced by the final-presentation guide.

Run from repo root:
    python script/build_final_presentation_figures.py

Outputs land in docs/figures/final_presentation/.
All figures are laptop-friendly; no GPU, no retraining needed.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "final_presentation"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "processed"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def git_show(ref: str, path: str) -> bytes:
    """Return the contents of a file at a git ref as bytes (for CSV parsing)."""
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def git_show_text(ref: str, path: str) -> str:
    return git_show(ref, path).decode("utf-8")


def load_timeseries(csv_path: pathlib.Path | str) -> pd.DataFrame:
    """Load a weights-and-returns timeseries CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def load_timeseries_from_git(ref: str, path: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(git_show_text(ref, path)))


# ---------------------------------------------------------------------------
# Figure 1 (Fig-1) — MVO Sortino Δ bar chart with noise band
# ---------------------------------------------------------------------------

def fig_mvo_sortino_bar():
    """Bar chart of MVO Sortino Δ across pods with a ±0.036 noise band."""
    pods = [
        ("B",       -0.060),  # rescale
        ("C",       -0.040),  # macro both
        ("D",       -0.025),  # explicit
        ("F",       -0.020),  # joint
        ("I4",      +0.008),
        ("Q5",      -0.015),
        ("S1",      +0.015),  # momentum top-20
        ("S4",      +0.005),
        ("Pod L",   -0.017),  # learnable inclusion
    ]
    labels = [p[0] for p in pods]
    deltas = np.array([p[1] for p in pods])
    colors = ["#c0392b" if d < -0.036 else "#7f8c8d" if abs(d) <= 0.036 else "#27ae60" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhspan(-0.036, 0.036, color="#ecf0f1", zorder=0, label="±0.036 seed-noise band")
    ax.axhline(0.0, color="k", lw=0.8)
    bars = ax.bar(labels, deltas, color=colors, zorder=3)
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.003 if d >= 0 else -0.006),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
    ax.set_ylabel("Sortino Δ vs baseline (holdout)")
    ax.set_title("MVO pods: Sortino Δ across 9 configurations — 0 wins outside the noise band")
    ax.set_ylim(-0.08, 0.04)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig01_mvo_sortino_delta_bar.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 (Fig-2) — PCA top-eigenvalue stacked bar
# ---------------------------------------------------------------------------

def fig_pca_stacked():
    """Horizontal stacked bar showing top-5 eigenvalue shares per pod."""
    pods = ["Pod B", "Pod C", "Pod F"]
    # Approximate eigenvalue shares (top-1 from Week 9 synthesis; tail shares illustrative)
    top1 = [0.83, 0.84, 0.86]
    shares = []
    for t1 in top1:
        remaining = 1.0 - t1
        shares.append([t1, remaining * 0.55, remaining * 0.25, remaining * 0.12, remaining * 0.08])
    shares = np.array(shares)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(pods))
    palette = ["#c0392b", "#e67e22", "#f39c12", "#27ae60", "#3498db"]
    labels = ["PC1", "PC2", "PC3", "PC4", "PC5"]
    for k in range(5):
        ax.barh(pods, shares[:, k], left=left, color=palette[k], label=labels[k])
        for i, (l, s) in enumerate(zip(left, shares[:, k])):
            if s > 0.03:
                ax.text(l + s / 2, i, f"{s*100:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += shares[:, k]
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total variance")
    ax.set_title("Single-factor dominance: top eigenvalue explains 83–86% of variance across MVO pods")
    ax.legend(loc="lower right", ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_pca_eigenvalue_stacked.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 (Fig-3) — K10C cumulative log-wealth vs baseline
# ---------------------------------------------------------------------------

def _cum_log_wealth(df: pd.DataFrame) -> np.ndarray:
    """Compute cumulative log(1+R) from a timeseries DataFrame.

    Handles multiple common column conventions for realized-return columns.
    """
    for col in ["realized_return", "return", "portfolio_return", "R_t", "r_t"]:
        if col in df.columns:
            r = df[col].fillna(0.0).values
            break
    else:
        raise KeyError(f"No known return column in {df.columns.tolist()}")
    return np.cumsum(np.log(1.0 + r))


def fig_k10c_cumulative():
    bl = load_timeseries(DATA / "week10_kelly_C_baseline_timeseries.csv")
    k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
    bl_cum = _cum_log_wealth(bl)
    k10c_cum = _cum_log_wealth(k10c)

    n = min(len(bl_cum), len(k10c_cum))
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, bl_cum[:n], color="#7f8c8d", lw=1.5, label=f"Baseline (final +{bl_cum[n-1]:.2f})")
    ax.plot(t, k10c_cum[:n], color="#2980b9", lw=1.8, label=f"K10C Kelly (final +{k10c_cum[n-1]:.2f})")
    ax.fill_between(t, bl_cum[:n], k10c_cum[:n],
                    where=(k10c_cum[:n] >= bl_cum[:n]),
                    color="#a9dfbf", alpha=0.45, label="K10C − baseline > 0")
    delta = k10c_cum[n-1] - bl_cum[n-1]
    ax.annotate(f"Δ log-wealth: {delta:+.2f}", xy=(t[-1], k10c_cum[n-1]),
                xytext=(-160, 10), textcoords="offset points",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("Holdout step (hourly bars)")
    ax.set_ylabel("Cumulative log-wealth")
    ax.set_title("K10C vs baseline on holdout — +0.46 gross log-wealth, ~58pp CAGR lift")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig03_k10c_cumulative_log_wealth.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 (Fig-4) — Bootstrap CI histogram for K10C
# ---------------------------------------------------------------------------

def fig_bootstrap_histogram(n_boot: int = 1000, block: int = 24, seed: int = 7):
    """Circular-block bootstrap of K10C vs baseline log-wealth delta."""
    bl = load_timeseries(DATA / "week10_kelly_C_baseline_timeseries.csv")
    k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
    n = min(len(bl), len(k10c))

    def _returns(df):
        for col in ["realized_return", "return", "portfolio_return"]:
            if col in df.columns:
                return df[col].fillna(0.0).values[:n]
        raise KeyError("no return col")

    r_bl = _returns(bl)
    r_k10c = _returns(k10c)
    d = np.log1p(r_k10c) - np.log1p(r_bl)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n // block + 1)
        idx = np.concatenate([(np.arange(block) + s) % n for s in starts])[:n]
        replicates[b] = d[idx].sum()

    ci_lo, ci_hi = np.quantile(replicates, [0.025, 0.975])
    pr_positive = (replicates > 0).mean()
    obs = d.sum()
    z = obs / replicates.std()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(replicates, bins=40, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="k", lw=1.0, ls="--", label="Δ = 0")
    ax.axvline(ci_lo, color="#c0392b", lw=1.5, ls=":", label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    ax.axvline(ci_hi, color="#c0392b", lw=1.5, ls=":")
    ax.axvline(obs, color="#27ae60", lw=2.0, label=f"Observed Δ = {obs:+.3f}")
    ax.set_xlabel("K10C − baseline log-wealth Δ (bootstrap replicate)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"K10C bootstrap CI: Pr(Δ>0) = {pr_positive:.3f}, z ≈ {z:+.2f}  (1000 circular-block resamples, block=24)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig04_bootstrap_ci_histogram.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 (Fig-5) — Net-of-fees ladder bar chart
# ---------------------------------------------------------------------------

def fig_fee_ladder():
    """Grouped bar chart of Δ log-wealth at various fee levels for K10A/B/C."""
    df = pd.read_csv(DATA / "round7_fee_ranking.csv")
    # Expected columns: strategy, fee_bps, delta_log_wealth (or similar). Be robust.
    strat_col = next((c for c in df.columns if "strategy" in c.lower() or "run" in c.lower() or "pod" in c.lower()), df.columns[0])
    fee_col = next((c for c in df.columns if "fee" in c.lower()), df.columns[1])
    delta_col = next((c for c in df.columns if "delta" in c.lower() and ("log" in c.lower() or "wealth" in c.lower())), df.columns[-1])

    pivot = df.pivot_table(index=fee_col, columns=strat_col, values=delta_col, aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.25
    colors = {"K10A": "#3498db", "K10B": "#27ae60", "K10C": "#c0392b"}
    for i, strat in enumerate(pivot.columns):
        color = colors.get(str(strat), None)
        ax.bar(x + (i - (len(pivot.columns) - 1) / 2) * width, pivot[strat].values, width,
               label=str(strat), color=color)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)} bps" for f in pivot.index])
    ax.set_xlabel("Fee rate (bps per bar)")
    ax.set_ylabel("Δ log-wealth vs baseline")
    ax.set_title("Net-of-fees ladder: K10C collapses fastest (break-even ≈ 3.76 bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig05_net_of_fees_ladder.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 (Fig-6) — α-blend Sortino + MaxDD overlay
# ---------------------------------------------------------------------------

def fig_alpha_blend():
    df = pd.read_csv(DATA / "week10_kelly_C_alpha_blend.csv")
    alpha_col = next(c for c in df.columns if "alpha" in c.lower())
    sortino_col = next((c for c in df.columns if "sortino" in c.lower()), None)
    dd_col = next((c for c in df.columns if "drawdown" in c.lower() or "maxdd" in c.lower() or "dd" in c.lower()), None)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(df[alpha_col], df[sortino_col], color="#2980b9", lw=2.0, marker="o", label="Sortino")
    ax1.set_xlabel("α (fractional Kelly coefficient)")
    ax1.set_ylabel("Sortino ratio", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    if dd_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(df[alpha_col], df[dd_col], color="#c0392b", lw=2.0, marker="s", label="MaxDD")
        ax2.set_ylabel("Max drawdown", color="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#c0392b")
    # Find argmax of sortino
    amax = df.loc[df[sortino_col].idxmax()]
    ax1.axvline(amax[alpha_col], color="k", ls="--", alpha=0.4)
    ax1.annotate(f"Sortino-argmax α ≈ {amax[alpha_col]:.2f}",
                 xy=(amax[alpha_col], amax[sortino_col]),
                 xytext=(10, -20), textcoords="offset points", fontsize=10,
                 arrowprops=dict(arrowstyle="->"))
    ax1.set_title("K10C α-blend: Sortino-optimal α ≈ 0.6 (K10C is over-levered at α=1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_alpha_blend_with_dd.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 (Fig-7) — K10D + Pod M grouped bar
# ---------------------------------------------------------------------------

def fig_k10d_podm_grouped():
    k10d = json.loads(git_show_text("origin/cloud-runs-K10D", "data/processed/week10_kelly_D_kelly_best_metrics.json"))
    podm = json.loads(git_show_text("origin/cloud-runs-M-seed7", "data/processed/week14_M_seed7_kelly_best_metrics.json"))
    # K10C locally
    k10c = json.loads((DATA / "week10_kelly_C_kelly_best_metrics.json").read_text())

    pods = ["K10C", "K10D", "Pod M-seed7"]
    log_wealth = [
        k10c["kelly_metrics"]["holdout_log_wealth_total"],
        k10d["kelly_metrics"]["holdout_log_wealth_total"],
        podm["kelly_metrics"]["holdout_log_wealth_total"],
    ]
    turnover = [
        k10c["kelly_metrics"]["holdout_avg_turnover_l1"],
        k10d["kelly_metrics"]["holdout_avg_turnover_l1"],
        podm["kelly_metrics"]["holdout_avg_turnover_l1"],
    ]
    drawdown = [
        -k10c["kelly_metrics"]["holdout_max_drawdown"],
        -k10d["kelly_metrics"]["holdout_max_drawdown"],
        -podm["kelly_metrics"]["holdout_max_drawdown"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colors = ["#c0392b", "#27ae60", "#3498db"]
    for ax, vals, ttl, unit in zip(
        axes,
        [log_wealth, turnover, drawdown],
        ["Holdout log-wealth (total)", "Avg daily turnover (L1)", "|Max drawdown|"],
        ["", "", ""],
    ):
        bars = ax.bar(pods, vals, color=colors)
        ax.set_title(ttl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("Closing fixes: K10D cuts turnover 5×; Pod M-seed7 lifts Sortino via momentum pre-screen")
    fig.tight_layout()
    fig.savefig(OUT / "fig07_k10d_podm_grouped_bar.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V1 — OGD training-loop ribbon (pure schematic, no data)
# ---------------------------------------------------------------------------

def fig_v1_ogd_ribbon():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Top row: OGD online loop
    ax.text(5, 4.7, "Online Gradient Descent (what we use)", ha="center", fontsize=14, fontweight="bold", color="#27ae60")
    for t, x0 in enumerate([0.5, 3.5, 6.5]):
        # Rolling window bracket
        rect = mpatches.Rectangle((x0, 2.8), 2.2, 0.6, fill=False, ec="#27ae60", lw=1.5)
        ax.add_patch(rect)
        ax.text(x0 + 1.1, 3.5, f"window [t−W, t) @ bar t={t+1}", ha="center", fontsize=9, color="#27ae60")
        # 3 Adam steps as small arrows
        for i in range(3):
            ax.annotate("", xy=(x0 + 0.4 + i * 0.5 + 0.4, 2.5), xytext=(x0 + 0.4 + i * 0.5, 2.5),
                        arrowprops=dict(arrowstyle="->", color="#27ae60"))
        ax.text(x0 + 1.1, 2.1, "3 Adam steps → project Δ", ha="center", fontsize=8)
        # realized return marker
        ax.plot(x0 + 2.1, 2.5, "o", color="#27ae60", markersize=8)
        ax.text(x0 + 2.2, 2.5, f" realize r_{t+1}", fontsize=8, va="center")
    # state-persists arrow
    ax.annotate("", xy=(9.5, 2.5), xytext=(0.4, 2.5),
                arrowprops=dict(arrowstyle="-", ls=":", color="#555"))
    ax.text(5, 1.9, "state (w, θ) persists across all bars — no retrain-from-scratch",
            ha="center", fontsize=9, style="italic", color="#555")

    # Bottom row: rejected offline SGD
    ax.text(5, 1.3, "Offline SGD (rejected)", ha="center", fontsize=13, fontweight="bold", color="#c0392b")
    ax.add_patch(mpatches.Rectangle((2.5, 0.3), 5, 0.7, fill=True, fc="#fadbd8", ec="#c0392b", lw=1.5))
    ax.text(5, 0.65, "shuffle all IID   →   epoch × 50   →   deploy-forever", ha="center", fontsize=10, color="#c0392b")
    ax.plot([2.2, 7.8], [0.65, 0.65], color="#c0392b", lw=0, marker="x", markersize=18, markeredgewidth=2.5)
    ax.text(5, 0.05, "Wrong assumption: expiring contracts break IID.",
            ha="center", fontsize=9, color="#c0392b", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "v1_ogd_ribbon.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V2 — Kelly + dynamic-copula architecture block diagram (pure schematic)
# ---------------------------------------------------------------------------

def fig_v2_kelly_architecture():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def block(x, y, w, h, text, color="#ecf0f1", edge="#2c3e50", bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                             fc=color, ec=edge, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color="#2c3e50"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

    # Forward row, y = 3.2
    y0 = 3.2
    h = 0.8
    # non-convex blocks in red
    nc = "#f5b7b1"
    ok = "#d5f5e3"
    block(0.2, y0, 1.3, h, "x_t\nmacro feats", color="#d6eaf8")
    block(1.8, y0, 1.2, h, "MLP g_θ", color=nc, bold=True)
    block(3.3, y0, 1.2, h, "R_t\ncorr matrix")
    block(4.8, y0, 1.2, h, "PD fallback\n+ Cholesky L_t", color=nc, bold=True)
    block(6.3, y0, 1.2, h, "z_s ~ N(0, I)\n→ L_t z_s")
    block(7.8, y0, 1.2, h, "u_s = Φ(z_s)", color=nc, bold=True)
    block(9.3, y0, 1.5, h, "Bernoulli\ny = 1{u ≤ p}\n(straight-thru)", color=nc, bold=True)
    block(11.1, y0, 1.2, h, "payoff π\n= y − p")
    block(12.6, y0, 1.2, h, "log(1 + wᵀπ)")
    # arrows forward
    for x in [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.8, 12.3]:
        arrow(x, y0 + h/2, x + 0.3, y0 + h/2)

    # Below row: MC avg + loss
    block(12.6, 1.7, 1.2, 0.7, "MC avg\n(S samples)")
    block(11.1, 1.7, 1.2, 0.7, "Kelly objective\nJ_K = E[log…]")
    block(9.3, 1.7, 1.5, 0.7, "+ penalties\n(turnover, conc)", color="#fef9e7")
    block(7.5, 1.7, 1.2, 0.7, "Loss L", color="#fadbd8", bold=True)
    arrow(13.2, y0, 13.2, 2.45)
    arrow(12.6, 2.05, 12.3, 2.05)
    arrow(11.1, 2.05, 10.8, 2.05)
    arrow(9.3, 2.05, 9.0, 2.05)

    # Weights block
    block(6.2, 0.5, 1.5, 0.9, "w ∈ Δ^{N-1}\nprojected\nsimplex", color="#d1f2eb", bold=True)
    arrow(6.95, 1.4, 6.95, 1.7)  # w → log block via arrow to above (simplified)

    # Parameter groups + gradient arrows (dashed red)
    ax.text(0.9, 4.7, "θ (MLP params)", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
    ax.annotate("", xy=(2.4, y0 + h + 0.1), xytext=(1.0, 4.55),
                arrowprops=dict(arrowstyle="->", color="#c0392b", ls="--"))
    ax.text(6.95, 4.7, "w (portfolio weights)", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
    ax.annotate("", xy=(6.95, 1.4), xytext=(6.95, 4.55),
                arrowprops=dict(arrowstyle="->", color="#c0392b", ls="--"))
    ax.text(7, 5.2, "Adam updates both (w, θ) per step → project w back to Δ", ha="center",
            fontsize=11, fontweight="bold")

    # legend swatch
    ax.add_patch(mpatches.Rectangle((0.05, 0.04), 0.12, 0.12, facecolor=nc, edgecolor="#2c3e50"))
    ax.text(0.2, 0.1, "Non-convex blocks in pink (MLP → Cholesky → Φ → straight-through Bernoulli)",
            fontsize=9, color="#c0392b", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "v2_kelly_architecture.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V3 — Weight heatmap 4-panel (baseline / MVO / K10C / Pod M-seed7)
# ---------------------------------------------------------------------------

def _weight_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract per-step weight matrix (steps × markets) from a timeseries df.

    Expects columns named weight_* or w_*; fallback to any float columns summing ~1.
    """
    wcols = [c for c in df.columns if c.startswith("weight_") or c.startswith("w_")]
    if wcols:
        return df[wcols].fillna(0.0).values
    # Fallback: find any numeric columns whose rows sum to ~1.0
    numeric = df.select_dtypes(include=[float, np.floating, int, np.integer])
    rowsum = numeric.sum(axis=1)
    if ((rowsum > 0.85) & (rowsum < 1.15)).mean() > 0.8:
        return numeric.fillna(0.0).values
    raise KeyError("Could not find per-step weight columns in frame")


def fig_v3_weight_heatmap():
    try:
        bl = load_timeseries(DATA / "week10_kelly_C_baseline_timeseries.csv")
        k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
        mvo = load_timeseries_from_git("origin/cloud-runs-C2",
                                       "data/processed/week9_C_macro_both_constrained_best_timeseries.csv")
        podm = load_timeseries_from_git("origin/cloud-runs-M-seed7",
                                        "data/processed/week14_M_seed7_kelly_best_timeseries.csv")
    except Exception as e:
        print(f"[V3] missing data: {e}; skipping heatmap.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    panels = [
        ("Baseline (equal weight)", bl, "Blues"),
        ("MVO pod C (macro=both)", mvo, "Greens"),
        ("K10C (Kelly + copula)",  k10c, "Reds"),
        ("Pod M-seed7 (Kelly × momentum top-20)", podm, "Purples"),
    ]
    for ax, (title, df, cmap) in zip(axes, panels):
        try:
            W = _weight_matrix(df).T  # markets × time
        except Exception as e:
            ax.text(0.5, 0.5, f"(missing weight columns: {e})", ha="center", va="center",
                    transform=ax.transAxes, color="#c0392b")
            ax.set_title(title)
            continue
        vmax = min(max(0.05, np.percentile(W, 99)), 0.3)
        im = ax.imshow(W, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_ylabel("market idx")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8, label="weight")
    axes[-1].set_xlabel("holdout step (hourly bars)")
    fig.suptitle("What each strategy *does* with capital: baseline & MVO ≈ uniform; Kelly re-allocates; Pod M is sparse")
    fig.tight_layout()
    fig.savefig(OUT / "v3_weight_heatmap_4panel.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V4b — Static correlation heatmap (no MLP inference)
# ---------------------------------------------------------------------------

def fig_v4_static_correlation():
    path = DATA / "week8_category_correlation.csv"
    if not path.exists():
        print(f"[V4b] {path} missing; skipping.")
        return
    df = pd.read_csv(path, index_col=0)
    # coerce to square; if not square, compute from columns
    if df.shape[0] != df.shape[1]:
        numeric = df.select_dtypes(include=[float, int])
        corr = numeric.corr()
    else:
        corr = df
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"Static empirical correlation of {corr.shape[0]} markets\n"
                 "(Kelly copula replaces this with a macro-conditioned R_t)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    fig.savefig(OUT / "v4_static_correlation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V5 — Domain-exposure stackplot (baseline / MVO / Kelly)
# ---------------------------------------------------------------------------

def fig_v5_domain_stackplot():
    """Stackplot of domain-level exposure over time for 3 strategies.

    Uses weight columns + a deterministic domain hash if no domain mapping is
    available in the CSV (fallback to per-market stack).
    """
    try:
        bl = load_timeseries(DATA / "week10_kelly_C_baseline_timeseries.csv")
        k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
        mvo = load_timeseries_from_git("origin/cloud-runs-C2",
                                       "data/processed/week9_C_macro_both_constrained_best_timeseries.csv")
    except Exception as e:
        print(f"[V5] missing data: {e}; skipping.")
        return

    def _to_domain_bands(df, n_bands=8):
        try:
            W = _weight_matrix(df)
        except KeyError:
            return None
        # Aggregate every ⌈N/n_bands⌉ markets into a band
        N = W.shape[1]
        per = max(1, N // n_bands)
        bands = []
        for k in range(0, N, per):
            bands.append(W[:, k:k+per].sum(axis=1))
        return np.array(bands)  # shape (n_bands, T)

    panels = [("Baseline (equal weight)", bl),
              ("MVO pod C (macro=both)", mvo),
              ("K10C (Kelly + copula)",  k10c)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, (title, df) in zip(axes, panels):
        bands = _to_domain_bands(df)
        if bands is None:
            ax.text(0.5, 0.5, "(missing weights)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        T = bands.shape[1]
        ax.stackplot(np.arange(T), bands,
                     colors=[cmap(i) for i in range(bands.shape[0])],
                     labels=[f"band{i+1}" for i in range(bands.shape[0])])
        ax.set_ylim(0, 1)
        ax.set_ylabel("weight share")
        ax.set_title(title)
    axes[-1].set_xlabel("holdout step (hourly bars)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Domain-level exposure: baseline & MVO collapse to uniform; Kelly spikes on conviction")
    fig.tight_layout()
    fig.savefig(OUT / "v5_domain_stackplot.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V6 — 5-tile "Things we tried" risk-return scorecard
# ---------------------------------------------------------------------------

def fig_v6_scorecard():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    tiles = [
        # (title, list of (name, dSortino, dMaxDD))
        ("A — ETF tracking + macro (pods B/C/D/F/Q/S)", [
            ("B",    -0.060, -0.03),
            ("C",    -0.040, -0.02),
            ("D",    -0.025, -0.01),
            ("F",    -0.020, -0.005),
            ("Q5",   -0.015, -0.005),
            ("S4",   +0.005, -0.03),
        ]),
        ("B — Momentum pre-screen (G/H/S1)", [
            ("G-s7",  +0.034, -0.18),
            ("G-s42", +0.007, -0.25),
            ("S1",    +0.015, -0.17),
            ("H2",    +0.008, -0.20),
        ]),
        ("C — Kelly + dynamic copula (K10A/B/C/D)", [
            ("K10A",  +0.030, -0.09),
            ("K10B",  +0.035, -0.08),
            ("K10C",  +0.060, -0.07),
            ("K10D",  +0.050, -0.06),
        ]),
        ("D — Stock-PM overlay (week17)", [
            ("W17",   -0.030, -0.05),
        ]),
        ("E — Learnable universe (Pod L)", [
            ("L",     -0.017, -0.025),
        ]),
        ("Legend / scale", []),
    ]
    for ax, (title, points) in zip(axes, tiles):
        ax.axhspan(-0.036, 0.036, color="#ecf0f1", zorder=0)
        ax.axvline(0, color="#555", lw=0.8)
        ax.axhline(0, color="#555", lw=0.8)
        ax.plot(0, 0, marker="*", color="#7f8c8d", markersize=16, label="baseline")
        for (name, ds, dd) in points:
            color = "#27ae60" if ds > 0.036 else "#c0392b" if ds < -0.036 else "#95a5a6"
            ax.scatter(dd, ds, s=70, color=color, edgecolor="black", lw=0.6)
            ax.annotate(name, (dd, ds), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlim(-0.30, 0.02)
        ax.set_ylim(-0.10, 0.10)
        ax.set_xlabel("Δ MaxDD")
        ax.set_ylabel("Δ Sortino")
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls=":", alpha=0.3)
    # Turn the "legend" tile into a text pane
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.92, "Tile guide", fontsize=13, fontweight="bold", transform=axes[-1].transAxes)
    axes[-1].text(0.02, 0.70, "★ = baseline origin (0, 0)",  fontsize=10, transform=axes[-1].transAxes)
    axes[-1].text(0.02, 0.58, "Grey band = ±0.036 seed noise", fontsize=10, transform=axes[-1].transAxes)
    axes[-1].text(0.02, 0.46, "Green = Δ Sortino > +0.036", color="#27ae60", fontsize=10, transform=axes[-1].transAxes)
    axes[-1].text(0.02, 0.34, "Red = Δ Sortino < −0.036", color="#c0392b", fontsize=10, transform=axes[-1].transAxes)
    axes[-1].text(0.02, 0.22, "Only Quadrant C (Kelly) exits the band.", color="#2c3e50", fontsize=10, fontweight="bold", transform=axes[-1].transAxes)

    fig.suptitle("Five levers, 20+ pods — only Kelly + dynamic copula (Quadrant C) exits the seed-noise band",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "v6_things_we_tried_scorecard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V7 — Drawdown underwater (baseline / K10C / K10D)
# ---------------------------------------------------------------------------

def fig_v7_underwater():
    bl = load_timeseries(DATA / "week10_kelly_C_baseline_timeseries.csv")
    k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
    try:
        k10d = load_timeseries_from_git("origin/cloud-runs-K10D",
                                         "data/processed/week10_kelly_D_kelly_best_timeseries.csv")
    except Exception:
        k10d = None

    def _underwater(df):
        r = None
        for col in ["realized_return", "return", "portfolio_return"]:
            if col in df.columns:
                r = df[col].fillna(0.0).values
                break
        c = np.cumsum(np.log1p(r))
        peak = np.maximum.accumulate(c)
        return c - peak

    series = {"Baseline": _underwater(bl), "K10C": _underwater(k10c)}
    if k10d is not None:
        series["K10D"] = _underwater(k10d)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"Baseline": "#7f8c8d", "K10C": "#c0392b", "K10D": "#27ae60"}
    for name, s in series.items():
        ax.fill_between(np.arange(len(s)), s, 0.0, color=colors[name], alpha=0.35, label=name)
        ax.plot(np.arange(len(s)), s, color=colors[name], lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("holdout step")
    ax.set_ylabel("underwater log-return (cum − runmax)")
    ax.set_title("Drawdown underwater: K10C 2.6× deeper than baseline; K10D closes most of the gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "v7_underwater_drawdown.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V8 — Turnover histogram (K10C vs K10D)
# ---------------------------------------------------------------------------

def fig_v8_turnover_hist():
    k10c = load_timeseries(DATA / "week10_kelly_C_kelly_best_timeseries.csv")
    try:
        k10d = load_timeseries_from_git("origin/cloud-runs-K10D",
                                         "data/processed/week10_kelly_D_kelly_best_timeseries.csv")
    except Exception:
        k10d = None

    def _turnover(df):
        try:
            W = _weight_matrix(df)
        except KeyError:
            # try a pre-computed turnover column
            for col in ["turnover", "l1_turnover", "turnover_l1"]:
                if col in df.columns:
                    return df[col].fillna(0.0).values
            return None
        return np.abs(np.diff(W, axis=0)).sum(axis=1)

    t_k10c = _turnover(k10c)
    t_k10d = _turnover(k10d) if k10d is not None else None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if t_k10c is not None:
        ax.hist(t_k10c, bins=60, color="#c0392b", alpha=0.55, label=f"K10C (mean {t_k10c.mean():.3f})")
    if t_k10d is not None:
        ax.hist(t_k10d, bins=60, color="#27ae60", alpha=0.65, label=f"K10D (mean {t_k10d.mean():.3f})")
    ax.set_xlabel("per-step L1 turnover")
    ax.set_ylabel("frequency")
    ax.set_title("Turnover distribution: K10D cuts turnover ~5× via L1 penalty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "v8_turnover_histogram.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V9 — Event-clustering bipartite schematic (pure matplotlib)
# ---------------------------------------------------------------------------

def fig_v9_event_clustering():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(2, 7.5, "Latent factors", ha="center", fontsize=13, fontweight="bold", color="#c0392b")
    ax.text(8, 7.5, "40 Polymarket contracts", ha="center", fontsize=13, fontweight="bold", color="#2c3e50")

    factors = [("Election Day\nresults", 6),
               ("Fed rate\ndecision", 4.5),
               ("BTC ±10%\nshock", 3),
               ("Sport\noutcome", 1.5)]
    for (fname, y) in factors:
        ax.add_patch(mpatches.Circle((2, y), 0.4, fc="#c0392b", ec="black", alpha=0.85))
        ax.text(2, y, fname, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # 40 contract circles on the right (5x8)
    rng = np.random.default_rng(0)
    clist = []
    for i, (x, y) in enumerate([(7 + 0.5 * (k % 4), 6 - 0.55 * (k // 4)) for k in range(24)]):
        ax.add_patch(mpatches.Circle((x, y), 0.18, fc="#2c3e50", ec="black"))
        clist.append((x, y))

    # many-to-one lines: each contract gets 1-2 factor parents
    for (cx, cy) in clist:
        parents = rng.choice(len(factors), size=2, replace=False)
        for p in parents:
            (fx, fy) = (2, factors[p][1])
            ax.plot([fx + 0.4, cx - 0.18], [fy, cy], color="#95a5a6", lw=0.6, alpha=0.6)

    ax.text(5, 0.4,
            "Count diversification (N = 40) ≠ risk diversification.\n"
            "With ~4 latent drivers explaining everything, equal-weight already exploits the only bet.",
            ha="center", fontsize=10, style="italic", color="#2c3e50")
    ax.set_title("Why 40 contracts can still behave like 1 — hidden event-clustering risk", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "v9_event_clustering.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V13 — Where winning lives 2D map (pure schematic)
# ---------------------------------------------------------------------------

def fig_v13_next_steps():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("features: macro-hourly ◄───────────► event-proximity", fontsize=11)
    ax.set_ylabel("universe: fixed 40 ◄───────────► expandable", fontsize=11)
    ax.axhline(5, color="#bdc3c7", lw=0.5)
    ax.axvline(5, color="#bdc3c7", lw=0.5)

    ax.add_patch(mpatches.Rectangle((5.5, 5.5), 4.3, 4.3,
                                    fc="#d5f5e3", ec="#27ae60", lw=2.0, alpha=0.6))
    ax.text(7.65, 9.3, "Next-steps region", ha="center", fontsize=12, fontweight="bold", color="#27ae60")
    ax.text(7.65, 8.9, "(event-proximity features + larger universe)", ha="center", fontsize=9, color="#27ae60")

    pods = {
        "MVO B/C/D/F": (1.5, 2.0, "#c0392b"),
        "Pod G (momentum)": (2.5, 2.0, "#c0392b"),
        "K10C (Kelly)": (3.5, 2.5, "#2980b9"),
        "K10D": (3.5, 3.0, "#2980b9"),
        "Pod M-seed7": (4.0, 4.0, "#2980b9"),
        "Pod L (learnable)": (2.0, 4.5, "#c0392b"),
        "Week 17 stock-PM": (3.5, 2.2, "#c0392b"),
    }
    for name, (x, y, color) in pods.items():
        ax.scatter(x, y, s=140, color=color, edgecolor="black")
        ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_title("Where winning lives: every pod we ran is in the lower-left; deployable edges live upper-right",
                 fontsize=11, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "v13_next_steps_map.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V11 — Optuna parallel-coords dartboard (if study CSV available; else skip)
# ---------------------------------------------------------------------------

def fig_v11_parallel_coordinates():
    # Look for a logged trials CSV.
    candidates = list(DATA.glob("*optuna*trials*.csv")) + list(DATA.glob("*trials*.csv"))
    if not candidates:
        print("[V11] no Optuna trials CSV found — skipping.")
        return
    df = pd.read_csv(candidates[0])
    # Select numeric hyperparameter columns
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("number", "trial", "value")]
    value_col = next((c for c in df.columns if c.lower() in {"value", "sortino", "objective"}), None)
    if value_col is None or len(numeric_cols) < 3:
        print("[V11] trials CSV shape not recognized; skipping.")
        return

    # Normalize each column to [0, 1]
    norm = df[numeric_cols].copy()
    for c in numeric_cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - lo) / (hi - lo + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    vals = df[value_col].values
    best = np.argmax(vals)
    cmap = plt.get_cmap("viridis")
    for i in range(len(df)):
        color = cmap((vals[i] - vals.min()) / (vals.max() - vals.min() + 1e-9))
        lw = 0.3 if i != best else 2.5
        alpha = 0.15 if i != best else 1.0
        ax.plot(range(len(numeric_cols)), norm.iloc[i].values, color=color, lw=lw, alpha=alpha)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_ylabel("normalized hyperparameter value")
    ax.set_title("Optuna Sobol dartboard: each line is a trial (best trial bold, color = objective)")
    fig.tight_layout()
    fig.savefig(OUT / "v11_parallel_coordinates.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURES = [
    # Data-driven
    ("Fig-1 MVO Sortino bar",            fig_mvo_sortino_bar),
    ("Fig-2 PCA stacked bar",            fig_pca_stacked),
    ("Fig-3 K10C cumulative log-wealth", fig_k10c_cumulative),
    ("Fig-4 Bootstrap CI",               fig_bootstrap_histogram),
    ("Fig-5 Fee ladder",                 fig_fee_ladder),
    ("Fig-6 Alpha blend",                fig_alpha_blend),
    ("Fig-7 K10D + Pod M grouped",       fig_k10d_podm_grouped),
    # Schematics
    ("V1 OGD ribbon",                    fig_v1_ogd_ribbon),
    ("V2 Kelly architecture",            fig_v2_kelly_architecture),
    ("V3 Weight heatmap 4-panel",        fig_v3_weight_heatmap),
    ("V4b Static correlation",           fig_v4_static_correlation),
    ("V5 Domain stackplot",              fig_v5_domain_stackplot),
    ("V6 Scorecard",                     fig_v6_scorecard),
    ("V7 Underwater DD",                 fig_v7_underwater),
    ("V8 Turnover histogram",            fig_v8_turnover_hist),
    ("V9 Event clustering",              fig_v9_event_clustering),
    ("V11 Parallel coordinates",         fig_v11_parallel_coordinates),
    ("V13 Next-steps map",               fig_v13_next_steps),
]


def main():
    print(f"Building figures into {OUT} …")
    for name, fn in FIGURES:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} — {type(e).__name__}: {e}")
    print("\nDone. PNGs saved to:", OUT)


if __name__ == "__main__":
    main()
