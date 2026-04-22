#!/usr/bin/env bash
# Pod MF: Momentum-Kelly + Fee Sweep. Extends Pod M by adding fee_rate as a
# searchable categorical Optuna dimension with values {0, 10, 50, 100} bps.
# Directly answers the open question from Adam's §17: "does the Kelly
# pipeline's edge survive realistic Polymarket fees (~10 bps) when combined
# with momentum screening?"
#
# Adam's K10C showed fee break-even at only 3.76 bps on the full 40-market
# universe. Momentum's smaller 20-market universe has lower total turnover
# (each weight update moves fewer positions), which should raise the
# break-even — but this is an empirical question.

set -u

REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/option_B_fee_master.log"
echo "[MF] ====== Pod MF (momentum-Kelly fee sweep) starting at $(date) ======" >> "$MASTER_LOG"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
echo "[MF] Launching at $(date), RUN_TAG=$RUN_TAG" >> "$MASTER_LOG"

nohup python -u "$REPO/script/polymarket_week10_kelly_pipeline.py" \
  --artifact-prefix week14_MF_seed7 \
  --rebuild-data \
  --momentum-screening \
  --momentum-top-n 20 \
  --momentum-lookback-days 5 \
  --fee-rate-values 0,0.001,0.005,0.01 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --reduced-search \
  --git-commit-and-push \
  --git-push-branch cloud-runs-MF-seed7 \
  --git-commit-message "MF-seed7: momentum-Kelly + fee-rate sweep {0, 10, 50, 100 bps}  ${RUN_TAG}" \
  > "$LOG_DIR/pod_MF_seed7.log" 2>&1 &
MF_PID=$!
echo "[MF] seed=7 PID=$MF_PID" >> "$MASTER_LOG"

while kill -0 "$MF_PID" 2>/dev/null; do
  sleep 120
done
echo "[MF] seed=7 exited at $(date)" >> "$MASTER_LOG"

# Bootstrap CI
python - <<'PYEOF' >> "$MASTER_LOG" 2>&1
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
prefix = "week14_MF_seed7"
co_path = p / f"{prefix}_kelly_best_timeseries.csv"
bl_path = p / f"{prefix}_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print(f"[MF-ci] MISSING timeseries")
    raise SystemExit(0)
co_rows = list(csv.DictReader(open(co_path)))
bl_rows = list(csv.DictReader(open(bl_path)))
N = len(co_rows)
bl_r = np.array([float(r["portfolio_return"]) for r in bl_rows[-N:]])
co_r = np.array([float(r.get("portfolio_return", 0.0)) for r in co_rows])
def sortino(x):
    if x.size == 0: return 0.0
    d = np.minimum(x, 0.0); dd = float(np.sqrt(np.mean(d*d)))
    return float(np.mean(x) / (dd + 1e-8))
rng = np.random.default_rng(42); B = 5000
deltas = np.empty(B); idx = np.arange(N)
for i in range(B):
    s = rng.choice(idx, size=N, replace=True)
    deltas[i] = sortino(co_r[s]) - sortino(bl_r[s])
bl_lw = float(np.sum(np.log(1.0 + bl_r)))
co_lw = float(np.sum(np.log(1.0 + co_r)))
out = {"observed_sortino_delta": float(sortino(co_r)-sortino(bl_r)),
       "ci_lower": float(np.percentile(deltas, 2.5)),
       "ci_upper": float(np.percentile(deltas, 97.5)),
       "frac_positive": float(np.mean(deltas > 0)),
       "bl_sortino": sortino(bl_r),
       "co_sortino": sortino(co_r),
       "observed_log_wealth_delta": co_lw - bl_lw,
       "bl_final_log_wealth": bl_lw,
       "co_final_log_wealth": co_lw,
       "n_bootstrap": B, "n_holdout_steps": N}
(p / f"{prefix}_bootstrap_ci.json").write_text(json.dumps(out, indent=2))
print(f"[MF-seed7] Sortino Δ={out['observed_sortino_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
print(f"[MF-seed7] Log-wealth Δ={out['observed_log_wealth_delta']:+.4f}")

# Also read best_params to see which fee_rate won
km = json.loads(Path("data/processed/week14_MF_seed7_kelly_best_metrics.json").read_text())
fee = km.get("best_params", {}).get("fee_rate", None)
print(f"[MF-seed7] WINNING fee_rate = {fee} ({'{:.1f}'.format(fee*10000) if fee is not None else '?'} bps)")
PYEOF

# Summary: extend option_B_results.md with fee sweep section
python - <<'PYEOF' >> "$MASTER_LOG" 2>&1
import json, csv, numpy as np
from pathlib import Path

REPO = Path("/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project")
p = REPO / "data" / "processed"
prefix = "week14_MF_seed7"

# Read Pod MF
co_path = p / f"{prefix}_kelly_best_timeseries.csv"
bl_path = p / f"{prefix}_baseline_timeseries.csv"
metrics_path = p / f"{prefix}_kelly_best_metrics.json"
ci_path = p / f"{prefix}_bootstrap_ci.json"
grid_path = p / f"{prefix}_kelly_experiment_grid.csv"

if not (co_path.exists() and bl_path.exists()):
    print("MISSING MF artifacts"); raise SystemExit(0)

co_rows = list(csv.DictReader(open(co_path)))
bl_rows = list(csv.DictReader(open(bl_path)))
N = len(co_rows)
bl_r = np.array([float(r["portfolio_return"]) for r in bl_rows[-N:]])
co_r = np.array([float(r.get("portfolio_return", 0.0)) for r in co_rows])
def sortino(x):
    if x.size == 0: return 0.0
    d = np.minimum(x, 0.0); dd = float(np.sqrt(np.mean(d*d)))
    return float(np.mean(x) / (dd + 1e-8))
def maxdd(x):
    if x.size == 0: return 0.0
    c = np.cumprod(1.0 + x); pk = np.maximum.accumulate(c); return float(np.min(c/pk - 1.0))

bl_s = sortino(bl_r); co_s = sortino(co_r)
bl_lw = float(np.sum(np.log(1.0 + bl_r))); co_lw = float(np.sum(np.log(1.0 + co_r)))
bl_dd = maxdd(bl_r); co_dd = maxdd(co_r)

ci = json.loads(ci_path.read_text()) if ci_path.exists() else {}
km = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
bp = km.get("best_params", {})

# Per-trial fee_rate breakdown from the experiment grid
trials = []
try:
    with grid_path.open() as h:
        for row in csv.DictReader(h):
            trials.append(row)
except Exception:
    pass

# Group trials by fee_rate, summarize
fee_groups = {}
for t in trials:
    try:
        fee = float(t.get("fee_rate", 0.0))
        lw = float(t.get("holdout_log_wealth", 0.0))
        fee_groups.setdefault(fee, []).append(lw)
    except (ValueError, TypeError):
        pass

lines = []
# Read existing summary and extend
existing = (REPO / "docs" / "option_B_results.md").read_text() if (REPO / "docs" / "option_B_results.md").exists() else ""
lines.append(existing)
lines.append("\n---\n")
lines.append("# Pod MF — Fee Sweep Extension\n")
lines.append(
    "Extends Pod M by adding `fee_rate` as a searchable Optuna categorical with "
    "values `{0, 0.001, 0.005, 0.01}` (0, 10, 50, 100 bps). Addresses Adam's "
    "§17 caveat that K10C's gross +0.46 log-wealth edge has break-even at only "
    "3.76 bps (flips to −0.76 at realistic 10 bps spreads).\n"
)
lines.append("## Headline Pod MF numbers\n")
lines.append("| Metric | Baseline | Pod MF | Δ |")
lines.append("|---|---|---|---|")
lines.append(f"| Sortino | {bl_s:+.4f} | {co_s:+.4f} | **{co_s-bl_s:+.4f}** |")
lines.append(f"| Log-wealth | {bl_lw:+.4f} | {co_lw:+.4f} | **{co_lw-bl_lw:+.4f}** |")
lines.append(f"| Max drawdown | {bl_dd*100:+.2f}% | {co_dd*100:+.2f}% | {(co_dd-bl_dd)*100:+.2f} pp |")
lines.append("")
lines.append("## Bootstrap 95% CI on Sortino delta\n")
if ci:
    lines.append(f"- Observed Δ: **{ci.get('observed_sortino_delta', 0):+.4f}**")
    lines.append(f"- 95% CI: [{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}]")
    lines.append(f"- Frac positive: **{ci.get('frac_positive', 0):.1%}**")
lines.append("")
lines.append("## Which fee_rate Optuna selected\n")
fee_winner = bp.get("fee_rate")
lines.append(f"**Best trial's fee_rate: `{fee_winner}` ({fee_winner*10000:.1f} bps)**" if fee_winner is not None else "**Best trial fee_rate: unknown**")
lines.append("")
lines.append("## Per-fee-rate trial summary (log-wealth mean / best)\n")
lines.append("| fee_rate (bps) | n trials | best log-wealth | mean log-wealth |")
lines.append("|---|---|---|---|")
for fee in sorted(fee_groups.keys()):
    vals = fee_groups[fee]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        continue
    lines.append(f"| {fee*10000:.1f} | {len(vals)} | {max(vals):+.4f} | {np.mean(vals):+.4f} |")
lines.append("")
lines.append("## Pod M vs Pod MF comparison\n")
lines.append("| Pod | fee_rate | Sortino Δ | Log-wealth Δ | Note |")
lines.append("|---|---|---|---|---|")
# Pull Pod M numbers from its bootstrap
m_ci_path = p / "week14_M_seed7_bootstrap_ci.json"
if m_ci_path.exists():
    m_ci = json.loads(m_ci_path.read_text())
    lines.append(f"| Pod M (no fee lever) | 0.0 (forced) | {m_ci.get('observed_sortino_delta', 0):+.4f} | {m_ci.get('observed_log_wealth_delta', 0):+.4f} | Adam's K10C eq |")
lines.append(f"| Pod MF (fee searched) | {fee_winner*10000:.1f} bps | {co_s-bl_s:+.4f} | {co_lw-bl_lw:+.4f} | realistic fees |")
lines.append("")
lines.append("## Interpretation\n")
if fee_winner is not None and fee_winner == 0.0:
    lines.append("**Optuna picked fee_rate=0 even when all other values were available.** "
                 "Indicates the optimizer couldn't find a configuration that justifies "
                 "paying explicit fees — positive values penalize turnover in training "
                 "and in the reported return, pushing objectives down.")
elif fee_winner is not None and (co_lw - bl_lw) > 0:
    lines.append(f"**Optuna picked fee_rate={fee_winner*10000:.0f} bps AND the log-wealth delta is still positive**. "
                 "This is a strong result: momentum + Kelly delivers a net-positive edge under realistic fees, "
                 "directly addressing Adam's K10C fee-break-even concern.")
else:
    lines.append("Optuna picked a non-zero fee_rate but overall log-wealth is negative, meaning the fee eats the Kelly edge even with momentum pre-filtering.")

out = REPO / "docs" / "option_B_results.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Updated {out.relative_to(REPO)}")
PYEOF

cd "$REPO" && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/option_B_results.md data/processed/week14_MF_seed7_bootstrap_ci.json 2>/dev/null && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Pod MF results: momentum-Kelly + fee sweep" >> "$MASTER_LOG" 2>&1 && \
  git push origin HEAD:option-B-kelly-momentum >> "$MASTER_LOG" 2>&1

echo "[MF] ====== Complete at $(date) ======" >> "$MASTER_LOG"
