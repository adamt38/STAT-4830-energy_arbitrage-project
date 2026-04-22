#!/usr/bin/env bash
# Option B Pod M: Momentum × Kelly pipeline, 2 parallel seeds.
#
# Contributes the one combination Adam's Round 3-7 Kelly experiments never test:
# momentum-screened universe. K10A/B/C/D/E/F all run on the full 40-market universe
# from cached week8 data. Pod M builds fresh data with momentum top-20 / 5d
# screening, then runs the same Kelly OGD + dynamic-copula optimizer on that
# 20-market subset.
#
# Kelly is more compute-intensive than MVO (~5 min per trial amortized with
# n_jobs=4 vs ~2.5 min for MVO), so running 3 parallel seeds would thrash.
# 2 parallel seeds at ~800% CPU each matches the M3 Max 16-core capacity.
#
# Usage: bash script/parallel_option_B.sh
set -u

REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/option_B_master.log"
echo "[B-parallel] ====== Option B (momentum-Kelly, 2 parallel seeds) starting at $(date) ======" >> "$MASTER_LOG"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Seed 7 (not passing --seed-override since Kelly pipeline doesn't have that flag;
# it uses KellyExperimentConfig.seed set to 7 by default, which is fine for one seed.
# For seed 42 we'd need to either modify the Kelly pipeline or fork its config —
# for simplicity we just run one seed first and add more later if the result is
# promising.)
echo "[B-parallel] Launching seed=7 (default) at $(date)" >> "$MASTER_LOG"
nohup python -u "$REPO/script/polymarket_week10_kelly_pipeline.py" \
  --artifact-prefix week14_M_seed7 \
  --rebuild-data \
  --momentum-screening \
  --momentum-top-n 20 \
  --momentum-lookback-days 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --reduced-search \
  --git-commit-and-push \
  --git-push-branch cloud-runs-M-seed7 \
  --git-commit-message "M-seed7: momentum-Kelly (top-20 / 5d) (Option B)  ${RUN_TAG}" \
  > "$LOG_DIR/pod_M_seed7.log" 2>&1 &
PID7=$!
echo "[B-parallel] seed=7 PID=$PID7" >> "$MASTER_LOG"

# Stagger the second seed launch by 20 seconds so data-fetch requests don't race
sleep 20

# For seed 42 we have a problem: Kelly pipeline's KellyExperimentConfig has seed=7
# hardcoded. To run a different seed we'd need to add a --seed-override flag.
# For now, run only the default seed (7). Multi-seed can be done as a follow-up
# after we confirm the Kelly+momentum combination produces something interesting.
#
# Replaced with a noop — single seed only for Option B.

echo "[B-parallel] Waiting for seed=7 (PID $PID7) to finish..." >> "$MASTER_LOG"
while kill -0 "$PID7" 2>/dev/null; do
  sleep 120
done
echo "[B-parallel] seed=7 exited at $(date)" >> "$MASTER_LOG"

# Compute bootstrap CI on the Kelly log-wealth delta (Kelly's native metric)
# AND on Sortino delta for direct comparison to MVO results
echo "[B-parallel] Computing bootstrap CIs..." >> "$MASTER_LOG"

python - <<'PYEOF' >> "$MASTER_LOG" 2>&1
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
prefix = "week14_M_seed7"
co_path = p / f"{prefix}_kelly_best_timeseries.csv"
bl_path = p / f"{prefix}_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print(f"[M-ci] MISSING timeseries for {prefix}")
    raise SystemExit(0)
co_rows = list(csv.DictReader(open(co_path)))
bl_rows = list(csv.DictReader(open(bl_path)))
N = len(co_rows)
bl_r = np.array([float(r["portfolio_return"]) for r in bl_rows[-N:]])
# Kelly CSV columns: timestamp, portfolio_return (net), log_wealth (cumulative)
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
obs_delta = sortino(co_r) - sortino(bl_r)

# Log-wealth delta (Kelly's native metric)
# cumulative log-wealth = cumsum(log(1 + r))
bl_lw = float(np.sum(np.log(1.0 + bl_r)))
co_lw = float(np.sum(np.log(1.0 + co_r)))
lw_delta = co_lw - bl_lw

out = {"observed_sortino_delta": obs_delta,
       "ci_lower": float(np.percentile(deltas, 2.5)),
       "ci_upper": float(np.percentile(deltas, 97.5)),
       "frac_positive": float(np.mean(deltas > 0)),
       "bl_sortino": sortino(bl_r),
       "co_sortino": sortino(co_r),
       "observed_log_wealth_delta": lw_delta,
       "bl_final_log_wealth": bl_lw,
       "co_final_log_wealth": co_lw,
       "n_bootstrap": B, "n_holdout_steps": N}
(p / f"{prefix}_bootstrap_ci.json").write_text(json.dumps(out, indent=2))
print(f"[M-seed7] Sortino Δ={obs_delta:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
print(f"[M-seed7] Log-wealth Δ={lw_delta:+.4f}")
PYEOF

echo "[B-parallel] Writing summary doc..." >> "$MASTER_LOG"
python "$REPO/script/write_option_B_summary.py" >> "$MASTER_LOG" 2>&1

cd "$REPO" && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/option_B_results.md data/processed/week14_M_seed7_bootstrap_ci.json 2>/dev/null && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Option B results: momentum-Kelly Pod M seed 7" >> "$MASTER_LOG" 2>&1 && \
  git push origin HEAD:direction-B >> "$MASTER_LOG" 2>&1

echo "[B-parallel] ====== Complete at $(date) ======" >> "$MASTER_LOG"
