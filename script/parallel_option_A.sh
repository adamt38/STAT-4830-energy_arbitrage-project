#!/usr/bin/env bash
# Option A: Momentum × Baseline-Shrinkage × macro=both, 3-seed robustness.
#
# Novel combination Adam's Round-2/4/5 matrices never test together:
#   - Pod B2 tested shrinkage alone (macro=rescale, Δ = 0.000 — α collapsed to 1)
#   - Pod G/I4 tested momentum alone (macro=rescale/both, Δ ∈ {+0.034, +0.008})
#   - NOTHING tests momentum × shrinkage × macro=both jointly
#
# All three seeds run IN PARALLEL for maximum M3 Max throughput.
#
# Per process: OMP=4 MKL=4 TORCH=4, --optuna-n-jobs 4 (Adam's 16-vCPU recipe).
# Three processes × 16 thread assignments = 48 over 16 physical cores = ~3x
# oversubscription at peak. macOS scheduler handles this for GIL-bound Python
# workloads; each process has its own GIL and is typically only using ~2-3 cores
# of real compute at a time (solo runs showed 240% CPU).
#
# Usage: bash script/parallel_option_A.sh

set -u

REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/option_A_master.log"
exec >> "$MASTER_LOG" 2>&1

echo "[A-parallel] ====== Option A (momentum × shrinkage × macro=both, 3 seeds in parallel) starting at $(date) ======"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

launch_seed() {
  local seed="$1"
  local prefix="week13_A_seed${seed}"
  local push_branch="cloud-runs-A-seed${seed}"
  local log="$LOG_DIR/pod_A_seed${seed}.log"
  local RUN_TAG
  RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
  echo "[A-parallel] Launching seed=${seed} at $(date), log: $log"
  nohup /Users/colinyu/miniconda3/bin/python -u "$REPO/script/polymarket_week8_pipeline.py" \
    --artifact-prefix "$prefix" \
    --macro-modes both \
    --reduced-search \
    --momentum-screening --momentum-top-n 20 --momentum-lookback-days 5 \
    --baseline-shrinkage \
    --top-k-bagging 5 \
    --optuna-n-jobs 4 \
    --optuna-trials 100 \
    --seed-override "$seed" \
    --git-commit-and-push \
    --git-push-branch "$push_branch" \
    --git-commit-message "A-seed${seed}: momentum top-20/5d + baseline-shrinkage + macro=both (novel combo, noise-floor test) ${RUN_TAG}" \
    > "$log" 2>&1 &
  local pid=$!
  echo "[A-parallel] seed=${seed} PID=$pid"
  echo "$pid"
}

PID7=$(launch_seed 7)
# Small stagger so data-fetch requests to Polymarket API don't stampede at the exact same instant
sleep 10
PID42=$(launch_seed 42)
sleep 10
PID123=$(launch_seed 123)

echo "[A-parallel] All 3 seeds launched. PIDs: seed7=$PID7, seed42=$PID42, seed123=$PID123"

# Wait for all 3 to finish
for pid in $PID7 $PID42 $PID123; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  echo "[A-parallel] PID $pid exited at $(date)"
done

echo "[A-parallel] All 3 seeds finished at $(date). Computing bootstrap CIs..."

# Compute bootstrap CIs for each seed
for seed in 7 42 123; do
  prefix="week13_A_seed${seed}"
  stem="${prefix}_macro_both"
  /Users/colinyu/miniconda3/bin/python - <<PYEOF
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co_path = p / "${stem}_constrained_best_timeseries.csv"
bl_path = p / "${prefix}_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print("[A-ci] missing files for ${prefix}"); raise SystemExit(0)
co = list(csv.DictReader(open(co_path)))
bl = list(csv.DictReader(open(bl_path)))
N = len(co)
bl_r = np.array([float(r["portfolio_return"]) for r in bl[-N:]])
co_r = np.array([float(r["portfolio_return"]) for r in co])
def sortino(x):
    if x.size == 0: return 0.0
    d = np.minimum(x, 0.0); dd = float(np.sqrt(np.mean(d*d)))
    return float(np.mean(x) / (dd + 1e-8))
rng = np.random.default_rng(42); B = 5000
deltas = np.empty(B); idx = np.arange(N)
for i in range(B):
    s = rng.choice(idx, size=N, replace=True)
    deltas[i] = sortino(co_r[s]) - sortino(bl_r[s])
out = {"observed_delta": float(sortino(co_r)-sortino(bl_r)),
       "ci_lower": float(np.percentile(deltas, 2.5)),
       "ci_upper": float(np.percentile(deltas, 97.5)),
       "frac_positive": float(np.mean(deltas > 0)),
       "bl_sortino_holdout": sortino(bl_r),
       "co_sortino_holdout": sortino(co_r),
       "n_bootstrap": B, "n_holdout_steps": N}
(p / "${prefix}_bootstrap_sortino_ci.json").write_text(json.dumps(out, indent=2))
print(f"[A-seed${seed}] delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF
done

# Summary writeup
echo "[A-parallel] Writing Option A summary doc..."
/Users/colinyu/miniconda3/bin/python "$REPO/script/write_option_A_summary.py"

# Commit and push the summary to direction-B
cd "$REPO" && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/option_A_results.md data/processed/week13_A_*_bootstrap_sortino_ci.json 2>/dev/null && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Option A results: momentum × shrinkage × macro=both, 3-seed robustness" 2>&1 | tail -3 && \
  git push origin HEAD:direction-B

echo "[A-parallel] ====== Option A complete at $(date) ======"
