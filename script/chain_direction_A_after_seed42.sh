#!/usr/bin/env bash
# Overnight orchestrator for Direction A (learnable market inclusion).
#
# Sequence:
#   1. Wait for Pod G seed=42 to finish (process PID passed as arg)
#   2. Compute bootstrap CI on Pod G seed=42 result
#   3. Smoke-test Direction A with 5 trials (--artifact-prefix week9_L_smoke)
#   4. If smoke passes, launch full Direction A run (100 trials, --artifact-prefix week9_L)
#   5. Compute bootstrap CI on Direction A result
#   6. Write comparison summary doc (docs/direction_A_results.md)
#   7. Commit + push the summary to origin/learnable-selection
#
# Usage: bash script/chain_direction_A_after_seed42.sh <POD_G_SEED42_PID>

set -u

POD_PID="${1:-96624}"
REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

LOG_SMOKE="$REPO/logs/pod_L_smoke.log"
LOG_L="$REPO/logs/pod_L.log"
CHAIN_LOG="$REPO/logs/chain_direction_A.log"
exec >> "$CHAIN_LOG" 2>&1

echo "[chain] ====== Direction A orchestrator starting at $(date) ======"
echo "[chain] Waiting for Pod G seed=42 (PID $POD_PID) to finish..."
while kill -0 "$POD_PID" 2>/dev/null; do
  sleep 60
done
echo "[chain] Pod G seed=42 has exited at $(date)."

# ----- Step A: Bootstrap CI for seed=42 -----
if [[ -s "$REPO/data/processed/week9_G_seed42_constrained_best_metrics.json" ]]; then
  echo "[chain] Computing bootstrap CI for Pod G seed=42..."
  cd "$REPO" && /Users/colinyu/miniconda3/bin/python - <<'PYEOF'
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co = list(csv.DictReader(open(p / "week9_G_seed42_constrained_best_timeseries.csv")))
bl = list(csv.DictReader(open(p / "week9_G_seed42_baseline_timeseries.csv")))
N = len(co)
bl_r = np.array([float(r["portfolio_return"]) for r in bl[-N:]])
co_r = np.array([float(r["portfolio_return"]) for r in co])
def sortino(x):
    if x.size == 0: return 0.0
    d = np.minimum(x, 0.0); dd = float(np.sqrt(np.mean(d*d)))
    return float(np.mean(x) / (dd + 1e-8))
rng = np.random.default_rng(42); B = 5000
deltas = np.empty(B)
idx = np.arange(N)
for i in range(B):
    s = rng.choice(idx, size=N, replace=True)
    deltas[i] = sortino(co_r[s]) - sortino(bl_r[s])
out = {"observed_delta": float(sortino(co_r)-sortino(bl_r)),
       "mean_delta": float(np.mean(deltas)),
       "ci_lower": float(np.percentile(deltas, 2.5)),
       "ci_upper": float(np.percentile(deltas, 97.5)),
       "frac_positive": float(np.mean(deltas > 0)),
       "bl_sortino_holdout": sortino(bl_r),
       "co_sortino_holdout": sortino(co_r),
       "n_bootstrap": B, "n_holdout_steps": N}
Path("data/processed/week9_G_seed42_bootstrap_sortino_ci.json").write_text(json.dumps(out, indent=2))
print(f"[seed42 bootstrap] delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF
else
  echo "[chain] WARNING: Pod G seed=42 produced no metrics file. Continuing anyway."
fi

# ----- Step B: Direction A smoke test -----
echo "[chain] ====== Direction A smoke test starting at $(date) ======"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

cd "$REPO" && /Users/colinyu/miniconda3/bin/python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_L_smoke \
  --macro-integration rescale \
  --reduced-search \
  --top-k-bagging 5 \
  --learnable-inclusion \
  --inclusion-target-k 15 \
  --optuna-n-jobs 4 \
  --optuna-trials 5 \
  > "$LOG_SMOKE" 2>&1

SMOKE_RC=$?
echo "[chain] Smoke test exited with rc=$SMOKE_RC at $(date)"

if [[ ! -s "$REPO/data/processed/week9_L_smoke_constrained_best_metrics.json" ]]; then
  echo "[chain] FATAL: Direction A smoke test produced no metrics file. Aborting full run."
  echo "[chain] Last 30 lines of smoke log:"
  tail -30 "$LOG_SMOKE"
  exit 3
fi
echo "[chain] Smoke test passed — metrics file exists. Proceeding to full run."

# ----- Step C: Direction A full 100-trial run -----
echo "[chain] ====== Direction A full run starting at $(date) ======"
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

cd "$REPO" && /Users/colinyu/miniconda3/bin/python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_L \
  --macro-integration rescale \
  --reduced-search \
  --top-k-bagging 5 \
  --learnable-inclusion \
  --inclusion-target-k 15 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-L \
  --git-commit-message "L: rescale + reduced-search + top-K=5 + learnable-inclusion (target-k=15) (no ETF) ${RUN_TAG}" \
  > "$LOG_L" 2>&1

FULL_RC=$?
echo "[chain] Direction A full run exited with rc=$FULL_RC at $(date)"

if [[ ! -s "$REPO/data/processed/week9_L_constrained_best_metrics.json" ]]; then
  echo "[chain] FATAL: Direction A full run produced no metrics file. Not writing summary."
  exit 4
fi

# ----- Step D: Bootstrap CI on full run -----
echo "[chain] Computing bootstrap CI for Direction A..."
cd "$REPO" && /Users/colinyu/miniconda3/bin/python - <<'PYEOF'
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co = list(csv.DictReader(open(p / "week9_L_constrained_best_timeseries.csv")))
bl = list(csv.DictReader(open(p / "week9_L_baseline_timeseries.csv")))
N = len(co)
bl_r = np.array([float(r["portfolio_return"]) for r in bl[-N:]])
co_r = np.array([float(r["portfolio_return"]) for r in co])
def sortino(x):
    if x.size == 0: return 0.0
    d = np.minimum(x, 0.0); dd = float(np.sqrt(np.mean(d*d)))
    return float(np.mean(x) / (dd + 1e-8))
rng = np.random.default_rng(42); B = 5000
deltas = np.empty(B)
idx = np.arange(N)
for i in range(B):
    s = rng.choice(idx, size=N, replace=True)
    deltas[i] = sortino(co_r[s]) - sortino(bl_r[s])
out = {"observed_delta": float(sortino(co_r)-sortino(bl_r)),
       "mean_delta": float(np.mean(deltas)),
       "ci_lower": float(np.percentile(deltas, 2.5)),
       "ci_upper": float(np.percentile(deltas, 97.5)),
       "frac_positive": float(np.mean(deltas > 0)),
       "bl_sortino_holdout": sortino(bl_r),
       "co_sortino_holdout": sortino(co_r),
       "n_bootstrap": B, "n_holdout_steps": N}
Path("data/processed/week9_L_bootstrap_sortino_ci.json").write_text(json.dumps(out, indent=2))
print(f"[L bootstrap] delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF

# ----- Step E: Comparison writeup -----
echo "[chain] Writing comparison doc..."
cd "$REPO" && /Users/colinyu/miniconda3/bin/python script/write_direction_A_summary.py

# ----- Step F: Commit and push summary -----
cd "$REPO" && git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/direction_A_results.md && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Direction A results: seed=42 replication + learnable-inclusion run summary" && \
  git push origin HEAD:learnable-selection
echo "[chain] ====== Orchestrator complete at $(date) ======"
