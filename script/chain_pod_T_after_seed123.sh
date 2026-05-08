#!/usr/bin/env bash
# Experiment Pod T: teammate's week17 winning config replicated with momentum screening
# and 3-seed robustness check. Designed to directly target the teammate's +0.1018 Sortino
# edge, while addressing the ~0.036 seed-noise floor Adam documented in §15.9.
#
# Runs sequentially after the currently-running Pod G seed=123 finishes. Each variant
# auto-pushes to its own cloud-runs-T-seed<N> branch.
#
# Usage: bash script/chain_pod_T_after_seed123.sh <POD_G_SEED123_PID>

set -u

POD_PID="${1:-76683}"
REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

CHAIN_LOG="$REPO/logs/chain_pod_T.log"
exec >> "$CHAIN_LOG" 2>&1

echo "[T-chain] ====== Pod T orchestrator starting at $(date) ======"
echo "[T-chain] Waiting for Pod G seed=123 (PID $POD_PID) to finish..."
while kill -0 "$POD_PID" 2>/dev/null; do
  sleep 60
done
echo "[T-chain] Pod G seed=123 has exited at $(date)."

# Compute seed=123 bootstrap CI
echo "[T-chain] Computing bootstrap CI for Pod G seed=123..."
/Users/colinyu/miniconda3/bin/python - <<'PYEOF'
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co_path = p / "week9_G_seed123_constrained_best_timeseries.csv"
bl_path = p / "week9_G_seed123_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print("[seed123-ci] missing files")
    raise SystemExit(0)
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
(p / "week9_G_seed123_bootstrap_sortino_ci.json").write_text(json.dumps(out, indent=2))
print(f"[seed123-ci] delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

# --- Pod T configuration (teammate's week17 winners centered + momentum) ---
# Search (tight bracket around teammate's values):
#   max_weight = 0.04        (pinned — teammate's sweet spot)
#   domain_limit = 0.08      (pinned)
#   uniform_mix = 0.0        (implicit: --reduced-search default already 0)
#   lr ∈ {0.005, 0.01}       (teammate's pick + one wider)
#   rolling_window ∈ {96, 288}  (cover both I4 and teammate winners)
#   variance_penalty ∈ {0.25, 0.5, 1.0}
#   downside_penalty ∈ {0.5, 1.0, 2.0}
#   covariance_penalty ∈ {0.25, 0.5, 1.0}
#   covariance_shrinkage ∈ {0.02, 0.05}
#   concentration_penalty ∈ {1.0, 2.0, 3.0}
# Momentum top-20 / 5d (our proven lever).
# Macro mode = both (R4 finding: both > rescale).
# 100 trials, top-K bagging = 5.

run_variant_T() {
  local seed="$1"
  local prefix="week13_T_seed${seed}"
  local push_branch="cloud-runs-T-seed${seed}"
  local log="$REPO/logs/pod_T_seed${seed}.log"
  local RUN_TAG
  RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
  echo ""
  echo "[T-chain] ====== Pod T seed=${seed} starting at $(date) ======"
  /Users/colinyu/miniconda3/bin/python -u "$REPO/script/polymarket_week8_pipeline.py" \
    --artifact-prefix "$prefix" \
    --macro-modes both \
    --reduced-search \
    --momentum-screening --momentum-top-n 20 --momentum-lookback-days 5 \
    --lr-values 0.005,0.01 \
    --rolling-windows 96,288 \
    --variance-penalty-values 0.25,0.5,1.0 \
    --downside-penalty-values 0.5,1.0,2.0 \
    --covariance-penalty-lambdas 0.25,0.5,1.0 \
    --covariance-shrinkage-values 0.02,0.05 \
    --domain-limit-values 0.08 \
    --max-weight-values 0.04 \
    --concentration-penalty-lambdas 1.0,2.0,3.0 \
    --top-k-bagging 5 \
    --optuna-n-jobs 4 \
    --optuna-trials 100 \
    --seed-override "$seed" \
    --git-commit-and-push \
    --git-push-branch "$push_branch" \
    --git-commit-message "T-seed${seed}: teammate config + momentum top-20/5d + macro=both (targets +0.1018) ${RUN_TAG}" \
    > "$log" 2>&1
  local rc=$?
  local metrics_stem
  # constrained outputs for macro=both mode use suffix _macro_both
  metrics_stem="${prefix}_macro_both"
  if [[ ! -s "$REPO/data/processed/${metrics_stem}_constrained_best_metrics.json" ]]; then
    echo "[T-chain] WARNING: Pod T seed=${seed} finished with rc=$rc but no metrics file. Continuing."
    return 1
  fi
  echo "[T-chain] Pod T seed=${seed} succeeded (rc=$rc) at $(date)."

  # Bootstrap CI
  /Users/colinyu/miniconda3/bin/python - <<PYEOF
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co_path = p / "${metrics_stem}_constrained_best_timeseries.csv"
bl_path = p / "${prefix}_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print("[T-ci] missing ${prefix}"); raise SystemExit(0)
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
print(f"[T-seed${2}] delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF
  return 0
}

# --- Run the 3 seeds sequentially ---
run_variant_T 7
run_variant_T 42
run_variant_T 123

# --- Summary writeup ---
echo "[T-chain] Writing Pod T summary..."
/Users/colinyu/miniconda3/bin/python "$REPO/script/write_pod_T_summary.py"

cd "$REPO" && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/pod_T_results.md data/processed/week9_G_seed123_bootstrap_sortino_ci.json data/processed/week13_T_*_bootstrap_sortino_ci.json 2>/dev/null && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Pod T results: teammate config + momentum + 3-seed robustness" 2>&1 | tail -3 && \
  git push origin HEAD:direction-B

echo "[T-chain] ====== Pod T orchestrator complete at $(date) ======"
