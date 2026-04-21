#!/usr/bin/env bash
# Extended overnight queue. Runs AFTER the current Direction A orchestrator (PID passed as arg).
#
# Sequence (each ~1h, sequential, auto-push to own branch):
#   1. Pod G seed=123 -> cloud-runs-G-seed123
#   2. Pod G seed=456 -> cloud-runs-G-seed456
#   3. Pod G seed=789 -> cloud-runs-G-seed789
#   4. Pod L (learnable inclusion) seed=42 -> cloud-runs-L-seed42
#   5. Pod L seed=123 -> cloud-runs-L-seed123
#   6. Hybrid (momentum prefilter + learnable inclusion) seed=7 -> cloud-runs-hybrid
#   7. Regenerate + commit + push direction_A_results.md with full multi-seed ensemble
#
# Usage: bash script/chain_ensemble_after_direction_A.sh <ORCHESTRATOR_PID>

set -u

ORCH_PID="${1:-27922}"
REPO="/Users/colinyu/Documents/STAT-4830-energy_arbitrage-project"
cd "$REPO" || exit 1

CHAIN_LOG="$REPO/logs/chain_ensemble.log"
exec >> "$CHAIN_LOG" 2>&1

echo "[ensemble] ====== Extended queue orchestrator starting at $(date) ======"
echo "[ensemble] Waiting for Direction A orchestrator (PID $ORCH_PID) to finish..."
while kill -0 "$ORCH_PID" 2>/dev/null; do
  sleep 60
done
echo "[ensemble] Direction A orchestrator has exited at $(date)."

# Common threading config
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 PYTHONUNBUFFERED=1

# Helper: run one pipeline invocation, check metrics file exists.
# Args: $1 = log filename, $2..$N = pipeline CLI args
# Returns 0 on success, non-zero on failure (metrics file missing or empty).
run_variant() {
  local log_name="$1"; shift
  local variant_label="$1"; shift
  local prefix_check="$1"; shift
  local metrics="$REPO/data/processed/${prefix_check}_constrained_best_metrics.json"
  echo ""
  echo "[ensemble] ====== $variant_label starting at $(date) ======"
  /Users/colinyu/miniconda3/bin/python -u "$REPO/script/polymarket_week8_pipeline.py" "$@" \
    > "$REPO/logs/$log_name" 2>&1
  local rc=$?
  if [[ ! -s "$metrics" ]]; then
    echo "[ensemble] WARNING: $variant_label finished with rc=$rc but no metrics file at $metrics. Continuing to next variant."
    return 1
  fi
  echo "[ensemble] $variant_label succeeded (rc=$rc) at $(date). Metrics: $metrics"
  return 0
}

# Bootstrap CI helper — pass $1 = artifact_prefix
compute_ci() {
  local prefix="$1"
  /Users/colinyu/miniconda3/bin/python - <<PYEOF
import json, csv, numpy as np
from pathlib import Path
p = Path("data/processed")
co_path = p / "${prefix}_constrained_best_timeseries.csv"
bl_path = p / "${prefix}_baseline_timeseries.csv"
if not (co_path.exists() and bl_path.exists()):
    print("[ci] missing files for ${prefix}")
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
Path(p / "${prefix}_bootstrap_sortino_ci.json").write_text(json.dumps(out, indent=2))
print(f"[ci] ${prefix}: delta={out['observed_delta']:+.4f}, CI=[{out['ci_lower']:+.4f},{out['ci_upper']:+.4f}], frac+={out['frac_positive']:.1%}")
PYEOF
}

# ----- 1: Pod G seed=123 -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_G_seed123.log" "Pod G seed=123" "week9_G_seed123" \
  --artifact-prefix week9_G_seed123 \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --momentum-screening --momentum-top-n 20 --momentum-lookback-days 5.0 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 123 \
  --git-commit-and-push --git-push-branch cloud-runs-G-seed123 \
  --git-commit-message "G seed=123: rescale + reduced-search + K=5 + mom20/5d (no ETF) ${RUN_TAG}"
compute_ci week9_G_seed123

# ----- 2: Pod G seed=456 -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_G_seed456.log" "Pod G seed=456" "week9_G_seed456" \
  --artifact-prefix week9_G_seed456 \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --momentum-screening --momentum-top-n 20 --momentum-lookback-days 5.0 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 456 \
  --git-commit-and-push --git-push-branch cloud-runs-G-seed456 \
  --git-commit-message "G seed=456: rescale + reduced-search + K=5 + mom20/5d (no ETF) ${RUN_TAG}"
compute_ci week9_G_seed456

# ----- 3: Pod G seed=789 -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_G_seed789.log" "Pod G seed=789" "week9_G_seed789" \
  --artifact-prefix week9_G_seed789 \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --momentum-screening --momentum-top-n 20 --momentum-lookback-days 5.0 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 789 \
  --git-commit-and-push --git-push-branch cloud-runs-G-seed789 \
  --git-commit-message "G seed=789: rescale + reduced-search + K=5 + mom20/5d (no ETF) ${RUN_TAG}"
compute_ci week9_G_seed789

# ----- 4: Pod L seed=42 -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_L_seed42.log" "Pod L seed=42 (learnable inclusion)" "week9_L_seed42" \
  --artifact-prefix week9_L_seed42 \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --learnable-inclusion --inclusion-target-k 15 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 42 \
  --git-commit-and-push --git-push-branch cloud-runs-L-seed42 \
  --git-commit-message "L seed=42: rescale + reduced-search + K=5 + learnable-inclusion (target-k=15) (no ETF) ${RUN_TAG}"
compute_ci week9_L_seed42

# ----- 5: Pod L seed=123 -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_L_seed123.log" "Pod L seed=123 (learnable inclusion)" "week9_L_seed123" \
  --artifact-prefix week9_L_seed123 \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --learnable-inclusion --inclusion-target-k 15 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 123 \
  --git-commit-and-push --git-push-branch cloud-runs-L-seed123 \
  --git-commit-message "L seed=123: rescale + reduced-search + K=5 + learnable-inclusion (target-k=15) (no ETF) ${RUN_TAG}"
compute_ci week9_L_seed123

# ----- 6: Hybrid (momentum prefilter top-30 + learnable inclusion target=15) -----
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
run_variant "pod_hybrid.log" "Pod Hybrid (mom30 prefilter + learnable inclusion target=15)" "week9_hybrid" \
  --artifact-prefix week9_hybrid \
  --macro-integration rescale --reduced-search --top-k-bagging 5 \
  --momentum-screening --momentum-top-n 30 --momentum-lookback-days 5.0 \
  --learnable-inclusion --inclusion-target-k 15 \
  --optuna-n-jobs 4 --optuna-trials 100 --seed 7 \
  --git-commit-and-push --git-push-branch cloud-runs-hybrid \
  --git-commit-message "Hybrid: mom30/5d prefilter + learnable-inclusion (target-k=15) + rescale + K=5 ${RUN_TAG}"
compute_ci week9_hybrid

# ----- 7: Regenerate ensemble summary and push -----
echo "[ensemble] ====== Regenerating summary with multi-seed ensemble at $(date) ======"
cd "$REPO" && /Users/colinyu/miniconda3/bin/python script/write_ensemble_summary.py

cd "$REPO" && git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" add docs/direction_A_results.md && \
  git -c user.email="yxk031219@gmail.com" -c user.name="colinyu" commit -m "Multi-seed ensemble results: Pod G x5, Pod L x3, Hybrid — addresses M4 noise finding" && \
  git push origin HEAD:learnable-selection
echo "[ensemble] ====== Extended queue complete at $(date) ======"
