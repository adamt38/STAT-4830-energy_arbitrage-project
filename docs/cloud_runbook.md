# Cloud runbook — Polymarket pipeline (parallel-safe, Round 2)

End-to-end guide for running every experiment **on its own pod, in parallel, with zero merge conflicts**. Each pod writes to a unique `--artifact-prefix` and pushes to its own `--git-push-branch`; results are merged back into `cloud-runs` on the laptop at the end.

**Round 2** (current section 4) replaces the Round 1 experiment matrix with a six-pod ablation that targets the four highest-leverage levers from the [Week 9 cross-pod synthesis](week9_cross_pod_synthesis.md): drop ETF tracking everywhere, narrow the Optuna search space, average top-K trial holdouts, and (for some pods) optimize against the baseline-relative Sortino delta or shrink toward equal-weight at evaluation time. Round 1 commands are preserved in [section 12](#12-archive--round-1-experiment-matrix).

This guide also folds in the speed fixes (hourly bars, smaller rolling-window set, fewer inner steps, more aggressive Optuna pruning), so a single 100-trial run is roughly an order of magnitude faster than the original config.

---

## 0. One-time prerequisites

- Code lives on the `cloud-runs` branch of `https://github.com/adamt38/STAT-4830-energy_arbitrage-project`. Every pod clones from it.
- `cloud-runs` already contains the speed knobs, the `--artifact-prefix` flag, and the Round 2 levers (`--top-k-bagging`, `--baseline-shrinkage`, `--beat-baseline-objective`, `--reduced-search`) — no edits needed.

---

## 1. SSH to a fresh CPU pod

```bash
prime pods ssh <pod-id>
```

(Use whatever SSH command Prime Intellect prints if you aren't using the CLI.)

Repeat once per experiment you plan to run in parallel — one pod per experiment. The recommended set is **6 pods, one for each of A–F**.

---

## 2. Pod setup (once per pod)

```bash
# 2a. Code on cloud-runs
cd ~
git clone https://github.com/adamt38/STAT-4830-energy_arbitrage-project.git
cd STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git reset --hard origin/cloud-runs

# 2b. OS deps (minimal Ubuntu)
sudo apt update
sudo apt install -y curl build-essential python3-venv tmux

# 2c. Python env (uv + .venv)
export PATH="$HOME/.local/bin:$PATH"
bash script/install.sh
source .venv/bin/activate

# 2d. scipy is required by Optuna QMCSampler but not in requirements.txt
pip install scipy   # or: uv pip install scipy

# 2e. Git identity (only needed if you use --git-commit-and-push)
git config --global user.email "you@example.edu"
git config --global user.name  "Your Name"
```

---

## 3. Open a tmux session and set environment

```bash
cd ~/STAT-4830-energy_arbitrage-project
tmux new -s week9
# (Or `tmux attach -t week8` to reuse a Round 1 session — names are arbitrary.)
```

Inside tmux (assuming a 16-vCPU pod — adjust per [section 3a](#3a-cpu--threading-rules-of-thumb) if not):

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
```

**Important:** these `export` lines must run *inside* tmux, not before `tmux new`. A fresh tmux session does not inherit your interactive shell's exports, and the pipeline locks its threading config at process start, so a misconfigured pod can only be fixed by restarting the run from inside the corrected env.

`Ctrl-b d` to detach; `tmux attach -t week9` to reattach later.

### 3a. CPU / threading rules of thumb

The product `--optuna-n-jobs × OMP_NUM_THREADS` should equal the pod's vCPU count, give or take 1. This avoids both **GIL contention** (n_jobs too high with low BLAS threads — every trial slows down) and **BLAS oversubscription** (OMP too high with high n_jobs — kernels thrash). For Optuna + PyTorch on CPU, a balanced split typically beats either extreme.

| Pod vCPUs (`nproc`) | `OMP_/MKL_/TORCH_NUM_THREADS` | `--optuna-n-jobs` |
|---|---|---|
| 4 | 2 | 2 |
| 8 | 2 | 4 |
| 16 (default in sec. 4) | **4** | **4** |
| 32 | 4 | 8 |
| 64 | 8 | 8 |

`PYTHONUNBUFFERED=1` (combined with `python -u` in section 4) forces unbuffered stdout/stderr so Optuna's `Trial N finished` lines surface in the SSH pane in real time instead of stalling behind `tee`'s buffer.

---

## 4. Round 2 experiments — one per pod, all parallel-safe

Round 2 is a six-pod ablation across the four highest-leverage levers from the [Week 9 cross-pod synthesis](week9_cross_pod_synthesis.md). Every pod **drops `--etf-tracking`** (Round 1's biggest single negative finding) and **adds `--reduced-search`** (denser Sobol coverage of the meaningful subspace). Pods then differ along three axes: macro mode, top-K bagging size, and whether to add `--baseline-shrinkage` and/or `--beat-baseline-objective`.

Every command below sets three things that make parallel execution safe:

- `--artifact-prefix week9_<X>` — every CSV/JSON/PNG/manifest this run writes is stamped with that prefix, so two pods never touch the same filename. **`week9_*` keeps Round 1's `week8_*` artifacts intact for comparison.**
- `--git-push-branch cloud-runs-<X>2` — results are pushed to a Round 2-specific branch (`-A2`, `-B2`, ...), so the Round 1 branches (`cloud-runs-A` ... `cloud-runs-F`) stay untouched.
- A unique combination of Round 2 flags per the matrix below.

`python -u` keeps stdout/stderr unbuffered (paired with `PYTHONUNBUFFERED=1` from section 3) so trial-completion lines surface in the SSH pane immediately. `--optuna-n-jobs 4` is sized for a 16-vCPU pod with `OMP/MKL/TORCH=4`; if your pod has a different `nproc`, adjust both per the table in [section 3a](#3a-cpu--threading-rules-of-thumb).

### 4.0. Round 2 ablation matrix

| Pod | Macro | Top-K bagging | Beat-baseline obj | Baseline shrinkage | Hypothesis |
|---|---|---|---|---|---|
| **A** | rescale | 5 | — | — | Pure top-K bagging on the simplest macro. Isolates the bagging effect. |
| **B** | rescale | 1 | — | yes | Pure shrinkage-α on the simplest macro. Isolates the shrinkage effect. |
| **C** | both | 5 | — | — | Top-K bagging on the macro mode that blends rescale + explicit. |
| **D** | explicit | 5 | yes | — | Bagging + objective change on the explicit-only macro. |
| **E** | both | 10 | yes | yes | Kitchen sink: bigger bag, beat-baseline objective, and shrinkage α together. |
| **F** | joint | 5 | yes | yes | Joint macro search with all three improvements on. |

### Experiment A — Pure top-K bagging on rescale macro

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_A \
  --macro-integration rescale \
  --reduced-search \
  --top-k-bagging 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-A2 \
  --git-commit-message "A2: rescale + reduced-search + top-K=5 (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_A2_${RUN_TAG}.log"
```

### Experiment B — Pure baseline shrinkage on rescale macro

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_B \
  --macro-integration rescale \
  --reduced-search \
  --baseline-shrinkage \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-B2 \
  --git-commit-message "B2: rescale + reduced-search + shrinkage alpha (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_B2_${RUN_TAG}.log"
```

### Experiment C — Top-K bagging on macro=both

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_C \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-C2 \
  --git-commit-message "C2: macro=both + reduced-search + top-K=5 (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_C2_${RUN_TAG}.log"
```

### Experiment D — Bagging + beat-baseline objective on macro=explicit

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_D \
  --macro-integration explicit \
  --reduced-search \
  --top-k-bagging 5 \
  --beat-baseline-objective \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-D2 \
  --git-commit-message "D2: macro=explicit + reduced-search + top-K=5 + beat-baseline obj (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_D2_${RUN_TAG}.log"
```

### Experiment E — Kitchen sink on macro=both

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_E \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 10 \
  --beat-baseline-objective \
  --baseline-shrinkage \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-E2 \
  --git-commit-message "E2: macro=both + reduced-search + top-K=10 + beat-baseline + shrinkage (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_E2_${RUN_TAG}.log"
```

### Experiment F — Joint macro search with all three improvements

One Optuna study with categorical `macro_mode` and conditional `regime_k` / `lambda_macro_explicit`. Do **not** combine with `--macro-modes`.

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week9_F \
  --joint-macro-mode-search \
  --reduced-search \
  --top-k-bagging 5 \
  --beat-baseline-objective \
  --baseline-shrinkage \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-F2 \
  --git-commit-message "F2: joint-macro + reduced-search + top-K=5 + beat-baseline + shrinkage (no ETF) ${RUN_TAG}" \
  2>&1 | tee "run_F2_${RUN_TAG}.log"
```

---

## 5. What each Round 2 experiment writes

Every run writes to `data/processed/`, `figures/`, and the manifest, all stamped with `week9_<X>`. There is **no overlap** across experiments, and **no overlap with Round 1's `week8_*` artifacts**.

| Pod | Constrained Optuna stems | Baseline / exog / covariance / figures stem | Manifest | Push branch |
|---|---|---|---|---|
| A (rescale, top-K=5) | `week9_A_constrained_*` | `week9_A_*` | `week9_A_run_manifest.json` | `cloud-runs-A2` |
| B (rescale, shrinkage) | `week9_B_constrained_*` | `week9_B_*` | `week9_B_run_manifest.json` | `cloud-runs-B2` |
| C (both, top-K=5) | `week9_C_macro_both_constrained_*` | `week9_C_*` | `week9_C_run_manifest.json` | `cloud-runs-C2` |
| D (explicit, top-K=5 + beat-baseline) | `week9_D_macro_explicit_constrained_*` | `week9_D_*` | `week9_D_run_manifest.json` | `cloud-runs-D2` |
| E (both, kitchen sink) | `week9_E_macro_both_constrained_*` | `week9_E_*` | `week9_E_run_manifest.json` | `cloud-runs-E2` |
| F (joint, kitchen sink) | `week9_F_constrained_*` (joint study tracked via trial params) | `week9_F_*` | `week9_F_run_manifest.json` | `cloud-runs-F2` |

The constrained best-metrics JSON (`*_constrained_best_metrics.json`) now also contains `holdout_sortino_minus_baseline`, `baseline_holdout_sortino`, `top_k_bagging`, `baseline_shrinkage`, `beat_baseline_objective`, `bagged_trial_numbers`, `bagged_alphas`, and `best_trial_baseline_alpha`, so cross-pod comparison is a single field-read.

The manifest carries `git_commit_and_push_requested`, the macro / ETF / Round 2 flags you passed, and the artifact prefix, so you can always reconstruct what was run.

---

## 6. How to confirm a run finished and pushed

In the tmux output / `run_<X>2_${RUN_TAG}.log` you should see, in order:

```
PIPELINE COMPLETE
  Total time: ...
- run_manifest: data/processed/week9_<X>_run_manifest.json
PIPELINE STAGE: Git commit and push
Git publish: committed and pushed branch 'cloud-runs' to origin/cloud-runs-<X>2.
```

(Or `Git publish: nothing new to commit; skipping pull/push.` if you re-ran an identical config.)

If git fails, the script exits non-zero **after** the science stages — the log shows the git error and there is no silent success. On GitHub → `cloud-runs-<X>2` you should see a new commit dated within the last few minutes.

Quick "did we beat the baseline?" peek (run on the pod after the pipeline completes, replacing `<X>` with the pod letter):

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('data/processed/week9_<X>*_constrained_best_metrics.json')):
    d = json.load(open(f))['best_params']
    print(f'{f.split(chr(47))[-1]:60s}  delta={d[\"holdout_sortino_minus_baseline\"]:+.4f}  '
          f'sortino={d[\"holdout_sortino_ratio\"]:.4f}  baseline={d[\"baseline_holdout_sortino\"]:.4f}  '
          f'K={d[\"top_k_bagging\"]}  shrink={d[\"baseline_shrinkage\"]}  beatobj={d[\"beat_baseline_objective\"]}')
"
```

A positive `delta` means this pod beat the baseline.

---

## 7. Detach / reconnect / cleanup

```bash
# Detach so you can close SSH safely:
Ctrl-b d

# Reconnect later:
prime pods ssh <pod-id>
tmux attach -t week9          # or `week8` if you reused a Round 1 session

# End session when done (inside tmux):
exit
# or, from outside tmux:
tmux kill-session -t week9

# Pull extra files to your Mac (optional):
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/run_*2_*.log' .
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/data/processed/week9_*.csv' .
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/data/processed/week9_*.json' .
```

---

## 8. After all pods finish — fan back in to `cloud-runs`

Each Round 2 pod pushed to its own branch (`cloud-runs-A2` … `cloud-runs-F2`). Because every artifact stem is `week9_<X>*` and Round 1 used `week8_*`, **the merges are conflict-free** with each other *and* with the Round 1 artifacts already on `cloud-runs`:

```bash
cd ~/.../STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git pull --ff-only

git merge --no-ff origin/cloud-runs-A2 -m "merge week9_A artifacts"
git merge --no-ff origin/cloud-runs-B2 -m "merge week9_B artifacts"
git merge --no-ff origin/cloud-runs-C2 -m "merge week9_C artifacts"
git merge --no-ff origin/cloud-runs-D2 -m "merge week9_D artifacts"
git merge --no-ff origin/cloud-runs-E2 -m "merge week9_E artifacts"
git merge --no-ff origin/cloud-runs-F2 -m "merge week9_F artifacts"
git push origin cloud-runs
```

Then summarize all six in one shot from your laptop:

```bash
for f in data/processed/week9_*_constrained_best_metrics.json; do
  python3 -c "
import json
d = json.load(open('$f'))['best_params']
print(f'{(\"$f\").split(chr(47))[-1]:60s}  delta={d[\"holdout_sortino_minus_baseline\"]:+.4f}  '
      f'sortino={d[\"holdout_sortino_ratio\"]:.4f}  baseline={d[\"baseline_holdout_sortino\"]:.4f}  '
      f'K={d[\"top_k_bagging\"]}  shrink={d[\"baseline_shrinkage\"]}  beatobj={d[\"beat_baseline_objective\"]}')
"
done
```

If you ever do see a conflict here, it means two experiments shared a prefix — fix the offending pod's `--artifact-prefix` and rerun that one only.

Optional cleanup once the merges are pushed:

```bash
git push origin --delete cloud-runs-A2 cloud-runs-B2 cloud-runs-C2 cloud-runs-D2 cloud-runs-E2 cloud-runs-F2
```

---

## 9. Cheat sheet — single pod, single Round 2 experiment

```bash
prime pods ssh <pod-id>

cd ~
git clone https://github.com/adamt38/STAT-4830-energy_arbitrage-project.git
cd STAT-4830-energy_arbitrage-project
git fetch origin && git checkout cloud-runs && git reset --hard origin/cloud-runs

sudo apt update && sudo apt install -y curl build-essential python3-venv tmux
export PATH="$HOME/.local/bin:$PATH"
bash script/install.sh
source .venv/bin/activate
uv pip install scipy
git config --global user.email "you@example.edu"
git config --global user.name  "Your Name"

tmux new -s week9
# inside tmux (16-vCPU pod; adjust per section 3a if not):
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Pick ONE block from section 4 (the --artifact-prefix and --git-push-branch
# values must be unique per pod). Paste it and let the run finish.
```

For a pod that already has Round 1 set up, the cheat sheet collapses to **only** the steps inside tmux — pull `cloud-runs`, re-export the env, paste the section 4 block:

```bash
prime pods ssh <pod-id>
tmux attach -t week8                  # or `tmux new -s week9`
cd ~/STAT-4830-energy_arbitrage-project
git fetch origin && git reset --hard origin/cloud-runs
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
# paste this pod's section 4 block.
```

That's the whole loop. Six pods, six commands from section 4, one merge sequence in section 8 → all six Round 2 experiments land cleanly on `cloud-runs` alongside the Round 1 artifacts.

---

## 10. Recovery — pod was launched with stale code or wrong env

Symptoms of a bad env: `top` shows python at ~300% CPU on a 16-core pod (only ~3 cores busy), or `run_*.log` stalls at the Optuna start banner with no `Trial N finished` lines for 30+ minutes. Symptoms of stale code: `error: unrecognized arguments: --top-k-bagging` or `--reduced-search` on launch.

The fix is to kill the bad process and restart with the new code + env + `python -u`. Don't try to `export` new env vars onto the running process — env vars are inherited at process start, so the running python is locked to whatever was set when it was launched.

```bash
prime pods ssh <pod-id>
tmux attach -t week9            # or week8 if that's what's running
# Ctrl-c to interrupt the python process; wait for the prompt to come back.
# Optionally, hard-kill anything python that's still around:
pkill -9 -f polymarket_week8_pipeline || true
exit                              # leaves tmux

# Pull the updated runbook + Round 2 levers from cloud-runs:
cd ~/STAT-4830-energy_arbitrage-project
git fetch origin
git reset --hard origin/cloud-runs

# Fresh tmux, fresh env, fresh RUN_TAG so the old log stays as evidence:
tmux new -s week9
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Paste the same section-4 block you used the first time (same --artifact-prefix
# and --git-push-branch — re-using them is safe; the pod's local data/processed
# files just get overwritten).
```

Within ~5–10 min you should see `top` show python at ~1500% CPU and `run_<X>2_*.log` produce trial-completion lines.

---

## 11. Notes on what changed under the hood

- **Round 2 levers (new):** `--top-k-bagging K` averages holdout returns/weights of the top-K Optuna trials (reduces selection bias from 100-trial Sobol search). `--baseline-shrinkage` adds `baseline_alpha ∈ [0,1]` as a search dimension; final returns are `α·r_constrained + (1−α)·r_equalweight` in both fold scoring and holdout. `--beat-baseline-objective` switches the Optuna objective from `Sortino_constrained_CV` to `Sortino_constrained_CV − Sortino_equalweight_CV`. `--reduced-search` narrows `learning_rates`, `penalties_lambda`, `domain_limits`, `max_weights`, `variance/downside_penalties`, `uniform_mixes`; pins `entropy_lambdas=(0.0,)`. All four default off, so legacy invocations behave exactly as before.
- **Best-metrics JSON (new fields):** `holdout_sortino_minus_baseline`, `baseline_holdout_sortino`, `top_k_bagging`, `baseline_shrinkage`, `beat_baseline_objective`, `bagged_trial_numbers`, `bagged_alphas`, `best_trial_baseline_alpha`. The first field alone tells you whether a pod beat the baseline.
- **Speed:** `history_fidelity` is 60 (1-hour bars), `rolling_windows=(24, 48, 96)`, `steps_per_window=3`, and `MedianPruner` starts pruning after only 3 startup trials. Together these make a 100-trial study finish in roughly 1/10th the time of the original config.
- **Per-prefix isolation:** The `--artifact-prefix` flag is threaded through every stage of `polymarket_week8_pipeline.py` — data ingestion, baseline, exogenous features, Optuna search, covariance diagnostics, figures, week-9 report, and the run manifest. Default is still `week8`, so single-pod runs without the flag behave exactly as before.
- **Figures correctness:** `_make_figures` reads the constrained outputs at `<prefix><suffix>_constrained_*`, so experiments using `--macro-integration explicit` / `both` generate the comparison figures correctly instead of falling back to the wrong files.
- **Threading:** Default env exports are `OMP/MKL/TORCH=4` + `--optuna-n-jobs 4` (sized for a 16-vCPU pod), with `PYTHONUNBUFFERED=1` and `python -u`. The old `OMP=1 + n_jobs=12/16` combination starved per-trial throughput by oversubscribing the GIL while under-using the BLAS pool; the new product (`n_jobs × OMP ≈ nproc`) keeps every core busy without thread thrashing.

---

## 12. Archive — Round 1 experiment matrix

Round 1 ran six pods (`week8_A` … `week8_F`) with `--etf-tracking` on most of them and no top-K / shrinkage / beat-baseline / reduced-search levers. Cross-pod synthesis: [`docs/week9_cross_pod_synthesis.md`](week9_cross_pod_synthesis.md). Headline finding: every pod underperformed the equal-weight baseline (Sortino Δ ∈ [−0.060, −0.020]); the four levers in section 4 are direct responses to that finding.

If you ever need to reproduce Round 1 verbatim:

| Pod | Command (omit Round 2 flags) |
|---|---|
| A | `--artifact-prefix week8_A --macro-integration rescale --git-push-branch cloud-runs-A` |
| B | `--artifact-prefix week8_B --macro-integration rescale --etf-tracking --git-push-branch cloud-runs-B` |
| C | `--artifact-prefix week8_C --macro-integration explicit --etf-tracking --git-push-branch cloud-runs-C` |
| D | `--artifact-prefix week8_D --macro-integration both --etf-tracking --git-push-branch cloud-runs-D` |
| E | `--artifact-prefix week8_E --macro-modes rescale,explicit,both --etf-tracking --git-push-branch cloud-runs-E` |
| F | `--artifact-prefix week8_F --joint-macro-mode-search --etf-tracking --git-push-branch cloud-runs-F` |

Add the standard wrappers (`python -u script/polymarket_week8_pipeline.py … --optuna-n-jobs 4 --optuna-trials 100 --git-commit-and-push --git-commit-message "..." 2>&1 | tee run_<X>_${RUN_TAG}.log`) and run inside tmux as in sections 3 and 4.
