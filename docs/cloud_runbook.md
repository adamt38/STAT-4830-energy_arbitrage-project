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

---

## 13. Round 3 — Week 10 Dynamic-Copula End-to-End Kelly OGD

Round 3 swaps the entire optimization stack: the continuous mean-variance / Sortino objective is replaced by a binary-resolution log-wealth (Kelly) objective, the Pearson covariance is replaced by a dynamic Gaussian copula whose correlation matrix is generated per step by a small MLP over the macro features (SPY / QQQ / BTC log returns), and a vanilla L1 turnover penalty is added to absorb Polymarket slippage. Online weight updates and MLP parameter updates are interleaved in one Adam-OGD pass, which is the source of the non-convexity. See [`docs/week10_kelly_academic_summary.md`](week10_kelly_academic_summary.md) for the math.

This round runs in **its own pipeline script** (`script/polymarket_week10_kelly_pipeline.py`) and **its own artifact namespace** (`week10_kelly_*`). It does not touch `week8_*` (Round 1) or `week9_*` (Round 2) files. It is therefore safe to launch in parallel with — or after — any in-flight Round 2 pods.

### 13.0. What the new pipeline writes

Every artifact stem is `<prefix>_kelly_*`. With the default `--artifact-prefix=week10_kelly` you get:

| Stem | Content |
|---|---|
| `*_kelly_experiment_grid.csv` | One row per Optuna trial (params + holdout metrics). |
| `*_kelly_best_metrics.json` | Best trial's params, holdout metrics, equal-weight comparison, copula diagnostics. |
| `*_kelly_best_timeseries.csv` | Per-step holdout: portfolio return, log-wealth, turnover, Σ log-wealth, equal-weight log-wealth. |
| `*_kelly_best_weights.csv` | Per-step holdout weight vector at the best trial. |
| `*_kelly_copula_diagnostics.json` | MLP head + final-step correlation matrix summary stats. |
| `*_kelly_run_manifest.json` | Argparse + experiment config + git SHA at run start. |
| `figures/*_kelly_log_wealth.png` | Cumulative log-wealth, Kelly vs equal-weight. |
| `figures/*_kelly_turnover.png` | Per-step L1 turnover (and rolling mean). |
| `figures/*_kelly_drawdown.png` | Drawdown of cumulative log-wealth, Kelly vs equal-weight. |
| `figures/*_kelly_copula_corr_heatmap.png` | Final-step copula correlation heatmap. |
| `docs/<prefix>_diagnostics_report.md` | Human-readable summary (log-wealth growth, turnover, copula stats — no Sortino/variance). |

### 13.1. Pre-reqs (delta vs sections 0–3)

1. **Code:** `cloud-runs` already contains `script/polymarket_week10_kelly_pipeline.py`, `src/kelly_copula_optimizer.py`, and the academic summary doc. No edits required.
2. **Cached week8 inputs:** by default the new script *consumes* `week8_*` markets/prices/exogenous CSVs (via `--input-artifact-prefix=week8`) and copies them to `week10_kelly_*` so it never re-fetches Polymarket / yfinance. On a freshly cloned pod those week8 CSVs are already in `data/processed/` on `cloud-runs`. If you'd rather rebuild from scratch, pass `--rebuild-data` (slow — full Polymarket + yfinance pull).
3. **Python deps:** identical to week8 — `scipy` is still required by Optuna's QMCSampler (`uv pip install scipy` per §2d). No new requirements; the MLP / copula are pure PyTorch on top of what's already installed.
4. **Threading:** the Kelly inner step is more PyTorch-heavy per trial than the week8 inner step (Monte-Carlo sampling × MLP forward × Cholesky × log1p), so BLAS parallelism matters more than Optuna trial parallelism. On a 16-vCPU pod use the same `OMP/MKL/TORCH=4` + `--optuna-n-jobs 4` from §3a. On 32 vCPU prefer `OMP=8 + n_jobs=4` (more BLAS, fewer concurrent trials) over the §3a default of `OMP=4 + n_jobs=8`.

### 13.2. Round 3 ablation matrix (one pod each)

Three pods are enough to bracket the contribution of each non-convex piece:

| Pod | Macro / copula | Turnover λ | Other flags | Hypothesis |
|---|---|---|---|---|
| **K10A** | dynamic copula (default, MLP over SPY/QQQ/BTC) | search | `--reduced-search` | Full system: Kelly + dynamic copula + L1 turnover, dense Sobol on a small budget. |
| **K10B** | **disabled** (`--no-exogenous` → R = I) | search | `--reduced-search` | Isolates the copula contribution. Same Kelly + L1, no MLP, no exogenous coupling. |
| **K10C** | dynamic copula | **pinned to 0** (`--lambda-turnover-override 0`) | `--reduced-search` | Isolates the turnover penalty. Same copula, zero L1 — measures slippage cost explicitly. |

### Experiment K10A — full Kelly + dynamic copula + L1 turnover

```bash
python -u script/polymarket_week10_kelly_pipeline.py \
  --artifact-prefix week10_kelly_A \
  --input-artifact-prefix week8 \
  --reduced-search \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-K10A \
  --git-commit-message "K10A: dynamic-copula Kelly OGD (default) ${RUN_TAG}" \
  2>&1 | tee "run_K10A_${RUN_TAG}.log"
```

### Experiment K10B — Kelly + L1 turnover, copula disabled

```bash
python -u script/polymarket_week10_kelly_pipeline.py \
  --artifact-prefix week10_kelly_B \
  --input-artifact-prefix week8 \
  --no-exogenous \
  --reduced-search \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-K10B \
  --git-commit-message "K10B: Kelly + L1, copula disabled (R=I) ${RUN_TAG}" \
  2>&1 | tee "run_K10B_${RUN_TAG}.log"
```

### Experiment K10C — Kelly + dynamic copula, turnover penalty pinned to 0

```bash
python -u script/polymarket_week10_kelly_pipeline.py \
  --artifact-prefix week10_kelly_C \
  --input-artifact-prefix week8 \
  --lambda-turnover-override 0 \
  --reduced-search \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-K10C \
  --git-commit-message "K10C: dynamic-copula Kelly, lambda_turnover=0 ${RUN_TAG}" \
  2>&1 | tee "run_K10C_${RUN_TAG}.log"
```

### 13.3. Parallel-safety vs in-flight Round 2 pods

- All Round 3 stems are `week10_kelly_<X>_*`. Zero overlap with `week8_*` or `week9_*`.
- Each Round 3 pod must use a **unique `--git-push-branch`** (the script's default is `cloud-runs`, same as week8 — overriding to `cloud-runs-K10<X>` is required to keep the parallel-safe convention from §4).
- Round 2 pods that are still running pushed off a `cloud-runs` snapshot from before the week10 commit landed. The eventual §8 fan-in merges (`cloud-runs-A2 … F2 → cloud-runs`) preserve the new week10 files via standard git three-way merge — those branches simply don't touch them.
- If you do invoke the §10 recovery flow on a Round 2 pod (`git fetch && git reset --hard origin/cloud-runs`), that pod's working tree will now also contain the three week10 files. Harmless — the week8 pipeline does not import them.

### 13.4. Quick "did Kelly beat equal-weight?" peek

Run on the pod after the pipeline completes (replace `<X>` with the pod letter):

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('data/processed/week10_kelly_<X>*_kelly_best_metrics.json')):
    d = json.load(open(f))
    bp = d.get('best_params', {})
    h  = d.get('holdout_metrics', {})
    e  = d.get('equal_weight_holdout', {})
    print(f'{f.split(chr(47))[-1]:60s}  '
          f'kelly_logW={h.get(\"sum_log_wealth\", 0):+.4f}  '
          f'eq_logW={e.get(\"sum_log_wealth\", 0):+.4f}  '
          f'delta={h.get(\"sum_log_wealth\",0)-e.get(\"sum_log_wealth\",0):+.4f}  '
          f'mean_turnover={h.get(\"mean_turnover\",0):.4f}  '
          f'lambda_t={bp.get(\"turnover_lambda\",0):.4f}')
"
```

A positive `delta` means the Kelly + dynamic-copula portfolio earned more cumulative log-wealth than equal-weight on the holdout window.

### 13.5. Fan back in to `cloud-runs` (mirrors §8)

```bash
cd ~/.../STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git pull --ff-only

git merge --no-ff origin/cloud-runs-K10A -m "merge week10_kelly_A artifacts"
git merge --no-ff origin/cloud-runs-K10B -m "merge week10_kelly_B artifacts"
git merge --no-ff origin/cloud-runs-K10C -m "merge week10_kelly_C artifacts"
git push origin cloud-runs

# Optional cleanup once the merges are pushed:
git push origin --delete cloud-runs-K10A cloud-runs-K10B cloud-runs-K10C
```

Conflict-free against `week8_*` and `week9_*` artifacts already on `cloud-runs` because every stem is uniquely `week10_kelly_<X>*`.

### 13.6. Cheat sheet — single Round 3 pod

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

tmux new -s week10
# inside tmux (16-vCPU pod):
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Pick ONE block from section 13.2 (--artifact-prefix and --git-push-branch
# values must be unique per pod). Paste it and let the run finish.
```

For a pod that already has Round 1 / Round 2 set up, this collapses to: `git fetch && reset --hard origin/cloud-runs`, re-export the env vars inside tmux, paste the §13.2 block.

---

## 14. Round 4 — post-C2/G2 cross-term week8 experiments

Round 4 builds on the two positive Round 2 results (C2: +0.0040 Sortino delta with `macro=both`, and G2: +0.0019 delta with `--momentum-screening`) and tests the experiments Round 2 never ran. Every Round 4 pod uses the **same `polymarket_week8_pipeline.py`** script as Round 2, but with post-Round-2 levers combined in ways that Round 2 did not cover. Round 4 is parallel-safe with in-flight Round 3 (Kelly) pods from §13 because stems are uniquely `week11_<X>_*` and branches are uniquely `cloud-runs-<X>4`.

### 14.0. Round 4 hypothesis and ablation matrix

Round 2's cross-pod synthesis identified five headline findings. Round 4 is the follow-up ablation:

| Round 2 finding | Round 4 response |
|---|---|
| `macro=both` > `rescale` > `explicit` (C2: +0.004 vs A2: −0.060) | Every Round 4 pod uses `--macro-integration both`. |
| Momentum screening buys ~10× the raw return (G2's 0.00126 vs ~0.00015 elsewhere) | Pods I, K, L add `--momentum-screening` on top of macro=both. |
| `--beat-baseline-objective` hurt on D2/E2/F2 | Dropped from Round 4 entirely. |
| `--baseline-shrinkage` alone collapsed to α=0 (B2 never used the constrained portfolio) | Dropped as a primary lever. |
| `top_k_bagging=10` underperformed 5 (E2) and `=1` collapsed (B2) | Fixed at 5 (pods I, K, L) or 3 (pod M's noise-reduction check). |

All five pods drop `--etf-tracking` (Round 1's largest negative finding) and keep `--reduced-search` (denser Sobol coverage of the meaningful subspace). The four pods differ along three axes: momentum window, rolling window, and Optuna budget.

| Pod | Macro | Momentum screening | Rolling window | Top-K | Trials | Hypothesis |
|---|---|---|---|---|---|---|
| **I** | both | top-20 / 5d (same as G2) | default (24, 48, 96) | 5 | 100 | **The missing C2 × G2 cross-term.** Predicted to be the strongest Round 4 pod. |
| **K** | both | top-20 / 5d | pinned to 96 only | 5 | 100 | G2 selected rolling=96. Does that carry over when macro=both is active? |
| **L** | both | top-25 / 10d | default | 5 | 100 | Universe sensitivity around G2's peak. Wider momentum window, larger universe. |
| **M** | both | none (full universe) | default | 3 | **200** | Denser Sobol + smaller bag on C2's winning config; tests whether C2's +0.004 was selection noise or real. |

### 14.1. Prerequisites (delta vs §0–§3 and §13.1)

1. **Code:** `cloud-runs` as of commit `1436c71` contains the merged `origin/momentum-lever` changes — so `--momentum-screening`, `--momentum-top-n`, `--momentum-lookback-days`, and the pandas-2.3 dtype fix are all present on `cloud-runs`. No extra branch-juggling.
2. **Round 2 fan-in:** if you have not yet merged `cloud-runs-A2 … H2` back into `cloud-runs` per §8, Round 4 pods will not see the Round 2 artifacts (`data/processed/week9_*_constrained_best_metrics.json`) locally for comparison. This does not block Round 4 — each Round 4 pod builds its own baseline — but it makes the cross-comparison on your laptop harder later. Recommended order: fan-in Round 2 → launch Round 4.
3. **Python deps:** identical to Round 2. `scipy` still required by Optuna's QMCSampler (§2d).
4. **Threading:** same rules as §3a. Use `OMP/MKL/TORCH=4 + --optuna-n-jobs 4` on a 16-vCPU pod. Pod M's 200-trial budget runs ~2× longer than the others but is otherwise unchanged.

### 14.2. Round 4 pod commands

Every block below sets `--artifact-prefix week11_<X>`, `--git-push-branch cloud-runs-<X>4`, and `--macro-integration both`. Launch inside tmux after exporting the threading env vars per §3/§13.1.

#### Experiment I — macro=both + momentum top-20 / 5d (the missing cross-term)

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week11_I \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 5 \
  --momentum-screening \
  --momentum-top-n 20 \
  --momentum-lookback-days 5.0 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-I4 \
  --git-commit-message "I4: macro=both + mom20/5d + top-K=5 (C2xG2 cross-term) ${RUN_TAG}" \
  2>&1 | tee "run_I4_${RUN_TAG}.log"
```

#### Experiment K — macro=both + momentum top-20 / 5d + longer rolling window

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week11_K \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 5 \
  --momentum-screening \
  --momentum-top-n 20 \
  --momentum-lookback-days 5.0 \
  --rolling-windows 96 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-K4 \
  --git-commit-message "K4: I + rolling_window=96 only ${RUN_TAG}" \
  2>&1 | tee "run_K4_${RUN_TAG}.log"
```

If `--rolling-windows` isn't wired as a CLI override in your working tree, drop that flag — the default `(24, 48, 96)` search still exercises 96. Pod K's delta vs pod I just becomes a pure reproduction run, which is still useful for variance measurement.

#### Experiment L — macro=both + wider momentum universe (top-25 / 10d)

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week11_L \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 5 \
  --momentum-screening \
  --momentum-top-n 25 \
  --momentum-lookback-days 10.0 \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-L4 \
  --git-commit-message "L4: macro=both + mom25/10d (universe sensitivity) ${RUN_TAG}" \
  2>&1 | tee "run_L4_${RUN_TAG}.log"
```

#### Experiment M — macro=both full universe + smaller bag + 2× trials (C2 robustness check)

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week11_M \
  --macro-integration both \
  --reduced-search \
  --top-k-bagging 3 \
  --optuna-n-jobs 4 \
  --optuna-trials 200 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-M4 \
  --git-commit-message "M4: macro=both + K=3 + 200 trials (C2 noise check) ${RUN_TAG}" \
  2>&1 | tee "run_M4_${RUN_TAG}.log"
```

### 14.3. What each Round 4 pod writes

Stems are `week11_<X>_*` and do not overlap with Round 1 (`week8_*`), Round 2 (`week9_*`), Round 3 Kelly (`week10_kelly_*`), or each other.

| Pod | Constrained Optuna stems | Baseline / exog / covariance / figures stem | Manifest | Push branch |
|---|---|---|---|---|
| I (both + mom20/5d + K=5) | `week11_I_macro_both_constrained_*` | `week11_I_*` | `week11_I_run_manifest.json` | `cloud-runs-I4` |
| K (both + mom20/5d + K=5 + rw=96) | `week11_K_macro_both_constrained_*` | `week11_K_*` | `week11_K_run_manifest.json` | `cloud-runs-K4` |
| L (both + mom25/10d + K=5) | `week11_L_macro_both_constrained_*` | `week11_L_*` | `week11_L_run_manifest.json` | `cloud-runs-L4` |
| M (both + full universe + K=3 + 200 trials) | `week11_M_macro_both_constrained_*` | `week11_M_*` | `week11_M_run_manifest.json` | `cloud-runs-M4` |

### 14.4. Quick "did we beat Round 2's best?" peek

Run on the pod after the pipeline completes (replace `<X>` with the pod letter). This compares against the Round 2 winner C2's `+0.0040` delta as the bar to clear:

```bash
python3 -c "
import json, glob
R2_BEST_DELTA = 0.0040  # C2 from Round 2
for f in sorted(glob.glob('data/processed/week11_<X>*_constrained_best_metrics.json')):
    d = json.load(open(f))['best_params']
    delta = d['holdout_sortino_minus_baseline']
    print(f'{f.split(chr(47))[-1]:60s}  delta={delta:+.4f}  '
          f'vs_C2={delta - R2_BEST_DELTA:+.4f}  '
          f'sortino={d[\"holdout_sortino_ratio\"]:.4f}  '
          f'baseline={d[\"baseline_holdout_sortino\"]:.4f}  '
          f'mean_ret={d[\"holdout_mean_return\"]:+.6f}')
"
```

A positive `delta` means the pod beat the equal-weight baseline; a positive `vs_C2` means it also beat Round 2's best.

### 14.5. Fan-back-in to `cloud-runs` (mirrors §8 and §13.5)

```bash
cd ~/.../STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git pull --ff-only

git merge --no-ff origin/cloud-runs-I4 -m "merge week11_I artifacts"
git merge --no-ff origin/cloud-runs-K4 -m "merge week11_K artifacts"
git merge --no-ff origin/cloud-runs-L4 -m "merge week11_L artifacts"
git merge --no-ff origin/cloud-runs-M4 -m "merge week11_M artifacts"
git push origin cloud-runs

# Optional cleanup once merged:
git push origin --delete cloud-runs-I4 cloud-runs-K4 cloud-runs-L4 cloud-runs-M4
```

Conflict-free against Round 1 / Round 2 / Round 3 Kelly artifacts already on `cloud-runs` because every stem is uniquely `week11_<X>_*`.

### 14.6. Cheat sheet — single Round 4 pod

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

# GitHub credentials for --git-commit-and-push (see §2e-style PAT setup):
git config --global credential.helper store
cat > ~/.git-credentials <<'EOF'
https://<USERNAME>:<FINE_GRAINED_PAT>@github.com
EOF
chmod 600 ~/.git-credentials

tmux new -s week11
# inside tmux (16-vCPU pod):
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Pick ONE block from §14.2 (--artifact-prefix and --git-push-branch
# must be unique per pod). Paste it and let the run finish.
```

For a pod that already has Round 1 / Round 2 / Round 3 set up, this collapses to: `git fetch && reset --hard origin/cloud-runs`, re-export the env vars inside tmux, paste the §14.2 block.

### 14.7. Priority ordering if you can't launch all four

Expected-value-per-pod ranking, highest first:

1. **Pod I** — the missing C2 × G2 cross-term. Highest probability of beating C2's +0.004. Launch first.
2. **Pod L** — tests whether G2's momentum sweet spot generalizes with a wider filter. Medium probability of a win; high information value regardless.
3. **Pod M** — C2-robustness check. Does not introduce a new lever but tightens statistical confidence on the +0.004 Round 2 headline. Useful for the writeup.
4. **Pod K** — confirmation / variance-measurement pod. Only launch if you have a spare pod.

Minimum viable Round 4 if compute-constrained: **just pod I**. If two pods: **I + L**. If three: **I + L + M**. Four pods exercises the full matrix.

### 14.8. Levers originally deferred — now implemented (used in §15 Round 5)

Round 4 left two levers on the table because they required code changes, not just flag changes. Both are now implemented and ready for Round 5:

- **LR-ceiling widening — shipped as `--lr-values`.** C2's winning learning rate of 0.045 sat at the top of the `--reduced-search` LR range `(0.005, 0.01, 0.02, 0.05)`. `script/polymarket_week8_pipeline.py` now takes `--lr-values LR1,LR2,...` (mirrors `--rolling-windows`). The flag overrides `ExperimentConfig.learning_rates` **after** `--reduced-search` is applied, so `--reduced-search --lr-values 0.04,0.06,0.08,0.10,0.12` widens the search past the Round 2 ceiling without touching the other reduced-search narrowings.
- **Post-hoc α-blend evaluator — shipped as `script/posthoc_alpha_blend.py`.** Takes an artifact prefix + constrained stem, loads the already-written baseline and constrained holdout timeseries, sweeps `α ∈ [0, 1]`, and reports Sortino / mean / volatility / max-dd / total-return / cumulative-log-wealth for each blend. Does **not** re-run Optuna. This is the risk-adjusted sensitivity test that `--baseline-shrinkage` cannot do (because every Round 2 pod that searched over shrinkage collapsed to α=0 — the training objective never rewards dilution inside a fold). Sanity-tested on `week8`: interior argmax at α=0.5 (Sortino +0.086) beats α=1.0 (+0.070), i.e. the pure constrained portfolio was over-exposed on a risk-adjusted basis.

Use these in §15 Round 5 below.

## 15. Round 5 — LR ceiling + post-hoc α-blend

**Status:** ready to launch. Gates on §14 Round 4 completion (pods I4/K4/L4/M4 fanned into `cloud-runs`). Purpose is twofold:

1. **Forward-search the LR plateau above C2's 0.045 cap.** C2 converged at the top of the reduced-search LR range, which means the true Sortino-optimal LR may be 0.06-0.10 (we never looked). Pods O and P sweep above the ceiling. Pod Q combines the LR lever with Round 4's winning `macro=both + momentum top-20 / 5d` recipe.
2. **Post-hoc α-blend on every Round 2 / Round 4 winner.** Pod R runs no optimization — it just scores already-completed runs. This tells us, for each pod, whether the reported α=1.0 Sortino is risk-adjusted-optimal or whether a blended allocation would have dominated. Often *that* table is the one that belongs in the writeup.

### 15.1. Round 5 ablation matrix

| Pod | Recipe | `--macro-modes` | momentum? | `--lr-values` | `--rolling-windows` | trials | push branch |
|---|---|---|---|---|---|---|---|
| O | LR sweep, macro both | `both` | no | `0.04,0.06,0.08,0.10,0.12` | default (24,48,96) | 200 | `cloud-runs-O5` |
| P | LR sweep + rw=96 (C2 x-term) | `both` | no | `0.04,0.06,0.08,0.10,0.12` | `96` | 200 | `cloud-runs-P5` |
| Q | full recipe (Pod I + LR sweep) | `both` | `top-20 / 5d` | `0.04,0.06,0.08,0.10,0.12` | `96` | 200 | `cloud-runs-Q5` |
| R | **post-hoc α-blend only** | n/a | n/a | n/a | n/a | 0 | `cloud-runs-R5` |

Pods O–Q share prior env setup from §14.6 (clone, venv, PAT). Pod R does not need a GPU pod — it only reads existing CSVs from `cloud-runs` and writes small artifacts; the cheapest CPU container is fine.

### 15.2. Pod commands (paste inside tmux)

**Pod O — LR sweep, macro both, default rolling windows:**

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week12_O \
  --macro-modes both \
  --reduced-search \
  --lr-values 0.04,0.06,0.08,0.10,0.12 \
  --top-k-bagging 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 200 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-O5 \
  --git-commit-message "O5: macro=both + LR sweep 0.04..0.12 + K=5 + 200 trials ${RUN_TAG}" \
  2>&1 | tee "run_O5_${RUN_TAG}.log"
```

**Pod P — LR sweep pinned to rw=96 (clean C2-ceiling cross-term):**

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week12_P \
  --macro-modes both \
  --reduced-search \
  --lr-values 0.04,0.06,0.08,0.10,0.12 \
  --rolling-windows 96 \
  --top-k-bagging 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 200 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-P5 \
  --git-commit-message "P5: macro=both + LR sweep + rw=96 + K=5 + 200 trials ${RUN_TAG}" \
  2>&1 | tee "run_P5_${RUN_TAG}.log"
```

**Pod Q — full recipe (Pod I's winning config + Round 5 LR sweep):**

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week12_Q \
  --macro-modes both \
  --reduced-search \
  --lr-values 0.04,0.06,0.08,0.10,0.12 \
  --rolling-windows 96 \
  --momentum-screening \
  --momentum-top-n 20 \
  --momentum-lookback-days 5 \
  --top-k-bagging 5 \
  --optuna-n-jobs 4 \
  --optuna-trials 200 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-Q5 \
  --git-commit-message "Q5: full recipe (both + mom20/5d + rw=96 + LR sweep + K=5 + 200 trials) ${RUN_TAG}" \
  2>&1 | tee "run_Q5_${RUN_TAG}.log"
```

**Pod R — post-hoc α-blend over every shipped pod (no optimization):**

```bash
source .venv/bin/activate
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Runs in seconds per pod. Skip gracefully if a given stem was never fanned in.
for prefix_stem in \
    "week8:week8" \
    "week9_A:week9_A_macro_both" \
    "week9_B:week9_B_macro_both" \
    "week9_C:week9_C_macro_both" \
    "week9_D:week9_D_macro_both" \
    "week9_E:week9_E_macro_both" \
    "week9_F:week9_F_macro_both" \
    "week9_G:week9_G_macro_both" \
    "week9_H:week9_H_macro_both" \
    "week11_I:week11_I_macro_both" \
    "week11_K:week11_K_macro_both" \
    "week11_L:week11_L_macro_both" \
    "week11_M:week11_M_macro_both"; do
  prefix="${prefix_stem%%:*}"
  stem="${prefix_stem##*:}"
  base="data/processed/${prefix}_baseline_timeseries.csv"
  cns="data/processed/${stem}_constrained_best_timeseries.csv"
  if [[ -f "$base" && -f "$cns" ]]; then
    python script/posthoc_alpha_blend.py \
      --artifact-prefix "$prefix" \
      --constrained-stem "$stem" \
      --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
      2>&1 | tee -a "run_R5_${RUN_TAG}.log"
  else
    echo "[skip] $prefix / $stem (timeseries missing)" | tee -a "run_R5_${RUN_TAG}.log"
  fi
done

git add data/processed/*_alpha_blend*.csv \
        data/processed/*_alpha_blend_summary.md \
        data/processed/*_alpha_blend_sortino.png
git commit -m "R5: post-hoc alpha-blend sweep across Round 2 + Round 4 winners ${RUN_TAG}"
git checkout -b cloud-runs-R5
git push -u origin cloud-runs-R5
```

### 15.3. What each Round 5 pod writes

Stems are `week12_<X>_*` and do not overlap with prior rounds. Pod R uses a different naming convention because it only writes α-blend artifacts (no new `_constrained_best_*` files).

| Pod | Constrained Optuna stems | Baseline / figures stem | Manifest | Push branch |
|---|---|---|---|---|
| O | `week12_O_macro_both_constrained_*` | `week12_O_*` | `week12_O_run_manifest.json` | `cloud-runs-O5` |
| P | `week12_P_macro_both_constrained_*` | `week12_P_*` | `week12_P_run_manifest.json` | `cloud-runs-P5` |
| Q | `week12_Q_macro_both_constrained_*` | `week12_Q_*` | `week12_Q_run_manifest.json` | `cloud-runs-Q5` |
| R | _none_ (post-hoc) | `{stem}_alpha_blend.csv`, `{stem}_alpha_blend_summary.md`, `{stem}_alpha_blend_sortino.png` per prior pod | _none_ | `cloud-runs-R5` |

### 15.4. Quick "where is LR optimum?" peek (pods O / P / Q)

```bash
python3 -c "
import json, glob
# Round 4 winner (Pod I) is the new bar to clear; fall back to Round 2 C2 if I4 missing.
for f in sorted(glob.glob('data/processed/week12_*_constrained_best_metrics.json')):
    d = json.load(open(f))['best_params']
    print(f'{f.split(chr(47))[-1]:60s}  '
          f'lr={d[\"learning_rate\"]:.4f}  '
          f'rw={d[\"rolling_window\"]}  '
          f'sortino={d[\"holdout_sortino_ratio\"]:.4f}  '
          f'baseline={d[\"baseline_holdout_sortino\"]:.4f}  '
          f'delta={d[\"holdout_sortino_minus_baseline\"]:+.4f}')
"
```

If the winning `learning_rate` on pods O / P / Q lands in the interior of `{0.04, 0.06, 0.08, 0.10, 0.12}` and the Sortino delta is meaningfully above Round 4 Pod I's, Round 5 has found a real plateau. If the argmax pins to 0.12 (the new ceiling), a Round 6 with an even wider grid is warranted.

### 15.5. Quick "is α=1 efficient?" peek (pod R)

```bash
python3 -c "
import csv, glob
for path in sorted(glob.glob('data/processed/*_alpha_blend.csv')):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    best = max(rows, key=lambda r: float(r['sortino']))
    at1  = next((r for r in rows if abs(float(r['alpha']) - 1.0) < 1e-6), None)
    stem = path.split('/')[-1].replace('_alpha_blend.csv', '')
    s1   = float(at1['sortino']) if at1 else float('nan')
    ba   = float(best['alpha'])
    bs   = float(best['sortino'])
    gap  = bs - s1
    flag = '  <-- interior argmax dominates' if (ba < 1.0 - 1e-6 and gap > 1e-4) else ''
    print(f'{stem:40s}  a*={ba:.2f}  S*={bs:+.4f}  S(a=1)={s1:+.4f}  gap={gap:+.4f}{flag}')
"
```

Pods tagged `<-- interior argmax dominates` are candidates for a blended allocation in the final writeup: the optimizer's recommended portfolio is dominated on risk-adjusted grounds by a shrinkage toward the equal-weight baseline.

### 15.6. Fan-back-in to `cloud-runs`

```bash
cd ~/.../STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git pull --ff-only

git merge --no-ff origin/cloud-runs-O5 -m "merge week12_O artifacts (LR sweep, macro both)"
git merge --no-ff origin/cloud-runs-P5 -m "merge week12_P artifacts (LR sweep + rw=96)"
git merge --no-ff origin/cloud-runs-Q5 -m "merge week12_Q artifacts (full recipe + LR sweep)"
git merge --no-ff origin/cloud-runs-R5 -m "merge Round 5 post-hoc alpha-blend artifacts"
git push origin cloud-runs

git push origin --delete cloud-runs-O5 cloud-runs-P5 cloud-runs-Q5 cloud-runs-R5
```

Conflict-free against Rounds 1–4 because `week12_*_*` stems and `*_alpha_blend*` files are new.

### 15.7. Priority ordering if you can't launch all four

1. **Pod R** — cheapest by an order of magnitude (seconds-to-minutes, CPU-only, no optimization). Run first: it tells you whether Round 4's reported Sortinos were already risk-adjusted-optimal or whether a blended allocation dominates. Mandatory for the writeup.
2. **Pod Q** — the one most likely to produce a new best: stacks the Round 4 winning config (I4) with the Round 5 new lever (widened LR). Launch second.
3. **Pod O** — isolates the LR lever cleanly (no momentum, default rolling windows). Useful as an ablation of Pod Q.
4. **Pod P** — LR sweep + rw=96 cross-term. Lowest priority; run if you have a fourth pod.

Minimum viable Round 5: **just pod R**. If two pods: **R + Q**. If three: **R + Q + O**. Four pods exercises the full matrix.

### 15.8. Open levers for Round 6+

- **LR plateau past 0.12.** If any Round 5 pod pins to lr=0.12 (the new ceiling), widen again: `--lr-values 0.08,0.12,0.16,0.20,0.25`.
- **Joint (LR, entropy) search.** Every reduced-search pod fixes `entropy_lambda=0`. If Round 5 unlocks a higher-LR regime, the optimal entropy may no longer be zero. Would require adding an `--entropy-lambdas` override flag (same 10-line pattern as `--lr-values`).
- **Post-hoc α-blend with Sortino-optimal α as a reportable metric.** If Pod R shows that α* < 1 dominates on every pod, we could formalize this as a second-stage estimator ("pipeline + post-hoc shrinkage") and report both. Not a new experiment — a writeup decision.
- **α-blend frontier on the Kelly (week10) outputs.** The `kelly_best_timeseries.csv` has `portfolio_return` too; `posthoc_alpha_blend.py` accepts arbitrary prefixes, so blending Kelly-vs-baseline is a one-liner once Round 3 finishes.
