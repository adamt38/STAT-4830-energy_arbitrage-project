# Cloud runbook — Week 8 Polymarket pipeline (parallel-safe)

End-to-end guide for running every Week 8 experiment **on its own pod, in parallel, with zero merge conflicts**. Each pod writes to a unique `--artifact-prefix` and pushes to its own `--git-push-branch`; results are merged back into `cloud-runs` on the laptop at the end.

This guide also folds in the recent speed fixes (hourly bars, smaller rolling-window set, fewer inner steps, more aggressive Optuna pruning), so a single 100-trial run is roughly an order of magnitude faster than the original config.

---

## 0. One-time prerequisites

- Code lives on the `cloud-runs` branch of `https://github.com/adamt38/STAT-4830-energy_arbitrage-project`. Every pod clones from it.
- `cloud-runs` already contains the speed knobs and the new `--artifact-prefix` flag — no edits needed.

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
tmux new -s week8
```

Inside tmux (assuming a 16-vCPU pod — adjust per [section 3a](#3a-cpu--threading-rules-of-thumb) if not):

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
```

`Ctrl-b d` to detach; `tmux attach -t week8` to reattach later.

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

## 4. Experiments — one per pod, all parallel-safe

Every command below sets two things that make parallel execution safe:

- `--artifact-prefix week8_<X>` — every CSV/JSON/PNG/manifest this run writes is stamped with that prefix, so two pods never touch the same filename.
- `--git-push-branch cloud-runs-<X>` — results are pushed to a branch named after the experiment, so two pods never push to the same ref.

`python -u` keeps stdout/stderr unbuffered (paired with `PYTHONUNBUFFERED=1` from section 3) so trial-completion lines surface in the SSH pane immediately. `--optuna-n-jobs 4` is sized for a 16-vCPU pod with `OMP/MKL/TORCH=4`; if your pod has a different `nproc`, adjust both per the table in [section 3a](#3a-cpu--threading-rules-of-thumb).

### Experiment A — Baseline Optuna (rescale macro, no ETF)

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_A \
  --macro-integration rescale \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-A \
  --git-commit-message "A: rescale baseline (week8_A)" \
  2>&1 | tee "run_A_${RUN_TAG}.log"
```

### Experiment B — Rescale macro + ETF tracking

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_B \
  --macro-integration rescale \
  --etf-tracking \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-B \
  --git-commit-message "B: rescale + ETF (week8_B)" \
  2>&1 | tee "run_B_${RUN_TAG}.log"
```

### Experiment C — Explicit macro + ETF tracking

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_C \
  --macro-integration explicit \
  --etf-tracking \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-C \
  --git-commit-message "C: explicit + ETF (week8_C)" \
  2>&1 | tee "run_C_${RUN_TAG}.log"
```

### Experiment D — Both macro modes combined + ETF tracking

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_D \
  --macro-integration both \
  --etf-tracking \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-D \
  --git-commit-message "D: both + ETF (week8_D)" \
  2>&1 | tee "run_D_${RUN_TAG}.log"
```

### Experiment E — Three separate macro studies (rescale, explicit, both) + ETF

Runs three back-to-back Optuna studies on a single pod with distinct constrained-output suffixes (`week8_E`, `week8_E_macro_explicit`, `week8_E_macro_both`).

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_E \
  --macro-modes rescale,explicit,both \
  --etf-tracking \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-E \
  --git-commit-message "E: three-macro + ETF (week8_E)" \
  2>&1 | tee "run_E_${RUN_TAG}.log"
```

### Experiment F — Joint macro search + ETF

One Optuna study with categorical `macro_mode` and conditional `regime_k` / `lambda_macro_explicit`. Do **not** combine with `--macro-modes`.

```bash
python -u script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_F \
  --joint-macro-mode-search \
  --etf-tracking \
  --optuna-n-jobs 4 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-F \
  --git-commit-message "F: joint macro + ETF (week8_F)" \
  2>&1 | tee "run_F_${RUN_TAG}.log"
```

---

## 5. What each experiment writes

Every run writes to `data/processed/`, `figures/`, and the manifest, all stamped with the experiment's prefix. There is **no overlap** across experiments.

| Experiment | Constrained Optuna stems | Baseline / exog / covariance / figures stem | Manifest |
|---|---|---|---|
| A (rescale, no ETF) | `week8_A_constrained_*` | `week8_A_*` | `week8_A_run_manifest.json` |
| B (rescale + ETF) | `week8_B_constrained_*` | `week8_B_*` | `week8_B_run_manifest.json` |
| C (explicit + ETF) | `week8_C_macro_explicit_constrained_*` | `week8_C_*` | `week8_C_run_manifest.json` |
| D (both + ETF) | `week8_D_macro_both_constrained_*` | `week8_D_*` | `week8_D_run_manifest.json` |
| E (three macros + ETF) | `week8_E_constrained_*`, `week8_E_macro_explicit_constrained_*`, `week8_E_macro_both_constrained_*` | `week8_E_*` | `week8_E_run_manifest.json` |
| F (joint macro + ETF) | `week8_F_constrained_*` (joint study tracked via trial params) | `week8_F_*` | `week8_F_run_manifest.json` |

The manifest carries `git_commit_and_push_requested`, the ETF / macro flags you passed, and the artifact prefix, so you can always reconstruct what was run.

---

## 6. How to confirm a run finished and pushed

In the tmux output / `run_<X>_${RUN_TAG}.log` you should see, in order:

```
PIPELINE COMPLETE
  Total time: ...
- run_manifest: data/processed/week8_<X>_run_manifest.json
PIPELINE STAGE: Git commit and push
Git publish: committed and pushed branch 'cloud-runs' to origin/cloud-runs-<X>.
```

(Or `Git publish: nothing new to commit; skipping pull/push.` if you re-ran an identical config.)

If git fails, the script exits non-zero **after** the science stages — the log shows the git error and there is no silent success. On GitHub → `cloud-runs-<X>` you should see a new commit dated within the last few minutes.

---

## 7. Detach / reconnect / cleanup

```bash
# Detach so you can close SSH safely:
Ctrl-b d

# Reconnect later:
prime pods ssh <pod-id>
tmux attach -t week8

# End session when done (inside tmux):
exit
# or, from outside tmux:
tmux kill-session -t week8

# Pull extra files to your Mac (optional):
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/run_*.log' .
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/data/processed/week8_*.csv' .
scp 'user@<host>:~/STAT-4830-energy_arbitrage-project/data/processed/week8_*.json' .
```

---

## 8. After all pods finish — fan back in to `cloud-runs`

Each pod pushed to its own branch (`cloud-runs-A` … `cloud-runs-F`). Because every artifact stem is unique, **the merges are conflict-free**:

```bash
cd ~/.../STAT-4830-energy_arbitrage-project
git fetch origin
git checkout cloud-runs
git pull --ff-only

git merge --no-ff origin/cloud-runs-A -m "merge week8_A artifacts"
git merge --no-ff origin/cloud-runs-B -m "merge week8_B artifacts"
git merge --no-ff origin/cloud-runs-C -m "merge week8_C artifacts"
git merge --no-ff origin/cloud-runs-D -m "merge week8_D artifacts"
git merge --no-ff origin/cloud-runs-E -m "merge week8_E artifacts"
git merge --no-ff origin/cloud-runs-F -m "merge week8_F artifacts"
git push origin cloud-runs
```

If you ever do see a conflict here, it means two experiments shared a prefix — fix the offending pod's `--artifact-prefix` and rerun that one only.

Optional cleanup once the merges are pushed:

```bash
git push origin --delete cloud-runs-A cloud-runs-B cloud-runs-C cloud-runs-D cloud-runs-E cloud-runs-F
```

---

## 9. Cheat sheet — single pod, single experiment

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
pip install scipy
git config --global user.email "you@example.edu"
git config --global user.name  "Your Name"

tmux new -s week8
# inside tmux (16-vCPU pod; adjust per section 3a if not):
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Pick ONE block from section 4 (the --artifact-prefix and --git-push-branch
# values must be unique per pod). Paste it and let the run finish.
```

That's the whole loop. Six pods, six commands from section 4, one merge sequence in section 8 → all six experiments land cleanly on `cloud-runs`.

---

## 10. Recovery — pod was launched with old `OMP=1 --optuna-n-jobs 12/16` settings

Symptoms of the bad config: `top` shows python at ~300% CPU on a 16-core pod (only ~3 cores busy), `run_*.log` stalls at the Optuna start banner with no `Trial N finished` lines for 30+ minutes.

The fix is to kill the bad process and restart with the new env + `python -u`. Don't try to `export` new env vars onto the running process — env vars are inherited at process start, so the running python is locked to whatever was set when it was launched.

```bash
prime pods ssh <pod-id>
tmux attach -t week8
# Ctrl-c to interrupt the python process; wait for the prompt to come back.
# Optionally, hard-kill anything python that's still around:
pkill -9 -f polymarket_week8_pipeline || true
exit                              # leaves tmux

# Pull the updated runbook + speed knobs from cloud-runs:
cd ~/STAT-4830-energy_arbitrage-project
git fetch origin
git reset --hard origin/cloud-runs

# Fresh tmux, fresh env, fresh RUN_TAG so the old log stays as evidence:
tmux new -s week8
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Paste the same section-4 block you used the first time (same --artifact-prefix
# and --git-push-branch — re-using them is safe; the pod's local data/processed
# files just get overwritten).
```

Within ~5–10 min you should see `top` show python at ~1500% CPU and `run_<X>_*.log` produce trial-completion lines.

---

## 11. Notes on what changed under the hood

- **Speed:** `history_fidelity` is now 60 (1-hour bars), `rolling_windows=(24, 48, 96)`, `steps_per_window=3`, and `MedianPruner` starts pruning after only 3 startup trials. Together these make a 100-trial study finish in roughly 1/10th the time of the original config.
- **Per-prefix isolation:** The `--artifact-prefix` flag is now threaded through every stage of `polymarket_week8_pipeline.py` — data ingestion, baseline, exogenous features, Optuna search, covariance diagnostics, figures, week-9 report, and the run manifest. Default is still `week8`, so single-pod runs without the flag behave exactly as before.
- **Figures correctness:** `_make_figures` now reads the constrained outputs at `<prefix><suffix>_constrained_*`, so experiments using `--macro-integration explicit` / `both` (C and D) generate the comparison figures correctly instead of falling back to the wrong files.
- **Threading:** Default env exports moved from `OMP/MKL/TORCH=1` + `--optuna-n-jobs 12` to `OMP/MKL/TORCH=4` + `--optuna-n-jobs 4` (sized for a 16-vCPU pod), with `PYTHONUNBUFFERED=1` and `python -u`. The old combination starved per-trial throughput by oversubscribing the GIL while under-using the BLAS pool; the new product (n_jobs × OMP ≈ nproc) keeps every core busy without thread thrashing.
