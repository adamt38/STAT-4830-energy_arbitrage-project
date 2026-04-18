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

Inside tmux:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"
```

`Ctrl-b d` to detach; `tmux attach -t week8` to reattach later.

---

## 4. Experiments — one per pod, all parallel-safe

Every command below sets two things that make parallel execution safe:

- `--artifact-prefix week8_<X>` — every CSV/JSON/PNG/manifest this run writes is stamped with that prefix, so two pods never touch the same filename.
- `--git-push-branch cloud-runs-<X>` — results are pushed to a branch named after the experiment, so two pods never push to the same ref.

Tune `--optuna-n-jobs` to the vCPU count on the pod (with `OMP_/MKL_/TORCH_NUM_THREADS=1` already exported, you can safely match it to the core count).

### Experiment A — Baseline Optuna (rescale macro, no ETF)

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_A \
  --macro-integration rescale \
  --optuna-n-jobs 12 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-A \
  --git-commit-message "A: rescale baseline (week8_A)" \
  2>&1 | tee "run_A_${RUN_TAG}.log"
```

### Experiment B — Rescale macro + ETF tracking

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_B \
  --macro-integration rescale \
  --etf-tracking \
  --optuna-n-jobs 12 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-B \
  --git-commit-message "B: rescale + ETF (week8_B)" \
  2>&1 | tee "run_B_${RUN_TAG}.log"
```

### Experiment C — Explicit macro + ETF tracking

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_C \
  --macro-integration explicit \
  --etf-tracking \
  --optuna-n-jobs 12 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-C \
  --git-commit-message "C: explicit + ETF (week8_C)" \
  2>&1 | tee "run_C_${RUN_TAG}.log"
```

### Experiment D — Both macro modes combined + ETF tracking

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_D \
  --macro-integration both \
  --etf-tracking \
  --optuna-n-jobs 12 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-D \
  --git-commit-message "D: both + ETF (week8_D)" \
  2>&1 | tee "run_D_${RUN_TAG}.log"
```

### Experiment E — Three separate macro studies (rescale, explicit, both) + ETF

Runs three back-to-back Optuna studies on a single pod with distinct constrained-output suffixes (`week8_E`, `week8_E_macro_explicit`, `week8_E_macro_both`).

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_E \
  --macro-modes rescale,explicit,both \
  --etf-tracking \
  --optuna-n-jobs 12 \
  --optuna-trials 100 \
  --git-commit-and-push \
  --git-push-branch cloud-runs-E \
  --git-commit-message "E: three-macro + ETF (week8_E)" \
  2>&1 | tee "run_E_${RUN_TAG}.log"
```

### Experiment F — Joint macro search + ETF

One Optuna study with categorical `macro_mode` and conditional `regime_k` / `lambda_macro_explicit`. Do **not** combine with `--macro-modes`.

```bash
python script/polymarket_week8_pipeline.py \
  --artifact-prefix week8_F \
  --joint-macro-mode-search \
  --etf-tracking \
  --optuna-n-jobs 12 \
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
# inside tmux:
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
RUN_TAG="$(date -u +%Y%m%dT%H%MZ)"

# Pick ONE block from section 4 (the --artifact-prefix and --git-push-branch
# values must be unique per pod). Paste it and let the run finish.
```

That's the whole loop. Six pods, six commands from section 4, one merge sequence in section 8 → all six experiments land cleanly on `cloud-runs`.

---

## 10. Notes on what changed under the hood

- **Speed:** `history_fidelity` is now 60 (1-hour bars), `rolling_windows=(24, 48, 96)`, `steps_per_window=3`, and `MedianPruner` starts pruning after only 3 startup trials. Together these make a 100-trial study finish in roughly 1/10th the time of the original config.
- **Per-prefix isolation:** The `--artifact-prefix` flag is now threaded through every stage of `polymarket_week8_pipeline.py` — data ingestion, baseline, exogenous features, Optuna search, covariance diagnostics, figures, week-9 report, and the run manifest. Default is still `week8`, so single-pod runs without the flag behave exactly as before.
- **Figures correctness:** `_make_figures` now reads the constrained outputs at `<prefix><suffix>_constrained_*`, so experiments using `--macro-integration explicit` / `both` (C and D) generate the comparison figures correctly instead of falling back to the wrong files.
