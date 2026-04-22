#!/usr/bin/env bash
# Poll origin for <branch> to appear, then `prime pods terminate <pod_id>`.
#
# Usage:
#   bash script/auto_terminate_on_push.sh <pod_id> <expected_branch> [check_interval_sec]
#
# Example:
#   bash script/auto_terminate_on_push.sh 7d9574ffe0c34ed6af777db2fe7cf38b cloud-runs-K10E
#
# Runs in the foreground — prefix with `nohup ... &` to keep it going if you
# close the shell, or put it in its own tmux pane on the laptop.

set -euo pipefail

POD_ID="${1:-}"
BRANCH="${2:-}"
INTERVAL="${3:-180}"

if [[ -z "$POD_ID" || -z "$BRANCH" ]]; then
  echo "usage: $0 <pod_id> <expected_branch> [check_interval_sec]" >&2
  exit 2
fi

if ! command -v prime >/dev/null 2>&1; then
  echo "prime CLI not found on PATH. Activate your laptop venv first." >&2
  exit 2
fi

echo "[auto-term] watching origin/$BRANCH for first appearance"
echo "[auto-term] will terminate pod $POD_ID on detection (poll every ${INTERVAL}s)"

while true; do
  if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    sha="$(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')"
    echo "[auto-term] origin/$BRANCH = $sha — terminating pod $POD_ID"
    prime pods terminate "$POD_ID"
    echo "[auto-term] termination request sent. Run 'prime pods list' to confirm."
    exit 0
  fi
  printf "[auto-term] %s — branch not yet pushed; sleeping %ss\n" "$(date -u +%H:%M:%SZ)" "$INTERVAL"
  sleep "$INTERVAL"
done
