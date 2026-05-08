#!/usr/bin/env bash
# Merge remaining tips from docs/branch_merge_order.txt into main.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while read -r t; do
  [[ -z "${t// }" ]] && continue
  if git merge-base --is-ancestor "origin/$t" HEAD 2>/dev/null; then
    echo "SKIP already merged: $t"
    continue
  fi
  echo "==== MERGING $t ===="
  if git merge "origin/$t" --no-ff -m "Merge branch '$t' into main" 2>&1; then
    :
  else
    unmerged="$(git diff --name-only --diff-filter=U || true)"
    sorted="$(printf '%s\n' "$unmerged" | sort | paste -sd, -)"
    case "$sorted" in
      docs/week9_diagnostics_report.md)
        bash script/append_incoming_week9_diagnostics.sh "$t"
        GIT_EDITOR=true git merge --continue
        ;;
      docs/week10_kelly_diagnostics_report.md)
        chmod +x script/append_incoming_week10_kelly_diagnostics.sh 2>/dev/null || true
        bash script/append_incoming_week10_kelly_diagnostics.sh "$t"
        GIT_EDITOR=true git merge --continue
        ;;
      *)
        echo "Unexpected merge conflicts for $t:" >&2
        echo "$unmerged" >&2
        exit 2
        ;;
    esac
  fi
  if ! .venv/bin/python -m pytest tests/ -v --tb=short >/tmp/pytest_merge.txt 2>&1; then
    echo "PYTEST FAILED after $t" >&2
    cat /tmp/pytest_merge.txt >&2
    exit 3
  fi
  tail -3 /tmp/pytest_merge.txt
  echo "OK $t"
done < docs/branch_merge_order.txt
