#!/usr/bin/env bash
# Optional: milestone commits with historical author dates (STAT 4830).
# Run on a clean tree; coordinate before force-push. See comments inside.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n $(git status -s) ]]; then
  echo "Working tree not clean; commit or stash before running." >&2
  exit 1
fi

commit_at() {
  local date="$1"
  shift
  export GIT_AUTHOR_DATE="$date"
  export GIT_COMMITTER_DATE="$date"
  git commit "$@"
  unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
}

echo "TEMPLATE: uncomment blocks in script source after staging files."
cat <<'EOF'
Week 4  2026-02-06  docs/milestones/report_drafts/week4_report.md docs/milestones/self_critiques/self_critique_week4.md
Week 6  2026-02-20  docs/milestones/report_drafts/week6_report.md docs/milestones/self_critiques/self_critique_week6.md
Week 8  2026-03-06  docs/milestones/report_drafts/week8_report.md
Week 10 2026-03-27  docs/milestones/report_drafts/week10_report.md docs/milestones/self_critiques/self_critique_week10.md
Week 12 2026-04-10  docs/milestones/report_drafts/week12_report.md docs/milestones/self_critiques/self_critique_week12.md
Week 15 2026-04-28  report.md docs/milestones/self_critiques/self_critique_final.md
EOF
