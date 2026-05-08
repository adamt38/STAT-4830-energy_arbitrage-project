#!/usr/bin/env bash
# During an in-progress merge with conflicts only in docs/week9_diagnostics_report.md:
# keep the current mainline aggregated report and append the incoming branch's
# single-run report as a new ## section (headings demoted by one level).
set -euo pipefail
BRANCH="${1:?branch label for log e.g. cloud-runs-S1}"
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  echo "MERGE_HEAD not set; run from repo root during a merge" >&2
  exit 1
fi
incoming="$(git show MERGE_HEAD:docs/week9_diagnostics_report.md)"
prefix="$(printf '%s\n' "$incoming" | grep -m1 '^\- artifact prefix:' | sed -n 's/.*`\([^`]*\)`.*/\1/p')"
if [[ -z "$prefix" ]]; then
  prefix="unknown-prefix"
fi
tmp="$(mktemp)"
git show HEAD:docs/week9_diagnostics_report.md >"$tmp"
{
  printf '\n---\n\n## Run: `%s` (source branch: `%s`)\n\n' "$prefix" "$BRANCH"
  printf '%s\n' "$incoming" | tail -n +3 | awk 'BEGIN{OFS=""} /^## / && $0 !~ /^###/ { sub(/^## /,"### ") } {print}'
} >>"$tmp"
mv "$tmp" docs/week9_diagnostics_report.md
git add docs/week9_diagnostics_report.md
