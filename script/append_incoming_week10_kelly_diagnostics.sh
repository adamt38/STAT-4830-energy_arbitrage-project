#!/usr/bin/env bash
# During merge: keep HEAD aggregated week10 Kelly report and append MERGE_HEAD
# single-run body as a new ## section (demote incoming ## headings to ###).
set -euo pipefail
BRANCH="${1:?branch label e.g. cloud-runs-K10B}"
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  echo "MERGE_HEAD not set" >&2
  exit 1
fi
incoming="$(git show MERGE_HEAD:docs/week10_kelly_diagnostics_report.md)"
prefix="$(printf '%s\n' "$incoming" | grep -m1 '^\- artifact prefix:' | sed -n 's/.*`\([^`]*\)`.*/\1/p')"
if [[ -z "$prefix" ]]; then
  prefix="unknown-prefix"
fi
tmp="$(mktemp)"
git show HEAD:docs/week10_kelly_diagnostics_report.md >"$tmp"
{
  printf '\n---\n\n## Run: `%s` (source branch: `%s`)\n\n' "$prefix" "$BRANCH"
  printf '%s\n' "$incoming" | tail -n +3 | awk 'BEGIN{OFS=""} /^## / && $0 !~ /^###/ { sub(/^## /,"### ") } {print}'
} >>"$tmp"
mv "$tmp" docs/week10_kelly_diagnostics_report.md
git add docs/week10_kelly_diagnostics_report.md
