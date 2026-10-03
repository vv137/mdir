#!/usr/bin/env bash
# Decision numbers are given at merge time. A branch writes its decision as
# a label, D[<label>] (lowercase letters, digits, and dashes), in
# docs/decisions.md, comments, docs, and the paper; the merger runs
#
#   scripts/decision-number.sh assign <label>   # D[<label>] -> next free Dnnn
#   scripts/decision-number.sh check            # no labels, no number twice
#
# on the merged tree. `assign` rewrites every tracked file and leaves the
# change for the merger to commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Numbers handed out before labels were introduced, to work still open.
# Remove a number when its decision is merged.
reserved=(161)

decisions=docs/decisions.md
label_pattern='(^|[^A-Za-z0-9_])D\[[a-z0-9-]+\]'

next_number() {
  local highest
  highest=$(grep -oE '^\| D[0-9]+' "$decisions" | grep -oE '[0-9]+' | sort -n | tail -1)
  for r in "${reserved[@]}"; do
    (( r > highest )) && highest=$r
  done
  echo $((highest + 1))
}

check() {
  local status=0 files duplicates
  files=$(git grep -lE "$label_pattern" -- . ':!scripts/decision-number.sh' || true)
  if [[ -n "$files" ]]; then
    echo "decision labels without a number:" >&2
    git grep -nE "$label_pattern" -- . ':!scripts/decision-number.sh' >&2
    status=1
  fi
  duplicates=$(grep -oE '^\| D[0-9]+[a-z]? ' "$decisions" | sort | uniq -d)
  if [[ -n "$duplicates" ]]; then
    echo "decision numbers given twice in $decisions:" >&2
    echo "$duplicates" >&2
    status=1
  fi
  return $status
}

case "${1:-}" in
  assign)
    label=${2:?usage: $0 assign <label>}
    [[ "$label" =~ ^[a-z0-9-]+$ ]] || { echo "bad label: $label" >&2; exit 2; }
    files=$(git grep -lE "(^|[^A-Za-z0-9_])D\\[$label\\]" -- . ':!scripts/decision-number.sh' || true)
    [[ -n "$files" ]] || { echo "no D[$label] in the tree" >&2; exit 1; }
    number=$(next_number)
    # shellcheck disable=SC2086
    sed -i -E "s/(^|[^A-Za-z0-9_])D\\[$label\\]/\\1D$number/g" $files
    echo "D[$label] -> D$number in:"
    echo "$files" | sed 's/^/  /'
    ;;
  check)
    check && echo "decision numbers: ok"
    ;;
  *)
    echo "usage: $0 assign <label> | check" >&2
    exit 2
    ;;
esac
