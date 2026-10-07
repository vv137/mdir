#!/usr/bin/env bash
# Decision numbers are given at merge time. A branch writes its decision as
# a label, D[<label>] (lowercase letters, digits, and dashes), in
# docs/decisions.md, comments, docs, and the paper; the merger runs
#
#   scripts/decision-number.sh assign <label>   # D[<label>] -> next free Dnnn
#   scripts/decision-number.sh check            # no labels, no number twice
#
# on the merged tree. Between a merge and its number, `check --pending`
# (run on every push to main) accepts a label that has its row in
# docs/decisions.md; a tag runs the strict `check`. `assign` rewrites every tracked file and leaves the
# change for the merger to commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Numbers handed out before labels were introduced, to work still open.
# Remove a number when its decision is merged.
reserved=()

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
  local pending=${1:-} status=0 files labels label duplicates rowless=()
  files=$(git grep -lE "$label_pattern" -- . ':!scripts/decision-number.sh' || true)
  if [[ -n "$files" && -z "$pending" ]]; then
    echo "decision labels without a number:" >&2
    git grep -nE "$label_pattern" -- . ':!scripts/decision-number.sh' >&2
    status=1
  elif [[ -n "$files" ]]; then
    # A merged decision waits for its number; its label needs a row.
    labels=$(git grep -hoE "$label_pattern" -- . ':!scripts/decision-number.sh' |
      grep -oE 'D\[[a-z0-9-]+\]' | sort -u)
    for label in $labels; do
      if grep -qF "| $label |" "$decisions"; then
        echo "waiting for a number: $label"
      else
        rowless+=("$label")
      fi
    done
    if (( ${#rowless[@]} )); then
      echo "decision labels without a row in $decisions:" >&2
      for label in "${rowless[@]}"; do
        git grep -nF "$label" -- . ':!scripts/decision-number.sh' >&2
      done
      status=1
    fi
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
    case "${2:-}" in
      '') check && echo "decision numbers: ok" ;;
      --pending) check pending && echo "decision numbers: ok" ;;
      *) echo "usage: $0 check [--pending]" >&2; exit 2 ;;
    esac
    ;;
  *)
    echo "usage: $0 assign <label> | check [--pending]" >&2
    exit 2
    ;;
esac
