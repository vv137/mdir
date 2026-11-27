#!/usr/bin/env bash
# Decision numbers are given at merge time. A branch writes its decision as
# a label, D[<label>] (lowercase letters, digits, and dashes), in its file
# docs/decisions/<label>.md, comments, docs, and the paper; the merger runs
#
#   scripts/decision-number.sh assign <label>   # D[<label>] -> next free Dnnn
#   scripts/decision-number.sh check            # no labels, no number twice
#
# on the merged tree. Between a merge and its number, `check --pending`
# (run on every push to main) accepts a label that has its file in
# docs/decisions/; a tag runs the strict `check`. `assign` renames the file
# to docs/decisions/Dnnn.md, dates it, rewrites every tracked file, and
# writes the table of docs/decisions.md (scripts/decisions-table.py); it
# leaves the change for the merger to commit.
#
# A label is also found in the escaped forms that tests and documents write
# (#1010): a backslash run before either bracket (D\[x\] in a regular
# expression or a Markdown link, D\\[x\\] in a string that is not raw), the
# bracket expressions D[[]x[]], and FileCheck's D{{\[}}x{{\]}} and
# D{{[[]}}x{{[]]}}. `assign` rewrites each to Dnnn, then refuses, naming
# file and line, anything still left that writes the label between
# brackets; `check` refuses these forms as it refuses D[x].
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Numbers handed out before labels were introduced, to work still open.
# Remove a number when its decision is merged.
reserved=()

decisions=docs/decisions.md
directory=docs/decisions
# Before the D: the start of a line, a character not of a word, or an
# escape such as \b or \n.
boundary='(^|[^A-Za-z0-9_]|\\[A-Za-z])'
open='(\\*\[|\[\[\]|\{\{(\\*\[|\[\[\])\}\})'
close='(\\*\]|\[\]\]|\{\{(\\*\]|\[\]\])\}\})'
label_pattern="${boundary}D${open}[a-z0-9-]+${close}"
# Groups: 1 the boundary, 2 and 3 the opening bracket, 4 the label.
label_of="s/^${boundary}D${open}([a-z0-9-]+)${close}\$/\\4/"

next_number() {
  local highest
  highest=$(git ls-files "$directory" | grep -oE '/D[0-9]+[a-z]?\.md$' |
    grep -oE '[0-9]+' | sort -n | tail -1)
  for r in "${reserved[@]}"; do
    (( r > highest )) && highest=$r
  done
  echo $((highest + 1))
}

check() {
  local pending=${1:-} status=0 files labels label name rowless=()
  files=$(git grep -lE "$label_pattern" -- . ':!scripts/decision-number.sh' || true)
  if [[ -n "$files" && -z "$pending" ]]; then
    echo "decision labels without a number:" >&2
    git grep -nE "$label_pattern" -- . ':!scripts/decision-number.sh' >&2
    status=1
  elif [[ -n "$files" ]]; then
    # A merged decision waits for its number; its label needs a file.
    labels=$(git grep -hoE "$label_pattern" -- . ':!scripts/decision-number.sh' |
      sed -E "$label_of" | sort -u)
    for name in $labels; do
      if [[ -f "$directory/$name.md" ]]; then
        echo "waiting for a number: D[$name]"
      else
        rowless+=("$name")
      fi
    done
    if (( ${#rowless[@]} )); then
      # A label numbered already, or never given a file, in any of its forms.
      echo "decision labels without a file in $directory/:" >&2
      for name in "${rowless[@]}"; do
        git grep -nE "${boundary}D${open}${name}${close}" -- . \
          ':!scripts/decision-number.sh' >&2
      done
      status=1
    fi
  fi
  # A decision is a file; a row of the earlier form in the log is an error.
  if grep -nE '^\| D([0-9]+[a-z]?|\[[a-z0-9-]+\]) \|' "$decisions" >&2; then
    echo "decision rows in $decisions: a decision is a file in $directory/" >&2
    status=1
  fi
  if ! python3 scripts/decisions-table.py check; then
    status=1
  fi
  if ! python3 scripts/decision-number-anchors.py check; then
    status=1
  fi
  return $status
}

case "${1:-}" in
  assign)
    label=${2:?usage: $0 assign <label>}
    [[ "$label" =~ ^[a-z0-9-]+$ ]] || { echo "bad label: $label" >&2; exit 2; }
    [[ -f "$directory/$label.md" ]] || { echo "no $directory/$label.md" >&2; exit 1; }
    number=$(next_number)
    # Rewrite links using the headings' GitHub anchors before changing their text.
    python3 scripts/decision-number-anchors.py assign "$label" "$number"
    git mv "$directory/$label.md" "$directory/D$number.md"
    sed -i "s/^Recorded at its merge\.$/Recorded $(date -u +%F)./" "$directory/D$number.md"
    files=$(git grep -lE "${boundary}D${open}${label}${close}" -- . ':!scripts/decision-number.sh')
    # shellcheck disable=SC2086
    sed -i -E "s/${boundary}D${open}${label}${close}/\\1D$number/g" $files
    python3 scripts/decisions-table.py write
    # Any form not rewritten above: D, up to eight brackets, backslashes, or
    # braces, the label, and one of them again.
    if git grep -nE "D[][\\{}]{1,8}${label}[][\\{}]" -- . \
        ':!scripts/decision-number.sh' >&2; then
      echo "D[$label] is left above in a form that assign does not rewrite:" \
        "write it as D$number before committing" >&2
      exit 1
    fi
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
