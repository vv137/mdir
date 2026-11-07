#!/bin/sh
# The checks of the `checks` job of CI (.github/workflows/ci.yml, which calls
# this script), run in the worktree that holds the script: the links, the
# decision numbers and their table, the changelog fragments, the list of the
# expected unsupported tests, the release path (#764), and the control
# characters in the text files, tabs included in Markdown (#768). No build,
# no network, no GPU; seconds. All checks run even when one fails; one line
# per check; the exit status is nonzero if any failed. The pre-push hook
# (scripts/hooks/pre-push) runs it before every push.
set -u
cd "$(dirname "$0")/.." || exit 2
# A git hook runs with GIT_DIR, GIT_INDEX_FILE, ... set, which would send the
# git commands of the checks (check-path.sh makes a repository of its own) to
# the repository being pushed.
unset $(git rev-parse --local-env-vars)

failed=0
run() {
  name=$1; shift
  if out=$("$@" 2>&1); then
    echo "ok    $name"
  else
    echo "FAIL  $name ($*)"
    printf '%s\n' "$out" | sed 's/^/      /'
    failed=$((failed + 1))
  fi
}

# A tracked text file (git grep -I skips the binary ones) holds no control
# character but the tab and the newline, and a Markdown file no tab either:
# an escape such as \b or \t in a shell or Python string turns the TeX of
# \boldsymbol, \times, \theta, \tau, or \text into bytes (#768). Prints
# file:line of each.
no_match() {
  hits=$(git grep -n -I -P "$@")
  case $? in
    0) printf '%s\n' "$hits" | cut -d: -f1,2; return 1 ;;
    1) return 0 ;;
    *) printf '%s\n' "$hits"; return 2 ;;
  esac
}
control_chars() {
  no_match '[\x00-\x08\x0b-\x1f]'; a=$?
  no_match '\t' -- '*.md'; b=$?
  [ "$a" -eq 0 ] && [ "$b" -eq 0 ]
}

run "Links" python3 scripts/check-markdown-links.py
run "Decision numbers and their table" scripts/decision-number.sh check --pending
run "Changelog fragments" python3 scripts/release/changelog.py fragments
run "The list of the expected unsupported tests" python3 test/mdir_lit.py
run "The release path" scripts/release/check-path.sh
run "Control characters in the text files" control_chars

if [ "$failed" -ne 0 ]; then
  echo "$failed check(s) failed"
  exit 1
fi
echo "all checks passed"
