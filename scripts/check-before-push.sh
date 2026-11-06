#!/bin/sh
# The checks of the `checks` job of CI (.github/workflows/ci.yml, which calls
# this script), run in the worktree that holds the script: the links, the
# decision numbers and their table, the changelog fragments, the list of the
# expected unsupported tests, and the release path (#764). No build, no
# network, no GPU; seconds. All checks run even when one fails; one line
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

run "Links" python3 scripts/check-markdown-links.py
run "Decision numbers and their table" scripts/decision-number.sh check --pending
run "Changelog fragments" python3 scripts/release/changelog.py fragments
run "The list of the expected unsupported tests" python3 test/mdir_lit.py
run "The release path" scripts/release/check-path.sh

if [ "$failed" -ne 0 ]; then
  echo "$failed check(s) failed"
  exit 1
fi
echo "all checks passed"
