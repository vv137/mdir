#!/usr/bin/env bash
# Opens the pull request of a release (docs/workflow.md, Releases):
#
#   scripts/release/prepare.sh VERSION [DATE]
#
# On a branch release/vVERSION from origin/main: sets project(mdir VERSION
# ...) in CMakeLists.txt, moves [Unreleased] of CHANGELOG.md under
# [VERSION] - DATE (today by default), checks the decision numbers, commits,
# pushes the branch, and opens a draft PR. The PR goes through the labels
# like any other; after its merge, scripts/release/publish.sh builds,
# tests, tags, and publishes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
version=${1:?usage: $0 VERSION [DATE]}
date=${2:-$(date +%F)}
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $version" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "the tree has changes" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/v$version" >/dev/null && { echo "v$version is tagged already" >&2; exit 1; }

git fetch -q origin
git checkout -q -b "release/v$version" origin/main
sed -i -E "s/^project\(mdir VERSION [0-9]+\.[0-9]+\.[0-9]+/project(mdir VERSION $version/" CMakeLists.txt
grep -q "project(mdir VERSION $version " CMakeLists.txt || { echo "CMakeLists.txt: version not set" >&2; exit 1; }
# The package that the Python example declares is the release's (D228).
sed -i -E "s/\"mdir\[cuda\]==[0-9]+\.[0-9]+\.[0-9]+\"/\"mdir[cuda]==$version\"/" examples/ala3/run.py
grep -q "\"mdir\[cuda\]==$version\"" examples/ala3/run.py || { echo "examples/ala3/run.py: version not set" >&2; exit 1; }
python3 scripts/release/changelog.py release "$version" "$date"
python3 scripts/release/changelog.py check "$version"
scripts/decision-number.sh check
git add CMakeLists.txt CHANGELOG.md examples/ala3/run.py
git commit -q -m "Release $version: the version and the changelog"
git push -q -u origin "release/v$version"
gh pr create --draft --base main --head "release/v$version" \
  --title "Release $version" \
  --body "The release of $version: \`project(mdir VERSION $version)\` and the changelog's [Unreleased] moved under [$version] - $date.

Before merge, review the [$version] section of CHANGELOG.md (Known limitations in particular) and \`docs/release-notes/v$version.md\`, and push any edits to this branch. After the merge, \`scripts/release/publish.sh $version\` builds from the merged main in a fresh tree, runs the full suite on a GPU, builds the PDF and the archives, and tags; \`--publish\` pushes the tag and creates the GitHub Release."
echo "Edit docs/release-notes/v$version.md on this branch if needed, then mark the PR ready."
