#!/usr/bin/env bash
# The edits of a release to the tree (docs/workflow.md, Releases):
#
#   scripts/release/bump.sh VERSION DATE [--pending]
#
# sets project(mdir VERSION ...) in CMakeLists.txt and the version of the
# package that the Python example declares, moves [Unreleased] of
# CHANGELOG.md and the fragments of changelog.d/ under [VERSION] - DATE,
# and checks the section and the decision numbers. It commits nothing.
# prepare.sh runs it on its release branch, and check-path.sh on a copy of
# the tree with the next version (#609), where --pending accepts labels that
# wait for their numbers.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
version=${1:?usage: $0 VERSION DATE [--pending]}
date=${2:?usage: $0 VERSION DATE [--pending]}
pending=${3:-}
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $version" >&2; exit 2; }
[[ -z "$pending" || "$pending" == --pending ]] || { echo "usage: $0 VERSION DATE [--pending]" >&2; exit 2; }

sed -i -E "s/^project\(mdir VERSION [0-9]+\.[0-9]+\.[0-9]+/project(mdir VERSION $version/" CMakeLists.txt
grep -q "project(mdir VERSION $version " CMakeLists.txt || { echo "CMakeLists.txt: version not set" >&2; exit 1; }
# The package that the Python example declares is the release's (D228).
sed -i -E "s/\"mdir\[cuda\]==[0-9]+\.[0-9]+\.[0-9]+\"/\"mdir[cuda]==$version\"/" examples/ala3/run.py
grep -q "\"mdir\[cuda\]==$version\"" examples/ala3/run.py || { echo "examples/ala3/run.py: version not set" >&2; exit 1; }
python3 scripts/release/changelog.py release "$version" "$date"
python3 scripts/release/changelog.py check "$version"
scripts/decision-number.sh check $pending
