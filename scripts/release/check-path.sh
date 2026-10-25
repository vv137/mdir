#!/usr/bin/env bash
# The release path, exercised on a copy of the tree before release day
# (#609; docs/workflow.md, Releases):
#
#   scripts/release/check-path.sh [VERSION]
#
# On a copy of HEAD it runs the edits of prepare.sh (bump.sh) with VERSION,
# by default the next minor version, and a dummy fragment, so that the move
# of the changelog always has an entry to move; then it checks:
#
#   - the section of VERSION and its entry count (changelog.py release);
#   - the notes of the GitHub Release (changelog.py notes --absolute): no
#     relative link left, and every link into the repository names a file;
#   - the links of the moved changelog (check-markdown-links.py);
#   - the previous version, pinned in a test or in code outside comments,
#     where the release would leave it stale (a FileCheck pattern of
#     checkpoint-contract.test, which held the version, failed the merge of
#     0.2.0).
#
# It needs no build and no GPU, and takes seconds; CI runs it on every push
# to main. Nothing of the tree is changed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
previous=$(sed -nE 's/^project\(mdir VERSION ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' CMakeLists.txt)
[[ -n "$previous" ]] || { echo "CMakeLists.txt: no version" >&2; exit 1; }
if [[ $# -ge 1 ]]; then
  version=$1
else
  IFS=. read -r major minor _ <<<"$previous"
  version=$major.$((minor + 1)).0
fi
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $version" >&2; exit 2; }
tag=v$version
status=0

copy=$(mktemp -d "${TMPDIR:-/tmp}/mdir-release-path.XXXXXX")
# The copy is removed however the check ends, and its removal never decides
# the result: an error of rm under `set -e` in the trap turned a check that
# had printed "release path: ok" into exit 1 (run 40238707543).
cleanup() {
  wait
  rm -rf "$copy" 2>/dev/null || { sleep 1; rm -rf "$copy" 2>/dev/null; } || true
}
trap cleanup EXIT
git archive HEAD | tar -x -C "$copy"
cd "$copy"
git init -q
# No automatic gc or maintenance: `git commit` of a whole tree of loose
# objects started a detached `gc --auto`, which was still writing into .git
# when the trap removed it ("Directory not empty").
git config gc.auto 0
git config gc.autoDetach false
git config maintenance.auto false
git add -A
git -c user.name=check -c user.email=check@invalid commit -q -m "HEAD"

mkdir -p changelog.d
cat > changelog.d/zzz-release-path-check.md <<'EOF'
### Added

- An entry of the check of the release path, which no release keeps.
EOF
before=$(python3 scripts/release/changelog.py unreleased | grep -c '^- ')
scripts/release/bump.sh "$version" "$(date -u +%F)" --pending > bump.log 2>&1 || {
  cat bump.log >&2
  echo "release path: the edits of prepare.sh failed" >&2
  exit 1
}
after=$(python3 scripts/release/changelog.py notes "$version" | grep -c '^- ')
echo "release path: $previous -> $version, $before entries moved, $after in [$version]"
[[ -f changelog.d/zzz-release-path-check.md ]] && { echo "the fragments were not removed" >&2; status=1; }

# The notes of the GitHub Release.
python3 scripts/release/changelog.py notes "$version" --absolute "$tag" > notes.md
python3 - "$tag" <<'EOF' || status=1
import os, re, sys
tag = sys.argv[1]
text = open("notes.md").read()
prefix = re.escape("https://github.com/vv137/mdir/")
bad = 0
for target in re.findall(r"\]\(([^)\s]+)\)", text):
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        print(f"notes: a relative link: {target}", file=sys.stderr)
        bad += 1
        continue
    m = re.match(prefix + r"(?:blob|tree)/" + re.escape(tag) + r"/([^#]+)", target)
    if m and not os.path.exists(m.group(1)):
        print(f"notes: no such file at {tag}: {m.group(1)}", file=sys.stderr)
        bad += 1
print(f"release path: notes, {len(text.splitlines())} lines, {bad} bad link(s)")
sys.exit(1 if bad else 0)
EOF

python3 scripts/check-markdown-links.py . | tail -1
python3 scripts/check-markdown-links.py . > /dev/null || status=1

# The previous version, pinned outside comments and released notes.
python3 - "$previous" <<'EOF' || status=1
import re, subprocess, sys
previous = sys.argv[1]
files = subprocess.run(["git", "grep", "-lF", previous, "--", "test", "lib",
                        "include", "tools", "python", "runtime", "scripts",
                        ":!scripts/release/version-pins.txt"],
                       capture_output=True, text=True).stdout.split()
directive = re.compile(r"\b[A-Z][A-Z0-9_-]*(-NEXT|-SAME|-NOT|-DAG|-LABEL|-EMPTY)?:")
# The uses that stay right (scripts/release/version-pins.txt).
pins = []
for line in open("scripts/release/version-pins.txt"):
    line = line.strip()
    if line and not line.startswith("#"):
        version, path, text = line.split(" ", 2)
        pins.append((version, path, text))
found = []
for path in files:
    for number, line in enumerate(open(path, errors="replace"), 1):
        if previous not in line:
            continue
        code = line.strip()
        if path.startswith("test/") and path.endswith((".test", ".mlir", ".toml")):
            # A lit test: only its directives (RUN, CHECK, ...) are read.
            if not directive.search(code):
                continue
        elif code.startswith(("#", "//", "///", "*", "/*", ";")):
            continue
        if any(v == previous and p == path and t in line for v, p, t in pins):
            continue
        found.append(f"{path}:{number}: {code[:120]}")
for line in found:
    print(f"the previous version {previous}: {line}", file=sys.stderr)
print(f"release path: {len(found)} use(s) of {previous} outside comments")
sys.exit(1 if found else 0)
EOF

[[ $status -eq 0 ]] && echo "release path: ok" || echo "release path: FAILED" >&2
exit $status
