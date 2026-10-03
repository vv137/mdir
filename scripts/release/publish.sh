#!/usr/bin/env bash
# Builds, tests, tags, and (with --publish) publishes a release, after the
# release PR of scripts/release/prepare.sh has merged:
#
#   scripts/release/publish.sh VERSION [--publish] [--container]
#
# From origin/main, in a fresh worktree and build tree under ~/build:
#  1. checks that CMakeLists.txt and CHANGELOG.md name VERSION, and the
#     decision numbers;
#  2. configures and builds Release, and runs the full suite on GPU 1 under
#     its lock (MDIR_RELEASE_GPU_LOCK, default ~/build/locks/gpu1.lock);
#  3. builds the white paper (PANDOC, default ~/opt/pandoc/bin/pandoc);
#  4. writes the assets to ~/build/release/vVERSION: the source archive
#     (git archive), the installed tree for Linux x86-64 with CUDA, the PDF,
#     the release notes, and SHA256SUMS; --container also builds the Docker
#     image mdir:VERSION (not uploaded);
#  5. makes the annotated tag vVERSION on that commit.
# Without --publish it stops there and prints the two publishing commands;
# with --publish it pushes the tag and creates the GitHub Release with the
# changelog section as its notes and the assets attached. Nothing is pushed
# unless the suite passed.
set -euo pipefail
version=${1:?usage: $0 VERSION [--publish] [--container]}; shift
publish=0; container=0
for a in "$@"; do case $a in --publish) publish=1;; --container) container=1;; *) echo "unknown: $a" >&2; exit 2;; esac; done
repo=$(git rev-parse --show-toplevel)
ninja=${NINJA:-/store/vv137/opt/llvm/tools-venv/bin/ninja}
llvm=${MDIR_LLVM:-/store/vv137/opt/llvm/23.1.2}
lock=${MDIR_RELEASE_GPU_LOCK:-$HOME/build/locks/gpu1.lock}
src=$HOME/build/release/src-v$version
build=$HOME/build/release/build-v$version
out=$HOME/build/release/v$version
log=$HOME/build/logs/release-v$version
tag=v$version

git -C "$repo" fetch -q origin
commit=$(git -C "$repo" rev-parse origin/main)
git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null && { echo "$tag exists" >&2; exit 1; }
rm -rf "$src" && git -C "$repo" worktree add -q --detach "$src" "$commit"
trap 'git -C "$repo" worktree remove --force "$src" 2>/dev/null || true' EXIT
cd "$src"
grep -q "project(mdir VERSION $version " CMakeLists.txt || { echo "CMakeLists.txt does not say $version" >&2; exit 1; }
python3 scripts/release/changelog.py check "$version"
scripts/decision-number.sh check
[[ -f docs/release-notes/$tag.md ]] || { echo "docs/release-notes/$tag.md is missing" >&2; exit 1; }

echo "== build $commit"
cmake -G Ninja -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$llvm/lib/cmake/llvm -DMLIR_DIR=$llvm/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(dirname "$ninja")/lit -DHDF5_ROOT=$HOME/opt/hdf5/1.14.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.4 -DCMAKE_MAKE_PROGRAM="$ninja" \
  -DCMAKE_INSTALL_PREFIX=/ > "$log-cmake.log"
"$ninja" -C "$build" > "$log-build.log"

echo "== suite on GPU 1 (log $log-check.log)"
flock "$lock" sh -c "nvidia-smi -i 1; CUDA_VISIBLE_DEVICES=1 '$ninja' -C '$build' check-mdir" > "$log-check.log" 2>&1 \
  || { grep -E '^FAIL:|Failed' "$log-check.log" >&2; echo "the suite failed: nothing is tagged" >&2; exit 1; }
grep -E '^\s+(Passed|Unsupported)' "$log-check.log"

echo "== assets in $out"
rm -rf "$out" && mkdir -p "$out"
git archive --format=tar.gz --prefix="mdir-$version/" -o "$out/mdir-$version-source.tar.gz" "$commit"
stage=$(mktemp -d)
DESTDIR="$stage/mdir-$version" cmake --install "$build" > "$log-install.log"
tar -C "$stage" -czf "$out/mdir-$version-linux-x86_64-cuda.tar.gz" "mdir-$version"
rm -rf "$stage"
PANDOC=${PANDOC:-$HOME/opt/pandoc/bin/pandoc} scripts/paper/build-pdf.sh "$build/paper" > "$log-pdf.log" 2>&1
cp "$build/paper/main.pdf" "$out/mdir-$version-white-paper.pdf"
cp "docs/release-notes/$tag.md" "$out/"
python3 scripts/release/changelog.py notes "$version" > "$out/notes.md"
if (( container )); then
  docker build -f packaging/Dockerfile -t "mdir:$version" --build-arg MDIR_GIT_COMMIT="$commit" . > "$log-docker.log" 2>&1
fi
(cd "$out" && sha256sum mdir-* "$tag.md" > SHA256SUMS)
ls -la "$out"

git -C "$repo" tag -a "$tag" "$commit" -m "MDIR $version"
echo "== tagged $tag at $commit"
if (( publish )); then
  git -C "$repo" push origin "$tag"
  gh release create "$tag" --repo vv137/mdir --title "MDIR $version" \
    --notes-file "$out/notes.md" "$out"/mdir-* "$out/$tag.md" "$out/SHA256SUMS"
else
  echo "To publish:"
  echo "  git -C $repo push origin $tag"
  echo "  gh release create $tag --repo vv137/mdir --title 'MDIR $version' --notes-file $out/notes.md $out/mdir-* $out/$tag.md $out/SHA256SUMS"
fi
