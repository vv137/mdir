#!/usr/bin/env bash
# Builds, tests, tags, and (with --publish) publishes a release, after the
# release PR of scripts/release/prepare.sh has merged:
#
#   scripts/release/publish.sh VERSION [--publish] [--container]
#   scripts/release/publish.sh VERSION --dry-run [--no-binary] [--container]
#
# From origin/main, in a fresh worktree and build tree under
# MDIR_RELEASE_DIR:
#  1. checks that CMakeLists.txt and CHANGELOG.md name VERSION, and the
#     decision numbers;
#  2. configures and builds Release, and runs the full suite on the GPU
#     MDIR_RELEASE_GPU under the locks of the machine, if it has them
#     (MDIR_RELEASE_SUITE_LOCK, then MDIR_RELEASE_GPU_LOCK),
#     with the tests of a release (lit's -Drelease=1: the whole matrix of
#     the tests that run a part of their cases by default);
#  3. builds the white paper (PANDOC, TECTONIC);
#  4. writes the assets to MDIR_RELEASE_DIR/vVERSION: the source archive
#     (git archive), the installed tree for manylinux_2_28 x86-64 with CUDA
#     (packaging/Dockerfile.manylinux; checked by check-binary.sh and run
#     on the CPU and on that GPU without CUDA_ROOT), the PDF, the release
#     notes, and SHA256SUMS; --container also builds the Docker
#     image mdir:VERSION (checked by check-image.sh; not uploaded);
#  5. makes the annotated tag vVERSION on that commit.
# Without --publish it stops there and prints the two publishing commands;
# with --publish it pushes the tag and creates the GitHub Release with the
# changelog section as its notes and the assets attached. Nothing is pushed
# unless the suite passed.
#
# --dry-run rehearses a release, also of a version that is tagged already:
# it does steps 1 to 4 in trees of its own (src-vVERSION-dry-run,
# build-vVERSION-dry-run, vVERSION-dry-run, and logs of that name) and
# stops before the tag. Nothing is tagged, pushed, or published. With it,
# --no-binary leaves out the manylinux tree and its two runs, for a machine
# without Docker.
#
# The script names no path of the machine: it reads these from the
# environment, or from the file that MDIR_ENV_FILE names
# (scripts/environment.sh, docs/environment.md), and stops at its start,
# naming each one that is missing.
#   MDIR_RELEASE_DIR   where it writes: src-vVERSION (the worktree),
#                      build-vVERSION, vVERSION (the assets), and
#                      logs/release-vVERSION-*.log
#   MDIR_LLVM          the install prefix of LLVM and MLIR
#   HDF5_PREFIX        the install prefix of HDF5
#   CUDA_ROOT          the CUDA toolkit of the build
#   MDIR_RELEASE_GPU   the index of the GPU of the suite and of the run of
#                      the manylinux tree
#   MDIR_RELEASE_GPU_LOCK, MDIR_RELEASE_SUITE_LOCK
#                      optional: lock files held, with flock, while the GPU
#                      is used and around the suite (the suite's first)
#   CMAKE, NINJA, LIT, PANDOC, TECTONIC, DOCKER
#                      optional: the programs, by name or path; cmake,
#                      ninja, pandoc, tectonic, and docker on the path by
#                      default, and the lit beside ninja, else on the path
# CC, CXX, PATH, LD_LIBRARY_PATH, LIT_OPTS, CUDA_CACHE_PATH, and
# MDIR_BENCH_DIR reach the build and the suite as the environment has them.
set -euo pipefail
usage="usage: $0 VERSION [--publish] [--container] | VERSION --dry-run [--no-binary] [--container]"
version=${1:?$usage}; shift
publish=0; container=0; dry=0; binary=1
for a in "$@"; do case $a in
  --publish) publish=1;; --container) container=1;; --dry-run) dry=1;; --no-binary) binary=0;;
  *) echo "unknown: $a" >&2; echo "$usage" >&2; exit 2;;
esac; done
(( dry || binary )) || { echo "--no-binary is for --dry-run: a release has its binary" >&2; exit 2; }
(( dry && publish )) && { echo "--dry-run publishes nothing: not with --publish" >&2; exit 2; }

. "$(dirname "$0")/../environment.sh"
need_variable MDIR_RELEASE_DIR "the directory where a release writes its worktree, build tree, assets, and logs"
need_directory MDIR_LLVM "the install prefix of LLVM and MLIR"
need_directory HDF5_PREFIX "the install prefix of HDF5"
need_directory CUDA_ROOT "the CUDA toolkit of the build"
need_variable MDIR_RELEASE_GPU "the index of the GPU that runs the suite"
need_program CMAKE cmake "CMake"
need_program NINJA ninja "ninja"
if [[ -z ${LIT:-} && -n ${NINJA:-} && -x $(dirname "$NINJA")/lit ]]; then LIT=$(dirname "$NINJA")/lit; fi
need_program LIT lit "LLVM's lit, which runs the suite"
need_program PANDOC pandoc "pandoc 3, for the white paper"
need_program TECTONIC tectonic "tectonic, for the white paper"
if (( binary || container )); then need_program DOCKER docker "Docker with BuildKit, for the manylinux tree and the image"; fi
if (( publish )); then need_program GH gh "the GitHub CLI, which creates the release"; fi
if [[ -n ${MDIR_RELEASE_GPU_LOCK:-}${MDIR_RELEASE_SUITE_LOCK:-} ]]; then need_program FLOCK flock "flock, for the locks"; fi
need_check
[[ $MDIR_RELEASE_GPU =~ ^[0-9]+$ ]] || { echo "MDIR_RELEASE_GPU is not the index of a GPU: $MDIR_RELEASE_GPU" >&2; exit 2; }
export PANDOC TECTONIC

repo=$(git rev-parse --show-toplevel)
gpu=$MDIR_RELEASE_GPU
# The lock of the suite is taken before that of the GPU, as everything else
# on a machine that has both does (docs/workflow.md, Shared machines).
gpu_locked=(); suite_locked=()
[[ -z ${MDIR_RELEASE_GPU_LOCK:-} ]] || gpu_locked=("$FLOCK" "$MDIR_RELEASE_GPU_LOCK")
[[ -z ${MDIR_RELEASE_SUITE_LOCK:-} ]] || suite_locked=("$FLOCK" "$MDIR_RELEASE_SUITE_LOCK")
tag=v$version
name=$tag; (( dry )) && name=$tag-dry-run
mkdir -p "$MDIR_RELEASE_DIR/logs"
root=$(cd "$MDIR_RELEASE_DIR" && pwd)
src=$root/src-$name
build=$root/build-$name
out=$root/$name
log=$root/logs/release-$name

git -C "$repo" fetch -q origin
commit=$(git -C "$repo" rev-parse origin/main)
(( dry )) || { git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null && { echo "$tag exists" >&2; exit 1; }; }
rm -rf "$src" && git -C "$repo" worktree add -q --detach "$src" "$commit"
trap 'git -C "$repo" worktree remove --force "$src" 2>/dev/null || true' EXIT
cd "$src"
grep -q "project(mdir VERSION $version " CMakeLists.txt || { echo "CMakeLists.txt does not say $version" >&2; exit 1; }
python3 scripts/release/changelog.py check "$version"
scripts/decision-number.sh check
[[ -f docs/release-notes/$tag.md ]] || { echo "docs/release-notes/$tag.md is missing" >&2; exit 1; }

echo "== build $commit"
"$CMAKE" -G Ninja -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$MDIR_LLVM/lib/cmake/llvm" -DMLIR_DIR="$MDIR_LLVM/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$LIT" -DHDF5_ROOT="$HDF5_PREFIX" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_INSTALL_PREFIX=/ > "$log-cmake.log"
"$NINJA" -C "$build" > "$log-build.log"

echo "== suite on GPU $gpu (log $log-check.log)"
"${suite_locked[@]}" "${gpu_locked[@]}" sh -c "nvidia-smi -i $gpu; CUDA_VISIBLE_DEVICES=$gpu LIT_OPTS=\"\${LIT_OPTS:-} -Drelease=1\" '$NINJA' -C '$build' check-mdir" > "$log-check.log" 2>&1 \
  || { grep -E '^FAIL:|Failed' "$log-check.log" >&2; echo "the suite failed: nothing is tagged" >&2; exit 1; }
grep -E '^\s+(Passed|Unsupported)' "$log-check.log"

echo "== assets in $out"
rm -rf "$out" && mkdir -p "$out"
git archive --format=tar.gz --prefix="mdir-$version/" -o "$out/mdir-$version-source.tar.gz" "$commit"
if (( binary )); then
  # The binary tarball is built for manylinux_2_28 in its own image
  # (packaging/Dockerfile.manylinux, D177), not from the host build;
  # it is checked, then run from its extracted tree, without CUDA_ROOT, on
  # the CPU and on the GPU of the suite.
  tarball=mdir-$version-manylinux_2_28_x86_64.tar.gz
  DOCKER_BUILDKIT=1 "$DOCKER" build -f packaging/Dockerfile.manylinux \
    --build-arg MDIR_GIT_COMMIT="$commit" --build-arg VERSION="$version" \
    --target tarball --output "type=local,dest=$out" . > "$log-manylinux.log" 2>&1
  tree=$(mktemp -d)
  tar -C "$tree" -xzf "$out/$tarball"
  scripts/release/check-binary.sh "$tree/mdir-$version"
  mdir="$tree/mdir-$version/bin/mdir"
  cp -r "$tree/mdir-$version/share/mdir/examples/ala3" "$tree/ala3"
  sed 's/^target *= *"GPU".*/target = "CPU"/' "$tree/ala3/1-min.toml" > "$tree/ala3/1-min-cpu.toml"
  (cd "$tree/ala3" && env -u CUDA_ROOT -u CUDA_HOME -u CUDA_PATH "$mdir" doctor --target=cpu \
    && env -u CUDA_ROOT -u CUDA_HOME -u CUDA_PATH "$mdir" run 1-min-cpu.toml) > "$log-tarball-cpu.log" 2>&1 \
    || { echo "the tarball failed on the CPU: $log-tarball-cpu.log" >&2; exit 1; }
  (cd "$tree/ala3" && "${gpu_locked[@]}" sh -c "nvidia-smi -i $gpu >/dev/null; export CUDA_VISIBLE_DEVICES=$gpu; env -u CUDA_ROOT -u CUDA_HOME -u CUDA_PATH '$mdir' doctor --target=gpu && env -u CUDA_ROOT -u CUDA_HOME -u CUDA_PATH '$mdir' run 1-min.toml") > "$log-tarball-gpu.log" 2>&1 \
    || { echo "the tarball failed on the GPU: $log-tarball-gpu.log" >&2; exit 1; }
  rm -rf "$tree"
else
  echo "== --no-binary: no manylinux tree"
fi
scripts/paper/build-pdf.sh "$build/paper" > "$log-pdf.log" 2>&1
cp "$build/paper/main.pdf" "$out/mdir-$version-white-paper.pdf"
cp "docs/release-notes/$tag.md" "$out/"
python3 scripts/release/changelog.py notes "$version" > "$out/notes.md"
if (( container )); then
  "$DOCKER" build -f packaging/Dockerfile -t "mdir:$version" --build-arg MDIR_GIT_COMMIT="$commit" . > "$log-docker.log" 2>&1
  scripts/release/check-image.sh "mdir:$version"
fi
(cd "$out" && sha256sum mdir-* "$tag.md" > SHA256SUMS)
ls -la "$out"

if (( dry )); then
  echo "== dry run of $tag at $commit: nothing is tagged"
  exit 0
fi
git -C "$repo" tag -a "$tag" "$commit" -m "MDIR $version"
echo "== tagged $tag at $commit"
if (( publish )); then
  git -C "$repo" push origin "$tag"
  "$GH" release create "$tag" --repo vv137/mdir --title "MDIR $version" \
    --notes-file "$out/notes.md" "$out"/mdir-* "$out/$tag.md" "$out/SHA256SUMS"
else
  echo "To publish:"
  echo "  git -C $repo push origin $tag"
  echo "  gh release create $tag --repo vv137/mdir --title 'MDIR $version' --notes-file $out/notes.md $out/mdir-* $out/$tag.md $out/SHA256SUMS"
fi
