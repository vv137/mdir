#!/usr/bin/env bash
# The one entry point for the container and binary builds of MDIR (#639):
#
#   scripts/packaging/build.sh TARGET [options]
#
# TARGET (packaging/README.md, Targets):
#   runtime      the runtime image, packaging/Dockerfile's last stage, tagged
#                mdir:VERSION (or --tag); never pushed by this script
#   tarball      the release tarball for manylinux_2_28,
#                packaging/Dockerfile.manylinux --target tarball, into --out
#   wheels       the Python wheels for manylinux_2_28, the same file
#                --target wheels, into --out
#   ci           the image of the CPU tier of CI, packaging/Dockerfile
#                --target ci, which builds LLVM in one go, tagged
#                ghcr.io/vv137/mdir-ci:llvm-23.1.2-<hash>
#   ci-prebuilt  the same image over an install of LLVM in ci-llvm/ of the
#                source (the stage ci-prebuilt; .github/workflows/ci-image.yml)
#
# Options:
#   --source DIR   the tree to build, as it is (default: this repository)
#   --clone BRANCH build a fresh clone of BRANCH of vv137/mdir instead, in
#                  WORK; removed after the build unless --keep
#   --keep         keep the clone
#   --commit SHA   the commit that the build reports (MDIR_GIT_COMMIT); by
#                  default the HEAD of the source if it is a git tree. Given
#                  with a git tree, its HEAD must be that commit.
#   --version V    the version of the tarball, the wheels, and the runtime
#                  image's tag (default: project(mdir VERSION ...) of the
#                  source's CMakeLists.txt)
#   --out DIR      where the tarball or the wheels go (default dist/TARGET
#                  of the source)
#   --tag NAME     the runtime image's tag (default mdir:VERSION); the tag of
#                  the CI image is computed and cannot be given
#   --push         ci and ci-prebuilt: push the image after the build. It
#                  publishes: only the maintainer runs it (docs/workflow.md)
#   --login        with --push: log in to GHCR with the token of `gh` (scope
#                  write:packages) first; without it the caller has logged in
#   --print-tag    ci and ci-prebuilt: print the tag and stop (ci.yml and
#                  ci-image.yml take it from here)
#   --dry-run      print the source, the commit, the result, and the command
#                  of the build, and stop: no Docker, no clone, no push
#
# Environment (all optional; docs/environment.md, Scripts):
#   DOCKER       the container program (default docker; one with the CLI of
#                docker build, and BuildKit's --output, such as podman)
#   JOBS         parallel compile jobs of LLVM and HDF5 (default 16)
#   LINK_JOBS    parallel link jobs of LLVM (default 4)
#   WORK         a scratch directory on a local disk, for --clone and the
#                default log (default /var/tmp/$USER)
#   LOG          the log of the build (default WORK/build-TARGET.log)
#   MIN_FREE_GB  refuse to start a build below this much free disk in WORK
#                (default 30)
set -euo pipefail

usage() {
  echo "usage: $0 runtime|tarball|wheels|ci|ci-prebuilt [--source DIR | --clone BRANCH [--keep]]" \
    "[--commit SHA] [--version V] [--out DIR] [--tag NAME] [--push [--login]] [--print-tag] [--dry-run]" >&2
  exit 2
}
(( $# )) || usage
target=$1; shift
case $target in runtime|tarball|wheels|ci|ci-prebuilt) ;; *) usage ;; esac
source=; clone=; keep=0; commit=; version=; out=; tag=; push=0; login=0; print_tag=0; dry=0
while (( $# )); do
  case $1 in
    --source) (( $# >= 2 )) || usage; source=$2; shift ;;
    --clone) (( $# >= 2 )) || usage; clone=$2; shift ;;
    --keep) keep=1 ;;
    --commit) (( $# >= 2 )) || usage; commit=$2; shift ;;
    --version) (( $# >= 2 )) || usage; version=$2; shift ;;
    --out) (( $# >= 2 )) || usage; out=$2; shift ;;
    --tag) (( $# >= 2 )) || usage; tag=$2; shift ;;
    --push) push=1 ;;
    --login) login=1 ;;
    --print-tag) print_tag=1 ;;
    --dry-run) dry=1 ;;
    *) usage ;;
  esac
  shift
done
refuse() { echo "$0: $*" >&2; exit 2; }
ci=0; [[ $target == ci || $target == ci-prebuilt ]] && ci=1
[[ -n $source && -n $clone ]] && refuse "--source and --clone exclude each other"
(( keep && ! ${#clone} )) && refuse "--keep is for --clone"
(( push && ! ci )) && refuse "--push is for the CI image; the runtime image, the tarball, and the wheels are published by the release"
(( login && ! push )) && refuse "--login is for --push"
(( print_tag && ! ci )) && refuse "--print-tag is for the CI image"
[[ -n $tag && $target != runtime ]] && refuse "--tag is for the runtime image"
[[ -n $out && $target != tarball && $target != wheels ]] && refuse "--out is for the tarball and the wheels"

DOCKER=${DOCKER:-docker}
JOBS=${JOBS:-16}
LINK_JOBS=${LINK_JOBS:-4}
WORK=${WORK:-/var/tmp/${USER:-$(id -un)}}
LOG=${LOG:-$WORK/build-$target.log}
MIN_FREE_GB=${MIN_FREE_GB:-30}

# The tag of the CI image: a hash of the Dockerfile and of the two scripts it
# builds HDF5 and LLVM with, so that a change of any of them makes a new one.
ci_tag() {
  local hash
  hash=$(cat "$1/packaging/Dockerfile" "$1/scripts/build-llvm.sh" "$1/scripts/build-hdf5.sh" |
    sha256sum | cut -c1-16)
  echo "ghcr.io/vv137/mdir-ci:llvm-23.1.2-$hash"
}
here=$(cd "$(dirname "$0")/../.." && pwd)

if (( print_tag )); then
  ci_tag "${source:-$here}"
  exit 0
fi

# The tree: as it is, or a fresh clone.
if [[ -n $clone ]]; then
  src=$WORK/mdir-build-$target
  if (( ! dry )); then
    mkdir -p "$WORK"
    rm -rf "$src"
    git clone -q --depth 1 --branch "$clone" https://github.com/vv137/mdir.git "$src"
    (( keep )) || trap 'rm -rf "$src"' EXIT
  fi
else
  src=$(cd "${source:-$here}" && pwd)
fi

# The commit the build reports, checked against a git tree.
if [[ -d $src/.git || -f $src/.git ]]; then
  head=$(git -C "$src" rev-parse HEAD)
  if [[ -n $commit && $head != "$commit"* ]]; then
    echo "$0: the source is at $head, not $commit" >&2; exit 1
  fi
  commit=${commit:-$head}
  if [[ -n $(git -C "$src" status --porcelain --untracked-files=no) ]]; then
    (( push )) && { echo "$0: the source has changes that are not committed: nothing is pushed" >&2; exit 1; }
    echo "$0: note: the source has changes that are not committed; the build reports $commit" >&2
  fi
elif [[ -z $commit && -z $clone ]]; then
  echo "$0: note: the source is not a git tree and no --commit is given; the build reports no commit" >&2
fi
if [[ -z $version && $target != ci && $target != ci-prebuilt && -f $src/CMakeLists.txt ]]; then
  version=$(sed -n 's/^project(mdir VERSION \([0-9.]*\) .*/\1/p' "$src/CMakeLists.txt")
fi

# The command of each target, from the root of the source.
args=()
case $target in
  runtime)
    result=${tag:-mdir:$version}
    args=(build -f packaging/Dockerfile -t "$result" --build-arg MDIR_GIT_COMMIT="$commit" .) ;;
  tarball|wheels)
    out=${out:-$src/dist/$target}
    result=$out
    args=(build -f packaging/Dockerfile.manylinux --build-arg MDIR_GIT_COMMIT="$commit"
          --build-arg VERSION="$version" --target "$target" --output "type=local,dest=$out" .) ;;
  ci|ci-prebuilt)
    result=$(ci_tag "$src")
    args=(build -f packaging/Dockerfile --target "$target"
          --build-arg JOBS="$JOBS" --build-arg LINK_JOBS="$LINK_JOBS" -t "$result" .) ;;
esac

if (( dry )); then
  echo "source: ${clone:+the clone of $clone, }$src"
  echo "commit: ${commit:-none}"
  [[ $ci == 1 ]] || echo "version: $version"
  echo "result: $result"
  echo "build: DOCKER_BUILDKIT=1 $DOCKER ${args[*]}"
  (( push )) && echo "push: $DOCKER push $result"
  echo "log: $LOG"
  exit 0
fi

command -v "$DOCKER" >/dev/null || { echo "$0: no container program: $DOCKER (set DOCKER)" >&2; exit 1; }
mkdir -p "$WORK" "$(dirname "$LOG")"
free_gb=$(df -Pk "$WORK" | awk 'NR==2 {print int($4 / 1048576)}')
(( free_gb >= MIN_FREE_GB )) ||
  { echo "$0: only $free_gb GB free in $WORK (need $MIN_FREE_GB)" >&2; exit 1; }
echo "== $target from $src at ${commit:-no commit}; $(hostname): load $(cut -d' ' -f1-3 /proc/loadavg)," \
  "$(nproc) cores, $free_gb GB free; JOBS=$JOBS LINK_JOBS=$LINK_JOBS"
echo "== $result (log $LOG)"
start=$(date +%s)
(cd "$src" && DOCKER_BUILDKIT=1 "$DOCKER" "${args[@]}") > "$LOG" 2>&1 ||
  { tail -n 20 "$LOG" >&2; echo "$0: the build of $target failed: $LOG" >&2; exit 1; }
echo "== built in $(( ($(date +%s) - start) / 60 )) min"
if (( ci )); then
  "$DOCKER" image inspect "$result" --format '{{.Size}}' | awk '{printf "== %.2f GB\n", $1 / 1e9}'
fi

if (( push )); then
  if (( login )); then
    gh auth status 2>&1 | grep -q write:packages ||
      { echo "$0: gh has no write:packages: gh auth refresh -h github.com -s write:packages" >&2; exit 1; }
    gh auth token | "$DOCKER" login ghcr.io -u "$(gh api user --jq .login)" --password-stdin > /dev/null
  fi
  "$DOCKER" push -q "$result"
  echo "== pushed $result"
elif (( ci )); then
  echo "== not pushed; the maintainer pushes with --push, or: $DOCKER push $result"
fi
