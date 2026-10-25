#!/usr/bin/env bash
# Builds the image of the CPU tier of CI (#611) with Docker on a machine of
# one's own, and pushes it to GHCR under the tag that
# .github/workflows/ci-image.yml and ci.yml compute, so that CI pulls it
# instead of building LLVM on GitHub (docs/workflow.md, Continuous
# integration):
#
#   scripts/ci/build-image.sh [--push] [--keep] [--dry-run] [--source DIR]
#
#   --push        log in to GHCR with the token of `gh` (it needs the scope
#                 write:packages) and push the image after the build
#   --keep        keep the clone of the branch
#   --dry-run     print the source, the tag, and the command of the build,
#                 and stop: no Docker, no clone of a --source tree
#   --source DIR  build from the tree DIR as it is, instead of a clone of
#                 BRANCH
#
# Environment (all optional; the defaults suit any machine):
#   WORK         scratch directory on a local disk (default /var/tmp/$USER)
#   BRANCH       branch of vv137/mdir to clone (default main)
#   COMMIT       expected short commit of the branch (default: not checked)
#   JOBS         parallel compile jobs of LLVM and HDF5 (default 16)
#   LINK_JOBS    parallel link jobs of LLVM (default 4)
#   MIN_FREE_GB  refuse to start below this much free disk (default 30)
#
# The image is the stage `ci` of packaging/Dockerfile, which builds LLVM in
# one go: the same image that ci-image.yml makes in parts on hosted runners.
# The log of the build is $WORK/ci-image.log.
set -euo pipefail

usage() { echo "usage: $0 [--push] [--keep] [--dry-run] [--source DIR]" >&2; exit 2; }
push=0; keep=0; dry=0; source=
while (( $# )); do
  case $1 in
    --push) push=1 ;;
    --keep) keep=1 ;;
    --dry-run) dry=1 ;;
    --source) (( $# >= 2 )) || usage; source=$2; shift ;;
    *) usage ;;
  esac
  shift
done

WORK=${WORK:-/var/tmp/$USER}
BRANCH=${BRANCH:-main}
JOBS=${JOBS:-16}
LINK_JOBS=${LINK_JOBS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-30}
log=$WORK/ci-image.log

# The tag of ci-image.yml and ci.yml: a hash of the Dockerfile and of its
# two build scripts.
tag_of() {
  local hash
  hash=$(cat "$1/packaging/Dockerfile" "$1/scripts/build-llvm.sh" "$1/scripts/build-hdf5.sh" |
    sha256sum | cut -c1-16)
  echo "ghcr.io/vv137/mdir-ci:llvm-23.1.2-$hash"
}
build_command() {
  echo "docker build -f packaging/Dockerfile --target ci" \
    "--build-arg JOBS=$JOBS --build-arg LINK_JOBS=$LINK_JOBS -t $1 ."
}

if (( dry )); then
  [[ -n $source ]] || { echo "--dry-run takes --source DIR" >&2; exit 2; }
  image=$(tag_of "$source")
  echo "source: $source"
  echo "image: $image"
  echo "build: $(build_command "$image")"
  (( push )) && echo "push: docker push $image"
  exit 0
fi

need() { command -v "$1" >/dev/null || { echo "missing: $1" >&2; exit 1; }; }
need docker; need git; need sha256sum
(( push )) && need gh

mkdir -p "$WORK"
free_gb=$(df -Pk "$WORK" | awk 'NR==2 {print int($4 / 1048576)}')
(( free_gb >= MIN_FREE_GB )) ||
  { echo "only $free_gb GB free in $WORK (need $MIN_FREE_GB)" >&2; exit 1; }
echo "== $(hostname): load $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) cores," \
  "$free_gb GB free; JOBS=$JOBS LINK_JOBS=$LINK_JOBS"

if [[ -n $source ]]; then
  src=$(cd "$source" && pwd)
  echo "== the tree $src"
else
  src=$WORK/mdir-ci-image
  rm -rf "$src"
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/vv137/mdir.git "$src"
  cleanup() { (( keep )) || rm -rf "$src"; }
  trap cleanup EXIT
  head=$(git -C "$src" log -1 --format=%h)
  echo "== $BRANCH at $head"
  if [[ -n ${COMMIT:-} && $head != "${COMMIT:0:${#head}}" ]]; then
    echo "expected $COMMIT, got $head" >&2; exit 1
  fi
fi
cd "$src"
image=$(tag_of .)
echo "== image $image (log $log)"

start=$(date +%s)
# shellcheck disable=SC2046
DOCKER_BUILDKIT=1 $(build_command "$image") > "$log" 2>&1 ||
  { tail -20 "$log" >&2; echo "the build failed: $log" >&2; exit 1; }
size=$(docker image inspect "$image" --format '{{.Size}}' | awk '{printf "%.2f GB", $1 / 1e9}')
echo "== built in $(( ($(date +%s) - start) / 60 )) min, $size"

if (( push )); then
  gh auth status 2>&1 | grep -q write:packages ||
    { echo "gh has no write:packages: gh auth refresh -h github.com -s write:packages" >&2; exit 1; }
  gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin > /dev/null
  docker push -q "$image"
  echo "== pushed $image"
else
  echo "== not pushed; run again with --push, or: docker push $image"
fi
