#!/usr/bin/env bash
# Build and install the LLVM/MLIR release that MDIR is developed against.
#
# Usage:
#   scripts/build-llvm.sh
#
# Environment variables (all optional):
#   LLVM_VERSION  Release to build, without the "llvmorg-" prefix.
#   LLVM_ROOT     Directory that holds the source tree and the install tree.
#   LLVM_SRC      Source directory. Cloned if it does not exist.
#   LLVM_BUILD    Build directory. Prefer a local disk over a network mount.
#   LLVM_PREFIX   Install prefix.
#   LLVM_TARGETS  Semicolon-separated list of LLVM targets.
#   JOBS          Parallel compile jobs.
#   LINK_JOBS     Parallel link jobs. Linking is memory-hungry; keep this low.

set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-23.1.2}"
LLVM_ROOT="${LLVM_ROOT:-$HOME/opt/llvm}"
LLVM_SRC="${LLVM_SRC:-$LLVM_ROOT/src/llvm-project-$LLVM_VERSION}"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_ROOT/build/$LLVM_VERSION}"
LLVM_PREFIX="${LLVM_PREFIX:-$LLVM_ROOT/$LLVM_VERSION}"
LLVM_TARGETS="${LLVM_TARGETS:-X86;NVPTX;AMDGPU}"
JOBS="${JOBS:-$(nproc)}"
LINK_JOBS="${LINK_JOBS:-8}"

log() { printf '[build-llvm %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

log "version:   $LLVM_VERSION"
log "source:    $LLVM_SRC"
log "build:     $LLVM_BUILD"
log "prefix:    $LLVM_PREFIX"
log "targets:   $LLVM_TARGETS"
log "generator: $GENERATOR"
log "jobs:      $JOBS compile, $LINK_JOBS link"

if [ ! -d "$LLVM_SRC/llvm" ]; then
    log "cloning llvmorg-$LLVM_VERSION"
    mkdir -p "$(dirname "$LLVM_SRC")"
    git clone --depth 1 --branch "llvmorg-$LLVM_VERSION" \
        https://github.com/llvm/llvm-project.git "$LLVM_SRC"
fi

log "configuring"
cmake -G "$GENERATOR" -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_PARALLEL_LINK_JOBS="$LINK_JOBS"

log "building"
cmake --build "$LLVM_BUILD" --parallel "$JOBS"

log "installing"
cmake --install "$LLVM_BUILD"

log "done: $("$LLVM_PREFIX/bin/mlir-opt" --version | sed -n '2p' | xargs)"
