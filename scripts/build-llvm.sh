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
#   OPENMP_BUILD  Build directory of the OpenMP runtime.
#   LLVM_PREFIX   Install prefix.
#   LLVM_TARGETS  Semicolon-separated list of LLVM targets.
#   JOBS          Parallel compile jobs.
#   LINK_JOBS     Parallel link jobs. Linking is memory-hungry; keep this low.

set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-23.1.2}"
LLVM_ROOT="${LLVM_ROOT:-$HOME/opt/llvm}"
LLVM_SRC="${LLVM_SRC:-$LLVM_ROOT/src/llvm-project-$LLVM_VERSION}"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_ROOT/build/$LLVM_VERSION}"
OPENMP_BUILD="${OPENMP_BUILD:-$LLVM_ROOT/build/openmp-$LLVM_VERSION}"
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

if [ -x "$LLVM_PREFIX/bin/mlir-opt" ]; then
    log "LLVM is already installed; skipping its build"
    SKIP_LLVM=1
fi

if [ ! -d "$LLVM_SRC/llvm" ]; then
    log "cloning llvmorg-$LLVM_VERSION"
    mkdir -p "$(dirname "$LLVM_SRC")"
    git clone --depth 1 --branch "llvmorg-$LLVM_VERSION" \
        https://github.com/llvm/llvm-project.git "$LLVM_SRC"
fi

if [ -z "${SKIP_LLVM:-}" ]; then

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

fi

# The OpenMP runtime is a separate project. Code that is lowered through the
# omp dialect needs it at run time.
if [ ! -f "$LLVM_PREFIX/lib/libomp.so" ]; then
    log "configuring the OpenMP runtime"
    cmake -G "$GENERATOR" -S "$LLVM_SRC/runtimes" -B "$OPENMP_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
        -DLLVM_ENABLE_RUNTIMES=openmp \
        -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
        -DLIBOMP_OMPD_SUPPORT=OFF \
        -DLIBOMP_FORTRAN_MODULES=OFF

    log "building the OpenMP runtime"
    cmake --build "$OPENMP_BUILD" --parallel "$JOBS"

    log "installing the OpenMP runtime"
    cmake --install "$OPENMP_BUILD"
fi

log "done: $("$LLVM_PREFIX/bin/mlir-opt" --version | sed -n '2p' | xargs)"
