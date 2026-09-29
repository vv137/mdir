#!/usr/bin/env bash
# Compiles an example and runs it.
#
# Usage:
#   examples/run.sh <example.mlir> [mode] [threads]
#
#   mode     The precision mode: double (the default), mixed, or single.
#            The mode single needs an example whose buffers hold f32.
#   threads  The number of OpenMP threads. Without it, the loops run
#            sequentially and the OpenMP runtime is not loaded.
#
# Environment variables:
#   MDIR_BUILD   The build directory of MDIR. The default is build.
#   LLVM_PREFIX  The install prefix of LLVM.
#   SKIN, WIDTH  The skin of the neighbor structures, and the number of
#                neighbors that they hold per particle.

set -euo pipefail

INPUT="${1:?usage: examples/run.sh <example.mlir> [mode] [threads]}"
MODE="${2:-double}"
THREADS="${3:-}"

MDIR_BUILD="${MDIR_BUILD:-build}"
LLVM_PREFIX="${LLVM_PREFIX:-$HOME/opt/llvm/23.1.2}"
SKIN="${SKIN:-0.1}"
WIDTH="${WIDTH:-160}"

LOOPS="--convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
--convert-vector-to-llvm --expand-strided-metadata --finalize-memref-to-llvm \
--convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm"
LIBS="$LLVM_PREFIX/lib/libmlir_c_runner_utils.so,$MDIR_BUILD/lib/libmdrt.so"

if [ -n "$THREADS" ]; then
    LOWER="--convert-scf-to-openmp --canonicalize $LOOPS \
--convert-openmp-to-llvm --reconcile-unrealized-casts"
    LIBS="$LIBS,$LLVM_PREFIX/lib/libomp.so"
    export OMP_NUM_THREADS="$THREADS"
else
    LOWER="$LOOPS --reconcile-unrealized-casts"
fi

"$MDIR_BUILD/bin/mdir-opt" "$INPUT" \
    --md-check-exchange --md-differentiate --md-expand-truncation \
    --md-inline --convert-md-to-md-exec="skin=$SKIN width=$WIDTH" \
    --md-exec-reuse-neighbors --md-exec-fuse-loops --canonicalize --cse \
    --md-exec-assign-precision="mode=$MODE" \
    --md-exec-assign-storage --convert-md-exec-to-loops \
  | "$LLVM_PREFIX/bin/mlir-opt" $LOWER \
  | "$LLVM_PREFIX/bin/mlir-runner" -O3 -e main --entry-point-result=void \
      --shared-libs="$LIBS"
