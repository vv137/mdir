#!/usr/bin/env bash
# Compiles an example and runs it.
#
# Usage:
#   examples/argon/run.sh <example.mlir> [mode] [target]
#
#   mode    The precision mode: double (the default), mixed, or single.
#           The mode single needs an example whose buffers hold f32.
#   target  A number of OpenMP threads, or gpu. Without it, the loops run
#           sequentially on the host and the OpenMP runtime is not loaded.
#
# Environment variables:
#   MDIR_BUILD   The build directory of MDIR. The default is build.
#   LLVM_PREFIX  The install prefix of LLVM.
#   CUDA_ROOT    The CUDA toolkit, for the target gpu. The kernels take
#                their math functions from it.
#   MDRT_DEVICE  The GPU to run on, among those that CUDA_VISIBLE_DEVICES
#                leaves visible. The default is 0.
#   SKIN, WIDTH  The skin of the neighbor structures, and the number of
#                neighbors that they hold per particle.

set -euo pipefail

INPUT="${1:?usage: examples/argon/run.sh <example.mlir> [mode] [target]}"
MODE="${2:-double}"
TARGET="${3:-}"

MDIR_BUILD="${MDIR_BUILD:-build}"
LLVM_PREFIX="${LLVM_PREFIX:-$HOME/opt/llvm/23.1.2}"
SKIN="${SKIN:-0.1}"
WIDTH="${WIDTH:-160}"

LOOPS="--convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
--convert-vector-to-llvm --expand-strided-metadata --finalize-memref-to-llvm \
--convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm"
LIBS="$LLVM_PREFIX/lib/libmlir_c_runner_utils.so,$MDIR_BUILD/lib/libmdrt.so"

if [ "$TARGET" = gpu ]; then
    export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
    STORAGE="--md-exec-assign-storage=memory=device --convert-md-exec-to-gpu"
    LOWER="--gpu-lower-to-nvvm-pipeline=cubin-format=isa \
--reconcile-unrealized-casts"
    LIBS="$LIBS,$MDIR_BUILD/lib/libmdrt_cuda.so"
elif [ -n "$TARGET" ]; then
    STORAGE="--md-exec-assign-storage --convert-md-exec-to-loops"
    LOWER="--convert-scf-to-openmp --canonicalize $LOOPS \
--convert-openmp-to-llvm --reconcile-unrealized-casts"
    LIBS="$LIBS,$LLVM_PREFIX/lib/libomp.so"
    export OMP_NUM_THREADS="$TARGET"
else
    STORAGE="--md-exec-assign-storage --convert-md-exec-to-loops"
    LOWER="$LOOPS --reconcile-unrealized-casts"
fi

"$MDIR_BUILD/bin/mdir-opt" "$INPUT" \
    --md-check-exchange --md-differentiate --md-expand-truncation \
    --md-inline --convert-md-to-md-exec="skin=$SKIN width=$WIDTH" \
    --md-exec-reuse-neighbors --md-exec-expose-validity \
    --md-exec-fuse-loops --canonicalize --cse \
    --md-exec-assign-precision="mode=$MODE" $STORAGE \
  | "$LLVM_PREFIX/bin/mlir-opt" $LOWER \
  | "$LLVM_PREFIX/bin/mlir-runner" -O3 -e main --entry-point-result=void \
      --shared-libs="$LIBS"
