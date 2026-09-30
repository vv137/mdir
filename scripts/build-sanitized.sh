#!/bin/sh
# Builds the driver and the runtimes under AddressSanitizer and
# UndefinedBehaviorSanitizer, and runs the tests with them.
#
#   scripts/build-sanitized.sh <build directory> [<arguments of lit>]
#
# LLVM, MLIR, and the libraries that MDIR links are not instrumented; the
# code of MDIR is. Compiled programs run in the same process, so their
# calls into the driver are checked too. The variables LLVM_DIR, MLIR_DIR,
# HDF5_ROOT, and CUDAToolkit_ROOT are passed to CMake if they are set.
# Leaks are not reported: the driver leaves the memory of a run to the end
# of the process.
set -eu

if [ $# -lt 1 ]; then
  echo "usage: $0 <build directory> [<arguments of lit>]" >&2
  exit 2
fi
build=$1
shift
source=$(cd "$(dirname "$0")/.." && pwd)
flags="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"

cmake -G Ninja -S "$source" -B "$build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  ${LLVM_DIR:+-DLLVM_DIR="$LLVM_DIR"} \
  ${MLIR_DIR:+-DMLIR_DIR="$MLIR_DIR"} \
  ${HDF5_ROOT:+-DHDF5_ROOT="$HDF5_ROOT"} \
  ${CUDAToolkit_ROOT:+-DCUDAToolkit_ROOT="$CUDAToolkit_ROOT"}
cmake --build "$build"

# - The CUDA driver maps memory that AddressSanitizer would take for its
#   own; protect_shadow_gap=0 lets it.
# - The headers of LLVM poison the memory of their allocators when they are
#   compiled with AddressSanitizer, and the uninstrumented libraries do not
#   unpoison it; allow_user_poisoning=0 ignores both.
# - The tests of the runtime load the instrumented runtime into mlir-runner,
#   which is not instrumented; verify_asan_link_order=0 lets them.
export ASAN_OPTIONS="detect_leaks=0:protect_shadow_gap=0:allow_user_poisoning=0:verify_asan_link_order=0${ASAN_OPTIONS:+:$ASAN_OPTIONS}"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1${UBSAN_OPTIONS:+:$UBSAN_OPTIONS}"
# The AddressSanitizer of older compilers (GCC 9) loops on the random
# placement of mappings of recent kernels; the tests run without it.
setarch "$(uname -m)" -R lit -j 16 "$build/test" "$@"
