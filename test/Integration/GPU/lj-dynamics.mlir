// The test of ../lj-dynamics.mlir on a GPU, in double and in mixed
// precision.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../lj-dynamics.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../lj-dynamics.mlir

// RUN: sed 's/%%tolerance = arith.constant 1.0e-9/%%tolerance = arith.constant 1.0e-6/' \
// RUN:     %S/../lj-dynamics.mlir \
// RUN: | mdir-opt %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | grep -c '^1$' | FileCheck %s --check-prefix=MIXED
// MIXED: 7
