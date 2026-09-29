// The test of ../tuples-mixed.mlir on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../tuples-mixed.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../tuples-mixed.mlir
