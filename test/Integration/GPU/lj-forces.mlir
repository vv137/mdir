// The test of ../lj-forces.mlir on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../lj-forces.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../lj-forces.mlir
