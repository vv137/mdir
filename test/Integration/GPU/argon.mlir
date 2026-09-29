// The example of liquid argon on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../../../examples/argon.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.1 width=160" \
// RUN:     %md_exec_gpu_passes \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | tail -n 1 \
// RUN: | awk '{ print ($1 < 1e-4) ? "conserved" : "not conserved" }' \
// RUN: | FileCheck %s

// CHECK: {{^}}conserved
