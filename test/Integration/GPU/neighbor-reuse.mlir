// The test of ../neighbor-reuse.mlir on a GPU. The pairs and the number of
// builds are those of the host.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../neighbor-reuse.mlir \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../neighbor-reuse.mlir
