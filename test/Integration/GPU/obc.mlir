// The test of ../obc.mlir on a GPU, with both neighbor structures, in
// double precision and, within 1e-5, in mixed precision (D143).
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../obc.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../obc.mlir

// RUN: mdir-opt %S/../obc.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../obc.mlir

// RUN: sed -e 's/arith.constant 1.0e-12 : f64/arith.constant 1.0e-5 : f64/' \
// RUN:     -e 's/arith.constant 1.0e-9 : f64/arith.constant 1.0e-5 : f64/g' \
// RUN:     %S/../obc.mlir \
// RUN: | mdir-opt %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../obc.mlir
