// The example of liquid argon conserves the total energy: over 2000 steps
// it changes by less than 1e-4 of its value.
//
// RUN: mdir-opt %S/../../examples/argon.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.1 width=160" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | tail -n 1 \
// RUN: | awk '{ print ($1 < 1e-4) ? "conserved" : "not conserved" }' \
// RUN: | FileCheck %s

// In mixed precision, by less than 1e-3.
//
// RUN: mdir-opt %S/../../examples/argon.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.1 width=160" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | tail -n 1 \
// RUN: | awk '{ print ($1 < 1e-3) ? "conserved" : "not conserved" }' \
// RUN: | FileCheck %s

// CHECK: {{^}}conserved
