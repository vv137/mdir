// The test of ../dist-eam-exec.mlir on a GPU: the loops of each domain through the
// device lowering of md_exec, the domains one after another on one
// device, and the transfers on the host (D[ml2-exec]).
//
// REQUIRES: cuda
//
// RUN: sed 's/GRID/1, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=1,1,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-5/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=unique potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-5/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-5/' %S/../dist-eam-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=eam,linear" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --convert-md-dist-to-md-exec="functions=derived,energy,vjp" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-eam-exec.mlir --check-prefixes=CHECK
//
