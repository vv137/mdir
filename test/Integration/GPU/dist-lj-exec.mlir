// The test of ../dist-lj-exec.mlir on a GPU: the loops of each domain through the
// device lowering of md_exec, the domains one after another on one
// device, and the transfers on the host (D290).
//
// REQUIRES: cuda
//
// RUN: sed 's/GRID/1, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=1,1,1 traversal=unique potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/1, 2, 3/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="domains=6 box=4.8,4.8,4.8 traversal=unique potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=unique potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/16, 1, 1/; s/TOLERANCE/1.0e-10/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=16,1,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK,DOUBLE
//
// RUN: sed 's/GRID/1, 2, 3/; s/TOLERANCE/1.0e-5/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="domains=6 box=4.8,4.8,4.8 traversal=unique potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-5/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK
//
// RUN: sed 's/GRID/3, 2, 1/; s/TOLERANCE/1.0e-5/' %S/../dist-lj-exec.mlir \
// RUN: | mdir-opt --md-distribute="grid=3,2,1 traversal=directed potentials=lj" \
// RUN:     %md_passes --md-dist-owner-computes \
// RUN:     --md-exec-assign-algebra --convert-md-dist-to-md-exec="functions=derived,energy,handwritten" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %S/../dist-lj-exec.mlir --check-prefixes=CHECK
//
