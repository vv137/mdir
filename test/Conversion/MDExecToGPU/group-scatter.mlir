// The table of a block in the kernels of the loops over groups
// (`scatter(hash(S, P, L))` on `md_exec.pair_for`, D291): the
// program of Integration/eam.mlir, whose density and forces are loops over
// groups, in mixed precision, lowered for a device without one.
//
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-choose-scatter="pair-hash=on" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | FileCheck %s
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-choose-scatter="pair-hash=on pair-hash-least=1000" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --check-prefix=LEAST
// RUN: mdir-opt %S/../../Integration/eam.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-choose-neighbors="kind=groups" \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-choose-scatter \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --check-prefix=OFF

// The loop of the density and the forces of its pairs: a table of 512
// slots (128 threads, 3 particles a thread), the keys, the density, and the
// forces, cleared by the threads of the block before a barrier; a slot is
// found by a compare-and-swap of block memory, the values added there, and
// a value whose search fails goes to the device's memory; after a barrier,
// each slot that holds a particle goes to the destinations once.
// CHECK:       gpu.launch {{.*}} workgroup(%[[KEYS:[a-z0-9]+]] : memref<512xi32, #gpu.address_space<workgroup>>, %[[RHO:[a-z0-9]+]] : memref<512xf32, #gpu.address_space<workgroup>>, %[[F:[a-z0-9]+]] : memref<1536xf32, #gpu.address_space<workgroup>>)
// CHECK:         scf.for
// CHECK:           memref.store %c-1_i32, %[[KEYS]][
// CHECK:           memref.store %{{.*}}, %[[RHO]][
// CHECK:           memref.store %{{.*}}, %[[F]][
// CHECK:         gpu.barrier
// CHECK:         scf.for
// CHECK:           scf.while
// CHECK:             "atom.shared.cas.b32 $0, [$1], $2, $3;"
// CHECK:           scf.if
// CHECK:             "red.shared.add.f32 [$0], $1;"
// CHECK:           } else {
// CHECK:             "red.relaxed.gpu.global.add.f32 [$0], $1;"
// CHECK:         gpu.barrier
// CHECK:         scf.for
// CHECK:           memref.load %[[KEYS]][
// CHECK:           arith.cmpi ne
// CHECK:           scf.if
// CHECK:             memref.load %[[RHO]][
// CHECK:             "red.relaxed.gpu.global.add.f32 [$0], $1;"
// CHECK:             memref.load %[[F]][
// CHECK-COUNT-3:     "red.relaxed.gpu.global.add.f32 [$0], $1;"
// CHECK:         gpu.terminator

// With L = 1000 units of work, the host compares the count of the launch
// and runs the kernel with the table or the one without it.
// LEAST:       arith.cmpi uge, %{{.*}}, %c1000
// LEAST:       scf.if %{{.*}} -> (index) {
// LEAST:         gpu.launch {{.*}} workgroup(
// LEAST:       } else {
// LEAST-NOT:     workgroup(
// LEAST:         gpu.launch
// LEAST:       }

// OFF-NOT: atom.shared.cas
// OFF-NOT: red.shared.add
