// By default (tuples-once), a loop over tuples that may share particles, has no
// global sums, and adds to its destination evaluates each tuple once, a
// thread for each tuple, and adds to the members with atomics: in f32 by
// PTX's own reduction.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu="tuples-once=true" | FileCheck %s
// RUN: mdir-opt %s --convert-md-exec-to-gpu="tuples-once=false" | FileCheck %s --check-prefix=ROWS

md.particle_set @atoms

// CHECK-LABEL: func.func @bonds(
// CHECK:         gpu.memcpy
// CHECK:         gpu.launch
// CHECK-COUNT-6:   llvm.inline_asm has_side_effects "red.relaxed.gpu.global.add.f32 [$0], $1;"
// ROWS-LABEL:  func.func @bonds(
// ROWS-NOT:      llvm.inline_asm
func.func @bonds(%members: memref<?x2xi32>, %n: index,
                 %x: memref<?x3xf32, 1>, %cell: !md.cell,
                 %f: memref<?x3xf32, 1>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  return
}
