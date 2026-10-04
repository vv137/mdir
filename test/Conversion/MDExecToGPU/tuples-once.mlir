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

// With global sums, a thread for each of the tuples or of the particles,
// whichever are fewer, takes every so manyth tuple and writes the sums of
// its tuples to its row, and the reduction is over those rows: a set of few
// tuples, as the pairs of the centers of groups, launches and reduces no
// more rows than it has tuples. The rows are reduced in one part when there
// are few; the block of that part then evaluates the tuples of each row
// itself, and the kernel that writes the rows to the buffer of
// contributions is launched only for several parts (#20).
//
// CHECK-LABEL: func.func @energy(
// CHECK:         %[[TUPLES:[a-z0-9_]+]] = memref.dim %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?x2xi32, 1>
// CHECK:         %[[COUNT:[0-9]+]] = arith.minui %{{[a-z0-9_]+}}, %[[TUPLES]] : index
// CHECK:         %[[PARTS:[0-9]+]] = arith.minsi
// CHECK:         %[[ONE:[0-9]+]] = arith.cmpi eq, %[[PARTS]]
// CHECK:         %[[SEVERAL:[0-9]+]] = arith.cmpi ne, %[[PARTS]]
// CHECK:         scf.if %[[SEVERAL]] {
// CHECK:           gpu.launch
// CHECK:             scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[TUPLES]] step %[[COUNT]]
// CHECK:             memref.store %{{[0-9]+}}, %[[A:[a-z0-9]+]][
// CHECK:         gpu.launch
// CHECK:           scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[COUNT]] step
// CHECK:             scf.if %[[ONE]] -> (f64) {
// CHECK:               scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[TUPLES]] step %[[COUNT]]
// CHECK:             } else {
// CHECK:               memref.load %[[A]][
func.func @energy(%members: memref<?x2xi32>, %n: index,
                  %x: memref<?x3xf32, 1>, %cell: !md.cell,
                  %f: memref<?x3xf32, 1>, %a: memref<?xf64, 1>,
                  %b: memref<?xf64, 1>) -> f64 {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    %s = vector.reduction <add>, %d : vector<3xf32> into f32
    %e = arith.extf %s : f32 to f64
    md_exec.yield %d, %n0, %e : vector<3xf32>, vector<3xf32>, f64
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1> -> f64
  return %u : f64
}
