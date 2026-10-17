// A loop over tuples that tells whether its kernel yields true for some
// tuple (D278): on the host the value is reduced as an
// integer of 32 bits, as that of a loop over particles, and compared with 0.
//
// RUN: mdir-opt %s --convert-md-exec-to-loops | FileCheck %s

md.particle_set @atoms

// CHECK-LABEL: func.func @stretched(
// CHECK:         %[[NO:[a-z0-9_]+]] = arith.constant false
// CHECK:         %[[WIDE:[0-9]+]] = arith.extui %[[NO]] : i1 to i32
// CHECK:         %[[ANY:[0-9]+]] = scf.parallel ({{.*}}) init (%[[WIDE]]) -> i32 {
// CHECK:           arith.extui %{{[0-9]+}} : i1 to i32
// CHECK:           scf.reduce(%{{[0-9]+}} : i32) {
// CHECK:             arith.ori %{{[a-z0-9]+}}, %{{[a-z0-9]+}} : i32
// CHECK:         %[[SET:[0-9]+]] = arith.cmpi ne, %[[ANY]], %{{[a-z0-9_]+}} : i32
// CHECK:         return %[[SET]] : i1
func.func @stretched(%pairs: memref<?x?xi32>, %x: memref<?x3xf64>,
                     %cell: !md.cell) -> i1 {
  %no = arith.constant false
  %any = md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      reduce(%no : i1) arity(2) disjoint {
  ^bb0(%d: vector<3xf64>):
    %sq = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 1.0 : f64
    %far = arith.cmpf ugt, %r2, %limit : f64
    md_exec.yield %far : i1
  } : memref<?x?xi32>, memref<?x3xf64> -> i1
  return %any : i1
}

// Over tuples that may share particles, each member takes the tuple, and
// the tuples of a particle are or-ed.
//
// CHECK-LABEL: func.func @shared(
// CHECK:         scf.parallel
// CHECK:           scf.for {{.*}} -> (i1) {
// CHECK:             arith.select %{{[0-9]+}}, %{{[0-9]+}}, %{{[a-z0-9_]+}} : i1
// CHECK:             arith.ori
// CHECK:           arith.extui %{{[0-9]+}} : i1 to i32
// CHECK:         arith.cmpi ne
func.func @shared(%pairs: memref<?x?xi32>, %x: memref<?x3xf64>,
                  %cell: !md.cell) -> i1 {
  %no = arith.constant false
  %any = md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      reduce(%no : i1) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %sq = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 1.0 : f64
    %far = arith.cmpf ugt, %r2, %limit : f64
    md_exec.yield %far : i1
  } : memref<?x?xi32>, memref<?x3xf64> -> i1
  return %any : i1
}
