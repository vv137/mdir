// md_alg.traverse_pairs (D415): the structure form, which
// the pipeline emits, with the inner list of a dual list and the execution
// group; and the buffer form, for the serial oracle and the tests, with
// its stated contract and reach. Both round-trip, in the custom and the
// generic form.
//
// RUN: mdir-opt %s | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --mlir-print-op-generic | mdir-opt | FileCheck %s

!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!i = memref<?xi32, 1>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
!s = memref<?xf64, 1>

// CHECK-LABEL: func.func @structure
// CHECK: md_alg.traverse_pairs structure %{{.*}} : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)> inner
// CHECK-SAME: positions(%{{.*}} : memref<?x3xf64, 1>) cell(%{{.*}} : !md.cell)
// CHECK-SAME: ins(%{{.*}} : memref<?xf32, 1>) outs(%{{.*}} : memref<?x3xf32, 1>) reduce(%{{.*}} : f64)
// CHECK-SAME: scratch(%{{.*}}, %{{.*}} : memref<?xf64, 1>, memref<?xf64, 1>)
// CHECK-SAME: unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
// CHECK-SAME: cutoff(9.000000e-01) weights [5.000000e-01] overwrite [true]
// CHECK-SAME: scatter(#md_exec.scatter<hash(512, 4, 2000)>) frames prune(4096, 1.000000e+00)
// CHECK: md_alg.yield %{{.*}}, %{{.*}} : vector<3xf32>, f64
func.func @structure(%nl: !groups, %x: !x, %cell: !md.cell, %q: !q, %f: !f,
                     %s0: !s, %s1: !s, %u0: f64) -> f64 {
  %u = md_alg.traverse_pairs structure %nl : !groups inner
      positions(%x : !x) cell(%cell : !md.cell)
      ins(%q : !q) outs(%f : !f) reduce(%u0 : f64) scratch(%s0, %s1 : !s, !s)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(0.9) weights [0.5] overwrite [true]
      scatter(#md_exec.scatter<hash(512, 4, 2000)>) frames prune(4096, 1.0) {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
    %qq = arith.mulf %qi, %qj : f32
    %k = vector.broadcast %qq : f32 to vector<3xf32>
    %kf = arith.mulf %k, %d : vector<3xf32>
    %e = arith.extf %qq : f32 to f64
    md_alg.yield %kf, %e : vector<3xf32>, f64
  } -> f64
  return %u : f64
}

// CHECK-LABEL: func.func @buffers
// CHECK: md_alg.traverse_pairs lists(%{{.*}} : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%{{[a-z0-9]+}}, %{{[a-z0-9]+}})
// CHECK-SAME: contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>) reach(1.200000e+00)
// CHECK-SAME: cell(%{{.*}} : vector<3xf64>)
func.func @buffers(%e: memref<?xi32>, %m: memref<?xi32>, %c: memref<?xi32>,
                   %un: memref<?xi32>, %o: memref<?xi32>, %p: memref<?xi32>,
                   %sh: memref<?xi32>, %places: index, %units: index,
                   %x: memref<?x3xf64>, %cell: vector<3xf64>,
                   %f: memref<?x3xf64>, %u0: f64) -> f64 {
  %u = md_alg.traverse_pairs
      lists(%e, %m, %c, %un, %o, %p, %sh : memref<?xi32>, memref<?xi32>,
            memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>,
            memref<?xi32>) sizes(%places, %units)
      contract(!groups) reach(1.2)
      positions(%x : memref<?x3xf64>) cell(%cell : vector<3xf64>)
      outs(%f : memref<?x3xf64>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(1.0) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_alg.yield %d, %r2 : vector<3xf64>, f64
  } -> f64
  return %u : f64
}
