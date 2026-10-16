// A loop over tuples that tells whether its kernel yields true for some
// tuple (D[constraint-failure-stop]) takes a flag of its own in memory of
// the host mapped for the device (D118): a thread whose tuple yields true
// stores 1, and the host reads the flag before the first op that uses the
// value, not where the loop is.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s

md.particle_set @atoms

// Two loops over the disjoint tuples of two sets, the test of SETTLE and
// that of a set of SHAKE: a kernel and a flag for each, read after the work
// between them is issued.
//
// CHECK-LABEL: func.func @two_sets(
// The flags are taken where the function begins, each before the ones
// taken earlier.
//
// CHECK:         %[[FLAG1:[0-9]+]] = call @mdrtFlagMemory(
// CHECK:         %[[FLAG0:[0-9]+]] = call @mdrtFlagMemory(
// CHECK:         gpu.launch
// CHECK:           scf.if
// CHECK:             memref.store %{{[a-z0-9_]+}}, %[[FLAG0]][
// CHECK:         memref.extract_aligned_pointer_as_index %[[FLAG0]]
// CHECK:         %[[SLOT0:[0-9]+]] = call @mdrtFlagStart(
// CHECK:         gpu.launch
// CHECK:           scf.if
// CHECK:             memref.store %{{[a-z0-9_]+}}, %[[FLAG1]][
// CHECK:         memref.extract_aligned_pointer_as_index %[[FLAG1]]
// CHECK:         %[[SLOT1:[0-9]+]] = call @mdrtFlagStart(
// CHECK:         gpu.launch
// CHECK:         call @mdrtFlagFinish(%[[SLOT0]])
// CHECK:         scf.if
// CHECK:           call @stop(
// CHECK:         call @mdrtFlagFinish(%[[SLOT1]])
// CHECK:         scf.if
// CHECK:           call @stop(
// CHECK-NOT:     md_exec
func.func private @stop(i64)
func.func @two_sets(%waters: memref<?x?xi32, 1>, %groups: memref<?x?xi32, 1>,
                    %x: memref<?x3xf64, 1>, %cell: !md.cell,
                    %v: memref<?x3xf64, 1>) {
  %no = arith.constant false
  %limit = arith.constant 1.0 : f64
  %settles = md_exec.tuple_for %waters, %x, %cell
      coordinates(displacement(1, 0), displacement(2, 0))
      reduce(%no : i1) arity(3) disjoint {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %sa = arith.mulf %a, %a : vector<3xf64>
    %ra = vector.reduction <add>, %sa : vector<3xf64> into f64
    %sb = arith.mulf %b, %b : vector<3xf64>
    %rb = vector.reduction <add>, %sb : vector<3xf64> into f64
    %fa = arith.cmpf ugt, %ra, %limit : f64
    %fb = arith.cmpf ugt, %rb, %limit : f64
    %f = arith.ori %fa, %fb : i1
    md_exec.yield %f : i1
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1> -> i1
  %shake = md_exec.tuple_for %groups, %x, %cell
      coordinates(displacement(1, 0)) reduce(%no : i1) arity(2) disjoint {
  ^bb0(%d: vector<3xf64>):
    %sq = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %far = arith.cmpf ugt, %r2, %limit : f64
    md_exec.yield %far : i1
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1> -> i1
  md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
      outs(%v : memref<?x3xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>):
    %w = arith.addf %v_i, %v_i : vector<3xf64>
    md_exec.yield %w : vector<3xf64>
  }
  %step = arith.constant 1 : i64
  scf.if %settles {
    func.call @stop(%step) : (i64) -> ()
  }
  scf.if %shake {
    func.call @stop(%step) : (i64) -> ()
  }
  return
}

