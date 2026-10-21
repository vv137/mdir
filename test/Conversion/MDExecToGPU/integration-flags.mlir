// The constraints of a step test the lengths they give in their own kernel
// (D278): a loop over disjoint tuples that writes its
// destinations and has a flag in `reduce` joins the integration run of the
// loops over particles around it (D110), and the member at place 0 of a
// tuple whose kernel yields true sets the loop's flag, in both kernels of
// the run. One event after them; the host reads the flag before the first
// op that uses it.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s

md.particle_set @atoms

// The first launch clears the count of the tuples across warps, where the
// function begins.
//
// CHECK-LABEL: func.func @integrate(
// CHECK:         gpu.launch
// CHECK:         %[[FLAG:[0-9]+]] = call @mdrtFlagMemory(
// CHECK:         gpu.launch {{.*}} function(@integrate_particle_for_l{{[0-9]+}}_integration1_{{[0-9]+}})
// CHECK:             memref.store %{{[a-z0-9_]+}}, %[[FLAG]][
// CHECK:         gpu.launch {{.*}} function(@integrate_particle_for_l{{[0-9]+}}_integration1_{{[0-9]+}})
// CHECK:             memref.store %{{[a-z0-9_]+}}, %[[FLAG]][
// CHECK:         %[[SLOT:[0-9]+]] = call @mdrtFlagStart(
// CHECK-NOT:     gpu.launch
// CHECK:         call @mdrtFlagFinish(%[[SLOT]])
// CHECK:         scf.if
// CHECK:           call @stop(
// CHECK-NOT:     md_exec
func.func private @stop(i64)
func.func @integrate(%pairs: memref<?xi32, 1>, %x: memref<?x3xf64, 1>,
                     %v: memref<?x3xf64, 1>, %y: memref<?x3xf64, 1>,
                     %d: memref<?x3xf64, 1>, %cell: !md.cell) {
  %no = arith.constant false
  md_exec.particle_for ins(%x, %v : memref<?x3xf64, 1>, memref<?x3xf64, 1>)
      outs(%y : memref<?x3xf64, 1>) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
    %s = arith.addf %x_i, %v_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  }
  %any = md_exec.tuple_for %pairs, %y, %cell coordinates(displacement(1, 0))
      outs(%d : memref<?x3xf64, 1>) reduce(%no : i1) arity(2)
      overwrite [true] disjoint {
  ^bb0(%r: vector<3xf64>):
    %sq = arith.mulf %r, %r : vector<3xf64>
    %r2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 1.0 : f64
    %bad = arith.cmpf ugt, %r2, %limit : f64
    %n = arith.negf %r : vector<3xf64>
    md_exec.yield %n, %r, %bad : vector<3xf64>, vector<3xf64>, i1
  } : memref<?xi32, 1>, memref<?x3xf64, 1> -> i1
  md_exec.particle_for ins(%y, %d : memref<?x3xf64, 1>, memref<?x3xf64, 1>)
      outs(%x : memref<?x3xf64, 1>) {
  ^bb0(%y_i: vector<3xf64>, %d_i: vector<3xf64>):
    %s = arith.addf %y_i, %d_i : vector<3xf64>
    md_exec.yield %s : vector<3xf64>
  }
  %step = arith.constant 1 : i64
  scf.if %any {
    func.call @stop(%step) : (i64) -> ()
  }
  return
}
