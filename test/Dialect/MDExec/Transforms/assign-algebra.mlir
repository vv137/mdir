// md-exec-assign-algebra (D301): the algebra of each value
// that an op reduces, `any_order_exact` for an integer, and for a floating-point
// number `any_order` by default and `fixed_order` in the deterministic mode;
// md-exec-assign-storage carries it onto the storage form, and the printed
// form parses again.
//
// RUN: mdir-opt %s --md-exec-assign-algebra | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true \
// RUN: | FileCheck %s --check-prefix=DET
// RUN: mdir-opt %s --md-exec-assign-algebra | mdir-opt | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra \
// RUN:     --md-exec-assign-storage \
// RUN: | FileCheck %s --check-prefix=STORAGE
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-check-algebra -o /dev/null

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!grid = !md.table<2, f32>
md.particle_set @atoms

// The deterministic mode marks the module, whose verifier then refuses an
// `any_order` floating-point sum (test/Dialect/MDExec/algebra-invalid.mlir).
// CHECK-NOT: md_exec.deterministic
// DET: module attributes {md_exec.deterministic}

// A sum and a flag.
// CHECK-LABEL: func.func @particles(
// CHECK: } {md_exec.algebra = #md_exec.algebra<any_order, any_order_exact>}
// DET-LABEL: func.func @particles(
// DET: } {md_exec.algebra = #md_exec.algebra<fixed_order, any_order_exact>}
// STORAGE-LABEL: func.func @particles(
// STORAGE: } {md_exec.algebra = #md_exec.algebra<any_order, any_order_exact>}
func.func @particles(%v: !vec) -> (f64, i1) {
  %zero = arith.constant 0.0 : f64
  %no = arith.constant false
  %k, %bad = md_exec.particle_for ins(%v : !vec) reduce(%zero, %no : f64, i1) {
  ^bb0(%v_i: vector<3xf32>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf32>
    %v2 = vector.reduction <add>, %sq : vector<3xf32> into f32
    %wide = arith.extf %v2 : f32 to f64
    %nan = arith.cmpf uno, %v2, %v2 : f32
    md_exec.yield %wide, %nan : f64, i1
  } -> f64, i1
  return %k, %bad : f64, i1
}

// The reciprocal sum: the spreading into its grid (in the type of its
// forces), its energy, its virial.
// CHECK-LABEL: func.func @reciprocal(
// CHECK: md_exec.reciprocal {{.*}} {md_exec.algebra = #md_exec.algebra<any_order, any_order, any_order>}
// DET-LABEL: func.func @reciprocal(
// DET: md_exec.reciprocal {{.*}} {md_exec.algebra = #md_exec.algebra<fixed_order, fixed_order, fixed_order>}
// STORAGE-LABEL: func.func @reciprocal(
// STORAGE: md_exec.reciprocal {{.*}} {md_exec.algebra = #md_exec.algebra<any_order, any_order, any_order>}
func.func @reciprocal(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !grid)
    -> (f64, !vec) {
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([16, 16, 16]) order(4) beta(3.0) coulomb(138.9)
      : !vec, !real, !grid -> f64, vector<9xf64>, !vec
  return %u, %f : f64, !vec
}

// A scatter-accumulate.
// CHECK-LABEL: func.func @scatter(
// CHECK: md_exec.scatter_add {{.*}} {md_exec.algebra = #md_exec.algebra<any_order>}
// DET-LABEL: func.func @scatter(
// DET: md_exec.scatter_add {{.*}} {md_exec.algebra = #md_exec.algebra<fixed_order>}
// STORAGE-LABEL: func.func @scatter(
// STORAGE: md_exec.scatter_add {{.*}} {md_exec.algebra = #md_exec.algebra<any_order>}
func.func @scatter(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, i32>,
                   %a: !md.field<@atoms, 3 x f64>) -> !md.field<@atoms, 3 x f64> {
  %a1 = md_exec.scatter_add %v at %i into %a
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return %a1 : !md.field<@atoms, 3 x f64>
}
