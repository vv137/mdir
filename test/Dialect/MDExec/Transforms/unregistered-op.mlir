// An op of no registered dialect, which --allow-unregistered-dialect
// admits, in a function with fields and in the kernel of a loop (#395).
// `md-exec-assign-storage` and `md-exec-fold-tables` took its dialect with
// `isa<...Dialect>`, which asserts on the null dialect of such an op; they
// now take it as an op of another dialect and leave it where it is.
//
// RUN: mdir-opt --allow-unregistered-dialect --md-exec-assign-storage %s \
// RUN:   | FileCheck %s --check-prefix=STORAGE
// RUN: mdir-opt --allow-unregistered-dialect --md-exec-fold-tables %s \
// RUN:   | FileCheck %s --check-prefix=TABLES

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>

md.particle_set @atoms

// STORAGE-LABEL: func.func @kick(
// STORAGE-SAME:    memref<?x3xf64{{.*}}>
// STORAGE:         "test.unknown"() : () -> ()
// STORAGE:         md_exec.particle_for
// STORAGE:           "test.scale"(%{{[0-9]+}}) : (f64) -> f64
// TABLES-LABEL:  func.func @kick(
// TABLES:          "test.unknown"() : () -> ()
// TABLES:          md_exec.particle_for
// TABLES:            "test.scale"(%{{[0-9]+}}) : (f64) -> f64
func.func @kick(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> !vec {
  "test.unknown"() : () -> ()
  %v0 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%v0 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %s2 = "test.scale"(%s) : (f64) -> f64
    %sv = vector.broadcast %s2 : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %v1 : !vec
}
