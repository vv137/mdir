// RUN: mdir-opt %s --md-exec-simplify-distance=radial=true --canonicalize --verify-each | FileCheck %s
// RUN: mdir-opt %s --md-exec-simplify-distance=radial=true --md-exec-assign-precision=mode=mixed --md-exec-expand-radial --canonicalize --verify-each | FileCheck %s --check-prefix=TABLE
// Several site pairs share the water region. Their internal squared
// distances and outputs survive successive common-interface rewrites.
// CHECK-LABEL: func.func @windows
// CHECK: md_exec.template_for
// CHECK-COUNT-2: md_exec.radial
// CHECK: mixed
// CHECK: md_exec.radial
// CHECK: atoms
// CHECK: md_exec.radial
// TABLE: memref.global "private" constant @md_radial_table
// TABLE-LABEL: func.func @windows
// TABLE-NOT: md_exec.radial
// TABLE: vector.load
// TABLE-NOT: math.erfc
// TABLE: return

md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
func.func @windows(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %beta = arith.constant 4.32 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [2, 1, 1] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf64>, %oh1: vector<3xf64>, %oh2: vector<3xf64>,
       %jh1: vector<3xf64>, %jh2: vector<3xf64>):
    %a = arith.addf %oo, %oh1 : vector<3xf64>
    %d1 = arith.subf %a, %jh1 : vector<3xf64>
    %b = arith.addf %oo, %oh2 : vector<3xf64>
    %d2 = arith.subf %b, %jh2 : vector<3xf64>
    %v1 = arith.mulf %d1, %d1 : vector<3xf64>
    %s1 = vector.reduction <add>, %v1 : vector<3xf64> into f64
    %r1 = math.sqrt %s1 : f64
    %br1 = arith.mulf %beta, %r1 : f64
    %e1 = math.erfc %br1 : f64
    %u1 = arith.divf %e1, %r1 : f64
    %v2 = arith.mulf %d2, %d2 : vector<3xf64>
    %s2 = vector.reduction <add>, %v2 : vector<3xf64> into f64
    %r2 = math.sqrt %s2 : f64
    %br2 = arith.mulf %beta, %r2 : f64
    %e2 = math.erfc %br2 : f64
    %u2 = arith.divf %e2, %r2 : f64
    md_exec.yield %u1, %u2, %d1, %d2 : f64, f64, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%ao: vector<3xf64>, %oh1: vector<3xf64>, %oh2: vector<3xf64>):
    %d = arith.subf %ao, %oh1 : vector<3xf64>
    %v = arith.mulf %d, %d : vector<3xf64>
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    %r = math.sqrt %s : f64
    %br = arith.mulf %beta, %r : f64
    %e = math.erfc %br : f64
    %u0 = arith.divf %e, %r : f64
    md_exec.yield %u0, %d : f64, vector<3xf64>
  } atoms {
  ^bb0(%d: vector<3xf64>):
    %v = arith.mulf %d, %d : vector<3xf64>
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    %r = math.sqrt %s : f64
    %br = arith.mulf %beta, %r : f64
    %e = math.erfc %br : f64
    %u0 = arith.divf %e, %r : f64
    md_exec.yield %u0, %d : f64, vector<3xf64>
  } : !list, !x -> f64
  return %u : f64
}

// The unchanged D94 admission rejects a radial function with a pole at
// the cutoff. Its original f32 expression remains the fallback (D159).
// CHECK-LABEL: func.func @fallback
// CHECK-NOT: md_exec.radial
// CHECK: math.exp
// CHECK: return
// TABLE-LABEL: func.func @fallback
// TABLE-NOT: md_exec.radial
// TABLE: math.exp {{.*}} : f32
// TABLE: return
func.func @fallback(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %cut = arith.constant 0.8 : f64
  %sigma = arith.constant 0.1 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [0, 0, 1] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %c: vector<3xf64>,
       %d: vector<3xf64>, %e: vector<3xf64>):
    md_exec.yield
  } mixed {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %c: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf64>):
    %v = arith.mulf %d, %d : vector<3xf64>
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    %r = math.sqrt %s : f64
    %denominator = arith.subf %r, %cut : f64
    %argument = arith.divf %sigma, %denominator : f64
    %value = math.exp %argument : f64
    md_exec.yield %value, %d : f64, vector<3xf64>
  } : !list, !x -> f64
  return %u : f64
}
