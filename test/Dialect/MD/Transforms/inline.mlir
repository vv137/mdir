// RUN: mdir-opt %s --md-inline | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>

md.particle_set @atoms

// CHECK-NOT: md.function @scale
md.function @scale(%v: !vec, %a: f64) -> !vec {
  %r = md.map_particles gather(%v : !vec) {
  ^bb0(%v_i: vector<3xf64>):
    %av = vector.broadcast %a : f64 to vector<3xf64>
    %s = arith.mulf %av, %v_i : vector<3xf64>
    md.yield %s : vector<3xf64>
  } : !vec
  md.return %r : !vec
}

// A function that calls a function.
//
// CHECK-NOT: md.function @twice
md.function @twice(%v: !vec, %a: f64) -> !vec {
  %once = md.call @scale(%v, %a) : (!vec, f64) -> !vec
  %again = md.call @scale(%once, %a) : (!vec, f64) -> !vec
  md.return %again : !vec
}

// CHECK-NOT: dyn.program @move
dyn.program @move(%x: !vec, %v: !vec, %dt: f64) -> !vec {
  %x1 = dyn.drift %x, %v, %dt : !vec
  dyn.return %x1 : !vec
}

// CHECK-LABEL: func.func @caller(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[A:[a-z0-9]+]]: f64)
func.func @caller(%x: !vec, %v: !vec, %a: f64) -> !vec {
  // CHECK:      %[[ONCE:[0-9]+]] = md.map_particles gather(%[[V]] :
  // CHECK:        vector.broadcast %[[A]]
  // CHECK:      %[[AGAIN:[0-9]+]] = md.map_particles gather(%[[ONCE]] :
  // CHECK-NOT:  md.call
  %w = md.call @twice(%v, %a) : (!vec, f64) -> !vec

  // CHECK:      %[[X1:[0-9]+]] = dyn.drift %[[X]], %[[AGAIN]], %[[A]]
  // CHECK-NOT:  dyn.step
  %x1 = dyn.step @move(%x, %w, %a) : (!vec, !vec, f64) -> !vec

  // CHECK:      return %[[X1]]
  return %x1 : !vec
}

// A potential that an evaluation still names stays.
//
// CHECK: md.potential @kept(
md.potential @kept(%x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}

func.func @user(%x: !vec, %cell: !md.cell) -> f64 {
  %u = md.evaluate @kept(%x, %cell) request [energy]
      : (!vec, !md.cell) -> f64
  return %u : f64
}
