// RUN: mdir-opt %s --canonicalize | FileCheck %s

// A step takes the effects that its program states (D355,
// #673): canonicalize removes a step whose results are unused if its
// program has none, and keeps one whose program writes a stop or states
// nothing.

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>

md.particle_set @atoms

func.func private @mdrtDeferStops()
    attributes {dyn.effects = {write = ["dyn.stop"]}}

dyn.program @pure(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> !vec
    attributes {effects = {}} {
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  dyn.return %v1 : !vec
}

dyn.program @stops(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> !vec
    attributes {effects = {read = ["dyn.stop"], write = ["dyn.stop"]}} {
  func.call @mdrtDeferStops() : () -> ()
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  dyn.return %v1 : !vec
}

dyn.program @unstated(%v: !vec, %f: !vec, %m: !real, %dt: f64) -> !vec {
  %v1 = dyn.kick %v, %f, %m, %dt : !vec
  dyn.return %v1 : !vec
}

// CHECK-LABEL: func.func @steps(
// CHECK-NOT:     dyn.step @pure
// CHECK:         dyn.step @stops
// CHECK-NEXT:    dyn.step @unstated
// CHECK-NEXT:    return
func.func @steps(%v: !vec, %f: !vec, %m: !real, %dt: f64) {
  %a = dyn.step @pure(%v, %f, %m, %dt) : (!vec, !vec, !real, f64) -> !vec
  %b = dyn.step @stops(%v, %f, %m, %dt) : (!vec, !vec, !real, f64) -> !vec
  %c = dyn.step @unstated(%v, %f, %m, %dt) : (!vec, !vec, !real, f64) -> !vec
  return
}
