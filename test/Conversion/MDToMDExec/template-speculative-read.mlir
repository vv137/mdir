// RUN: mdir-opt %s -split-input-file --test-md-exec-template-windows --verify-each | FileCheck %s
// The read of a force pair's constant table goes before the pair's cutoff
// guard only when it cannot fault: the base is a constant memref.global and
// every index is clamped into [0, extent - width]. Each case below changes
// one of these conditions. The last cases have the shape of a radial table:
// rows of four coefficients, indexed by the form md-exec-expand-radial
// emits in a template body or by a signed clamp.

// A scalar read of a constant table clamped into [0, 15] goes before the guard.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK: memref.load
// CHECK: scf.if
// CHECK-NOT: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %v = memref.load %table[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A two-wide read clamped into [0, 14] (extent minus width) goes before the guard.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK: vector.load
// CHECK: scf.if
// CHECK-NOT: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 14 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %w = vector.load %table[%i] : memref<16xf64>, vector<2xf64>
    %w0 = vector.extract %w[0] : f64 from vector<2xf64>
    %w1 = vector.extract %w[1] : f64 from vector<2xf64>
    %v = arith.addf %w0, %w1 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// An unclamped index stays inside the guard.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: memref.load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %i = arith.index_cast %k : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %v = memref.load %table[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A two-wide read clamped to 15, above extent minus width, stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: vector.load
// CHECK: scf.if
// CHECK: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %w = vector.load %table[%i] : memref<16xf64>, vector<2xf64>
    %w0 = vector.extract %w[0] : f64 from vector<2xf64>
    %w1 = vector.extract %w[1] : f64 from vector<2xf64>
    %v = arith.addf %w0, %w1 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A scalar read clamped to 16, past the extent, stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: memref.load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 16 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %v = memref.load %table[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A lower bound below zero stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: memref.load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant -1 : i32
    %hi = arith.constant 15 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %v = memref.load %table[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A read of a global that is not constant stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: memref.load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %table = memref.get_global @table : memref<16xf64>
    %v = memref.load %table[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A read of a kernel argument stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<16xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: memref.load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<16xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 15 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index

    %v = memref.load %arg[%i] : memref<16xf64>
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A radial row, (minui(maxui(bits, lo), hi) >> 16) - base, goes before the guard.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK: vector.load
// CHECK: scf.if
// CHECK-NOT: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %bits = arith.bitcast %scaled : f64 to i64
    %lo = arith.constant 978255872 : i64
    %hi = arith.constant 1062207487 : i64
    %low = arith.maxui %bits, %lo : i64
    %c = arith.minui %low, %hi : i64
    %shift = arith.constant 16 : i64
    %key = arith.shrui %c, %shift : i64
    %b = arith.constant 14927 : i64
    %row = arith.subi %key, %b overflow<nsw, nuw> : i64
    %i = arith.index_cast %row : i64 to index
    %column = arith.constant 0 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A radial row whose base is above lo >> 16 (a negative row) stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: vector.load
// CHECK: scf.if
// CHECK: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %bits = arith.bitcast %scaled : f64 to i64
    %lo = arith.constant 978255872 : i64
    %hi = arith.constant 1062207487 : i64
    %low = arith.maxui %bits, %lo : i64
    %c = arith.minui %low, %hi : i64
    %shift = arith.constant 16 : i64
    %key = arith.shrui %c, %shift : i64
    %b = arith.constant 14928 : i64
    %row = arith.subi %key, %b overflow<nsw, nuw> : i64
    %i = arith.index_cast %row : i64 to index
    %column = arith.constant 0 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A radial row whose hi >> 16 is one row past the table stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: vector.load
// CHECK: scf.if
// CHECK: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %bits = arith.bitcast %scaled : f64 to i64
    %lo = arith.constant 978255872 : i64
    %hi = arith.constant 1062273023 : i64
    %low = arith.maxui %bits, %lo : i64
    %c = arith.minui %low, %hi : i64
    %shift = arith.constant 16 : i64
    %key = arith.shrui %c, %shift : i64
    %b = arith.constant 14927 : i64
    %row = arith.subi %key, %b overflow<nsw, nuw> : i64
    %i = arith.index_cast %row : i64 to index
    %column = arith.constant 0 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A row read four wide from column 1 stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: vector.load
// CHECK: scf.if
// CHECK: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %bits = arith.bitcast %scaled : f64 to i64
    %lo = arith.constant 978255872 : i64
    %hi = arith.constant 1062207487 : i64
    %low = arith.maxui %bits, %lo : i64
    %c = arith.minui %low, %hi : i64
    %shift = arith.constant 16 : i64
    %key = arith.shrui %c, %shift : i64
    %b = arith.constant 14927 : i64
    %row = arith.subi %key, %b overflow<nsw, nuw> : i64
    %i = arith.index_cast %row : i64 to index
    %column = arith.constant 1 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// A signed clamp into the rows, read four wide from column 0, goes before the guard.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK: vector.load
// CHECK: scf.if
// CHECK-NOT: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 1280 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %column = arith.constant 0 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}

// -----

// The same read from column 1 stays inside.
md.particle_set @atoms
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
memref.global "private" constant @table : memref<1281x4xf64> = dense<1.0>
// CHECK-LABEL: func.func @window_forces
// CHECK-NOT: vector.load
// CHECK: scf.if
// CHECK: vector.load
// CHECK: } else {
func.func @entry(%pairs: !mdrt.segmented_pairs<@atoms>, %x: !x, %h: !md.cell, %arg: memref<1281x4xf64>) -> !x {
  %f0 = md_exec.zeros : !x
  %f = md_exec.template_for %pairs, %x, %h outs(%f0 : !x) pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75) water {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %as = arith.addf %a1, %a2 : vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pa = arith.mulf %weight, %as : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %fromA = arith.addf %base, %pa : vector<3xf64>
    %d = arith.subf %fromA, %pb : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %scale = arith.constant 10.0 : f64
    %scaled = arith.mulf %r2, %scale : f64
    %k = arith.fptosi %scaled : f64 to i32
    %zero = arith.constant 0 : i32
    %top = arith.constant 15 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 1280 : i32
    %low = arith.maxsi %k, %lo : i32
    %c = arith.minsi %low, %hi : i32
    %i = arith.index_cast %c : i32 to index
    %column = arith.constant 1 : index
    %table = memref.get_global @table : memref<1281x4xf64>
    %w = vector.load %table[%i, %column] : memref<1281x4xf64>, vector<4xf64>
    %w0 = vector.extract %w[0] : f64 from vector<4xf64>
    %w3 = vector.extract %w[3] : f64 from vector<4xf64>
    %v = arith.addf %w0, %w3 : f64
    %g = vector.broadcast %v : f64 to vector<3xf64>
    %term = arith.mulf %g, %d : vector<3xf64>
    %back = arith.negf %term : vector<3xf64>
    %z = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %term, %z, %z, %back, %z, %z, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } mixed {
  ^bb0(%m0: vector<3xf64>, %m1: vector<3xf64>, %m2: vector<3xf64>):
    md_exec.yield
  } atoms {
  ^bb0(%s0: vector<3xf64>):
    md_exec.yield
  } : !mdrt.segmented_pairs<@atoms>, !x -> !x
  func.return %f : !x
}
