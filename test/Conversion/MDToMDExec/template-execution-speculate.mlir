// RUN: mdir-opt %s --split-input-file --test-md-exec-template-windows --verify-each | FileCheck %s
// The select form of a template body, and its precondition
// (isTemplateBodySpeculatable). A radial table bin that md-exec-expand-radial
// clamped and a type lookup independent of the geometry run on every lane;
// the cutoff selects the value or +0.
// CHECK-LABEL: func.func @window_energy
// CHECK-NOT: scf.if
// CHECK: %[[IN:.*]] = arith.cmpf olt
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK: vector.load
// CHECK: md.lookup
// CHECK: math.rsqrt
// CHECK: arith.select %[[IN]]
// CHECK-NOT: scf.if
// CHECK: return
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
memref.global "private" constant @table : memref<4x4xf32> = dense<1.0>
func.func @clamped(%list: !list, %x: !x, %h: !md.cell, %lj: memref<?x?xf32>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %c0 = arith.constant 0 : index
    %t = arith.constant 1 : i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 3 : i32
    %shift = arith.constant 21 : i32
    %square = arith.mulf %oo, %oo : vector<3xf32>
    %s = vector.reduction <add>, %square : vector<3xf32> into f32
    %eps = md.lookup %lj[%t, %t] : memref<?x?xf32>, i32, i32 -> f32
    %bits = arith.bitcast %s : f32 to i32
    %key = arith.shrui %bits, %shift : i32
    %k = arith.minsi %hi, %key : i32
    %clamped = arith.maxsi %k, %lo : i32
    %i = arith.index_cast %clamped : i32 to index
    %g = memref.get_global @table : memref<4x4xf32>
    %c = vector.load %g[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    %c3 = vector.extract %c[3] : f32 from vector<4xf32>
    %p = math.fma %c3, %s, %eps : f32
    %inv = math.rsqrt %s : f32
    %v = arith.mulf %p, %inv : f32
    %w = arith.extf %v : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}

// -----
// A bin that is not bounded above may read past the table: guarded.
// CHECK-LABEL: func.func @window_energy
// CHECK: arith.cmpf olt
// CHECK: scf.if
// CHECK: vector.load
// CHECK: scf.yield
// CHECK: else
// CHECK-NOT: arith.select
// CHECK: return
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
memref.global "private" constant @table : memref<4x4xf32> = dense<1.0>
func.func @unbounded(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %c0 = arith.constant 0 : index
    %lo = arith.constant 0 : i32
    %square = arith.mulf %oo, %oo : vector<3xf32>
    %s = vector.reduction <add>, %square : vector<3xf32> into f32
    %bits = arith.bitcast %s : f32 to i32
    %k = arith.maxsi %bits, %lo : i32
    %i = arith.index_cast %k : i32 to index
    %g = memref.get_global @table : memref<4x4xf32>
    %c = vector.load %g[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    %c3 = vector.extract %c[3] : f32 from vector<4xf32>
    %w = arith.extf %c3 : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}

// -----
// A vector load whose last index leaves no room for its four lanes.
// CHECK-LABEL: func.func @window_energy
// CHECK: scf.if
// CHECK: vector.load
// CHECK-NOT: arith.select
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
memref.global "private" constant @table : memref<4x4xf32> = dense<1.0>
func.func @lanes(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %g = memref.get_global @table : memref<4x4xf32>
    %c = vector.load %g[%c0, %c1] : memref<4x4xf32>, vector<4xf32>
    %c3 = vector.extract %c[0] : f32 from vector<4xf32>
    %w = arith.extf %c3 : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}

// -----
// A clamped load of a mutable global is not a fixed table: guarded.
// CHECK-LABEL: func.func @window_energy
// CHECK: scf.if
// CHECK: vector.load
// CHECK-NOT: arith.select
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
memref.global "private" @table : memref<4x4xf32> = dense<1.0>
func.func @mutable(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %c0 = arith.constant 0 : index
    %g = memref.get_global @table : memref<4x4xf32>
    %c = vector.load %g[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
    %c3 = vector.extract %c[0] : f32 from vector<4xf32>
    %w = arith.extf %c3 : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}

// -----
// A table lookup indexed by the geometry: guarded.
// CHECK-LABEL: func.func @window_energy
// CHECK: scf.if
// CHECK: md.lookup
// CHECK-NOT: arith.select
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
func.func @geometric(%list: !list, %x: !x, %h: !md.cell, %lj: memref<?x?xf32>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %square = arith.mulf %oo, %oo : vector<3xf32>
    %s = vector.reduction <add>, %square : vector<3xf32> into f32
    %t = arith.fptosi %s : f32 to i32
    %eps = md.lookup %lj[%t, %t] : memref<?x?xf32>, i32, i32 -> f32
    %w = arith.extf %eps : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}

// -----
// An integer division may trap on a zero divisor outside the cutoff.
// CHECK-LABEL: func.func @window_energy
// CHECK: scf.if
// CHECK: arith.divsi
// CHECK-NOT: arith.select
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!list = !mdrt.segmented_pairs<@atoms>
func.func @division(%list: !list, %x: !x, %h: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  %u = md_exec.template_for %list, %x, %h reduce(%zero : f64)
      pairs [1, 0, 0] cutoff(0.8) extent(0.1) alpha(0.125) water {
  ^bb0(%oo: vector<3xf32>, %oh1: vector<3xf32>, %oh2: vector<3xf32>,
       %jh1: vector<3xf32>, %jh2: vector<3xf32>):
    %one = arith.constant 1 : i32
    %square = arith.mulf %oo, %oo : vector<3xf32>
    %s = vector.reduction <add>, %square : vector<3xf32> into f32
    %t = arith.fptosi %s : f32 to i32
    %q = arith.divsi %one, %t : i32
    %f = arith.sitofp %q : i32 to f32
    %w = arith.extf %f : f32 to f64
    md_exec.yield %w, %oo : f64, vector<3xf32>
  } mixed {
  ^bb0(%a: vector<3xf32>, %b: vector<3xf32>, %c: vector<3xf32>):
    md_exec.yield
  } atoms {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield
  } : !list, !x -> f64
  return %u : f64
}
