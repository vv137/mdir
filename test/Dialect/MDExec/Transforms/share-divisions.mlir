// RUN: mdir-opt %s --md-exec-share-divisions=rsqrt=true | FileCheck %s
// The host's form, without rsqrt: a root is a divisor like any other.
// RUN: mdir-opt %s --md-exec-share-divisions | FileCheck %s --check-prefix=HOST
// Without the pass (fast_math = false, or a pipeline without it), the
// kernels keep their divisions as written.
// RUN: mdir-opt %s | FileCheck %s --check-prefix=OFF

!vec = !md.field<@atoms, 3 x f64>
!vec32 = !md.field<@atoms, 3 x f32>
!nl  = !mdrt.neighbors<@atoms>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>
!bonds_inc = !mdrt.incidence<@atoms, 2, @bonds>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

// The f64 kernel of the direct sum of Ewald and Lennard-Jones: a division
// by sqrt(r2), then two divisions by r2. The division by the root becomes
// a product by rsqrt(r2); the two divisions by r2 share one reciprocal, at
// the start of the block (r2 is an argument of the block), which the
// division 1/r2 becomes. No other division is left.
//
// CHECK-LABEL: func.func @ewald(
// CHECK:         md_exec.pair_for
// CHECK:         ^bb0(%[[R2:[a-z0-9]+]]: f64,
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      %[[INV:.+]] = arith.divf %[[ONE]], %[[R2]] : f64
// CHECK:           %[[R:.+]] = math.sqrt %[[R2]] : f64
// CHECK-NEXT:      %[[ROOT:.+]] = math.rsqrt %[[R2]] : f64
// CHECK:           arith.mulf %{{.+}}, %[[ROOT]] : f64
// CHECK:           arith.mulf %{{.+}}, %[[INV]] : f64
// CHECK:           arith.mulf %[[INV]], %[[INV]] : f64
// CHECK-NOT:       arith.divf
// CHECK:           md_exec.yield
//
// HOST-LABEL: func.func @ewald(
// HOST:          ^bb0(%[[R2:[a-z0-9]+]]: f64,
// HOST-NEXT:       %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// HOST-NEXT:       %[[INV:.+]] = arith.divf %[[ONE]], %[[R2]] : f64
// HOST:            %[[R:.+]] = math.sqrt %[[R2]] : f64
// HOST:            arith.divf %{{.+}}, %[[R]] : f64
// HOST:            arith.mulf %{{.+}}, %[[INV]] : f64
// HOST-NOT:        arith.divf
// HOST-NOT:        math.rsqrt
// HOST:            md_exec.yield
//
// OFF-LABEL: func.func @ewald(
// OFF:           %[[R:.+]] = math.sqrt
// OFF:           arith.divf %{{.+}}, %[[R]] : f64
// OFF-COUNT-2:   arith.divf
// OFF-NOT:       math.rsqrt
// OFF:           md_exec.yield
func.func @ewald(%x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(8.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(8.0) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %beta = arith.constant 0.35 : f64
    %c = arith.constant 138.9 : f64
    %one = arith.constant 1.0 : f64
    %r = math.sqrt %r2 : f64
    %br = arith.mulf %r, %beta : f64
    %erfc = math.erfc %br : f64
    %qe = arith.mulf %erfc, %c : f64
    %coulomb = arith.divf %qe, %r : f64
    %lj = arith.divf %c, %r2 : f64
    %inverse = arith.divf %one, %r2 : f64
    %s6 = arith.mulf %inverse, %inverse : f64
    %a = arith.addf %coulomb, %lj : f64
    %b = arith.addf %a, %s6 : f64
    %s = vector.broadcast %b : f64 to vector<3xf64>
    %force = arith.mulf %s, %d : vector<3xf64>
    md_exec.yield %force : vector<3xf64>
  } : !nl, !vec -> !vec
  return %f : !vec
}

// Two divisions by a value of the block share its reciprocal, which
// follows the definition of the value.
//
// CHECK-LABEL: func.func @shared(
// CHECK:           %[[Y:.+]] = arith.addf
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      %[[INV:.+]] = arith.divf %[[ONE]], %[[Y]] : f64
// CHECK:           arith.mulf %{{.+}}, %[[INV]] : f64
// CHECK:           arith.mulf %{{.+}}, %[[INV]] : f64
// CHECK-NOT:       arith.divf
// CHECK:           md_exec.yield
func.func @shared(%x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(8.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(8.0) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %h = arith.constant 0.1 : f64
    %y = arith.addf %r2, %h : f64
    %p = arith.divf %r2, %y : f64
    %q = arith.divf %h, %y : f64
    %b = arith.addf %p, %q : f64
    %s = vector.broadcast %b : f64 to vector<3xf64>
    %force = arith.mulf %s, %d : vector<3xf64>
    md_exec.yield %force : vector<3xf64>
  } : !nl, !vec -> !vec
  return %f : !vec
}

// 1/sqrt(y) is the rsqrt itself, and the root, which nothing else takes,
// goes. Without rsqrt, two divisions by the root share its reciprocal.
//
// CHECK-LABEL: func.func @root(
// CHECK:           %[[Y:.+]] = arith.addf
// CHECK-NEXT:      %[[ROOT:.+]] = math.rsqrt %[[Y]] : f64
// CHECK-NOT:       math.sqrt
// CHECK:           arith.mulf %{{.+}}, %[[ROOT]] : f64
// CHECK:           arith.addf %[[ROOT]], %{{.+}} : f64
// CHECK-NOT:       arith.divf
// CHECK:           md_exec.yield
//
// HOST-LABEL: func.func @root(
// HOST:            %[[R:.+]] = math.sqrt
// HOST-NEXT:       %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// HOST-NEXT:       %[[INV:.+]] = arith.divf %[[ONE]], %[[R]] : f64
// HOST:            arith.mulf %{{.+}}, %[[INV]] : f64
// HOST:            arith.addf %[[INV]], %{{.+}} : f64
// HOST-NOT:        arith.divf
// HOST:            md_exec.yield
func.func @root(%x: !vec, %cell: !md.cell) -> !vec {
  %f0 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(8.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(8.0) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %h = arith.constant 0.1 : f64
    %one = arith.constant 1.0 : f64
    %y = arith.addf %r2, %h : f64
    %r = math.sqrt %y : f64
    %p = arith.divf %h, %r : f64
    %q = arith.divf %one, %r : f64
    %b = arith.addf %q, %p : f64
    %s = vector.broadcast %b : f64 to vector<3xf64>
    %force = arith.mulf %s, %d : vector<3xf64>
    md_exec.yield %force : vector<3xf64>
  } : !nl, !vec -> !vec
  return %f : !vec
}

// A lone division becomes a product by the reciprocal too, so that its
// rounding does not depend on how many divisions share the divisor. Where
// the preconditions fail the divisions stay: vector divisions, a kernel in
// f32, and a kernel of a loop over tuples, each with a root and a shared
// divisor.
//
// CHECK-LABEL: func.func @kept(
// CHECK:         md_exec.pair_for
// CHECK:           %[[Y:.+]] = arith.addf
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:      %[[INV:.+]] = arith.divf %[[ONE]], %[[Y]] : f64
// CHECK-NEXT:      arith.mulf %{{.+}}, %[[INV]] : f64
// CHECK:           arith.divf %{{.+}}, %{{.+}} : vector<3xf64>
// CHECK:           arith.divf %{{.+}}, %{{.+}} : vector<3xf64>
// CHECK:           md_exec.yield
// CHECK:         md_exec.pair_for
// CHECK-COUNT-3:   arith.divf %{{.+}}, %{{.+}} : f32
// CHECK-NOT:       math.rsqrt
// CHECK:           md_exec.yield
// CHECK:         md_exec.tuple_for
// CHECK-COUNT-3:   arith.divf %{{.+}}, %{{.+}} : f64
// CHECK-NOT:       math.rsqrt
// CHECK:           md_exec.yield
func.func @kept(%x: !vec, %cell: !md.cell, %x32: !vec32,
                %bonds_inc: !bonds_inc) -> (!vec, !vec32, !vec) {
  %f0 = md_exec.zeros : !vec
  %coverage_cells = md_exec.build_cells %x, %cell width(8.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %coverage_cells, %x, %cell
      cutoff(8.0) skin(0.0) kind(matrix) width(64)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %h = arith.constant 0.1 : f64
    %y = arith.addf %r2, %h : f64
    %q = arith.divf %h, %y : f64
    %s = vector.broadcast %q : f64 to vector<3xf64>
    %t = vector.broadcast %r2 : f64 to vector<3xf64>
    %u = arith.divf %d, %t : vector<3xf64>
    %v = arith.divf %u, %t : vector<3xf64>
    %force = arith.mulf %s, %v : vector<3xf64>
    md_exec.yield %force : vector<3xf64>
  } : !nl, !vec -> !vec
  %g0 = md_exec.zeros : !vec32
  %g = md_exec.pair_for %nl, %x32, %cell outs(%g0 : !vec32) cutoff(8.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %h = arith.constant 0.1 : f32
    %r = math.sqrt %r2 : f32
    %o = arith.divf %h, %r : f32
    %p = arith.divf %o, %r2 : f32
    %q = arith.divf %p, %r2 : f32
    %s = vector.broadcast %q : f32 to vector<3xf32>
    %force = arith.mulf %s, %d : vector<3xf32>
    md_exec.yield %force : vector<3xf32>
  } : !nl, !vec32 -> !vec32
  %e = md_exec.tuple_for %bonds_inc, %x, %cell
      coordinates(displacement(0, 1)) outs(%f0 : !vec) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %h = arith.constant 0.1 : f64
    %r2 = vector.reduction <add>, %d : vector<3xf64> into f64
    %r = math.sqrt %r2 : f64
    %o = arith.divf %h, %r : f64
    %p = arith.divf %o, %r2 : f64
    %q = arith.divf %p, %r2 : f64
    %s = vector.broadcast %q : f64 to vector<3xf64>
    %force = arith.mulf %s, %d : vector<3xf64>
    %z = arith.negf %force : vector<3xf64>
    md_exec.yield %force, %z : vector<3xf64>, vector<3xf64>
  } : !bonds_inc, !vec -> !vec
  return %f, %g, %e : !vec, !vec32, !vec
}
