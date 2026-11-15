// convert-md-to-md-exec (D417): the windows of the segments
// of D405 become loops over a list of units. The md.kind_pairs gives one
// list, built once for its positions, cell, and cutoff; each window gives
// a md_exec.segment_for with one region, whose table names the site pair of
// each slice, read from the proof of the window's verifier (the displacement
// d_0 + c_p - c_q). The displacement becomes an argument of the region, and
// the arithmetic of the placement stays behind: the loop forms the sites by
// the kinds' weights. A force of a site pair is the force of its first
// place, antisymmetric; an energy is symmetric, with the weight 1/2 of a
// pair visited once.
//
// RUN: mdir-opt %s --split-input-file --convert-md-to-md-exec=skin=0.1 --verify-diagnostics | FileCheck %s

md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>

// CHECK-LABEL: func.func @windows
// CHECK: %[[L:.*]] = md_exec.build_segmented_pairs %{{.*}} kinds(%{{.*}} : !md.relation<@atoms, 3, ordered, @kind>) at(%{{.*}}, %{{.*}}) cutoff(1.000000e+00) skin(1.000000e-01) : !md.field<@atoms, i32>, !md.field<@atoms, 3 x f64> -> !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 1.000000e-01>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
// CHECK-NOT: md_exec.build_segmented_pairs
// CHECK: md_exec.segment_for %[[L]], {{.*}} reduce(%{{.*}} : f64) cutoff(1.000000e+00) weights [5.000000e-01] exchange [symmetric] segment(0, 0) sites [(1, 1) [0], (2, 0) [0]] {
// CHECK-NEXT: ^bb0(%[[R0:.*]]: f64, %[[D0:.*]]: vector<3xf64>, %[[R1:.*]]: f64, %[[D1:.*]]: vector<3xf64>):
// CHECK-NOT: arith.addf
// CHECK: arith.mulf %[[D0]], %[[D0]]
// CHECK: arith.mulf %[[D1]], %[[D1]]
// CHECK: md_exec.yield
// CHECK: md_exec.segment_for %[[L]], {{.*}} outs(%{{.*}} : !md.field<@atoms, 3 x f64>) cutoff(1.000000e+00) exchange [antisymmetric] segment(0, 0) sites [(1, 1) [0]] {
// CHECK-NEXT: ^bb0(%[[R:.*]]: f64, %[[D:.*]]: vector<3xf64>):
// CHECK: md_exec.yield %[[D]] : vector<3xf64>
func.func @windows(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> (f64, !x) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // Two site pairs: (1, 1), and (2, 0) from the anchor of the second.
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %d2 = arith.addf %base, %a2 : vector<3xf64>
    %square = arith.mulf %d, %d : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    %square2 = arith.mulf %d2, %d2 : vector<3xf64>
    %s2 = vector.reduction <add>, %square2 : vector<3xf64> into f64
    md.yield %r2, %s2, %d, %d2 : f64, f64, vector<3xf64>, vector<3xf64>
  } : !s00, !x -> f64
  // The force of the site pair (1, 1): +F at slot 1, -F at slot 3 + 1.
  %f = md.gather_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %zero = arith.constant dense<0.0> : vector<3xf64>
    %minus = arith.negf %d : vector<3xf64>
    md.yield %zero, %d, %zero, %zero, %minus, %zero, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !s00, !x -> !x
  return %u, %f : f64, !x
}

// -----

md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>

// The energy of a site pair that reads the window's frame offsets as well
// as its displacement: not a pair term of its sites.
func.func @geometry(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> f64 {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error @+1 {{has a site pair whose contribution reads the window's geometry other than through its displacement}}
  %u = md.sum_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %square = arith.mulf %d, %a2 : vector<3xf64>
    %r2 = vector.reduction <add>, %square : vector<3xf64> into f64
    md.yield %r2, %d : f64, vector<3xf64>
  } : !s00, !x -> f64
  return %u : f64
}

// -----

md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 3, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>

// A force of a site pair on a third member: not +F and -F at its places.
func.func @forces(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> !x {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error @+1 {{yields forces of the site pair (1, 1) that are not +F at its first place and -F at its second}}
  %f = md.gather_segment_window %s00, %x, %h segment(0, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %fromA = arith.addf %base, %a1 : vector<3xf64>
    %d = arith.subf %fromA, %b1 : vector<3xf64>
    %zero = arith.constant dense<0.0> : vector<3xf64>
    %minus = arith.negf %d : vector<3xf64>
    md.yield %d, %d, %zero, %zero, %minus, %zero, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !s00, !x -> !x
  return %f : !x
}

// -----

// A site pair of a site whose forces are laid out by the frame atoms (no
// site_forces, as md-differentiate gives a body that is not radial): the
// site has no slot of its own there.
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!kind = !md.relation<@atoms, 4, ordered, @kind>
!s00 = !md.relation<@atoms, 6, ordered, @s00>
!s10 = !md.relation<@atoms, 4, ordered, @s10>
!s11 = !md.relation<@atoms, 2, unordered, @s11>
func.func @forces(%x: !x, %h: !md.cell, %types: !i, %kind: !kind) -> !x {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>] : (!i, !kind) -> (!s00, !s10, !s11)
  // expected-error @+1 {{yields forces of the site pair (0, 3), which names a site, in the slots of the frame atoms}}
  %f = md.gather_segment_window %s10, %x, %h segment(1, 0) cutoff(1.0) {
  ^bb0(%base: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %c = arith.constant dense<0.25> : vector<3xf64>
    %s = arith.addf %b1, %b2 : vector<3xf64>
    %w = arith.mulf %c, %s : vector<3xf64>
    %d = arith.subf %base, %w : vector<3xf64>
    %zero = arith.constant dense<0.0> : vector<3xf64>
    %minus = arith.negf %d : vector<3xf64>
    md.yield %d, %minus, %zero, %zero, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : !s10, !x -> !x
  return %f : !x
}
