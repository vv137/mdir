// The refusals of the ops of md_exec over a list of units
// (D417): the builds state no more than they guarantee, take
// the kinds of their type, and their anchors with a radius; the loop adds at
// both ends, needs each segment it visits to meet its requirement, and its
// regions follow the rules of md_alg.traverse_pairs (shared, one case here).
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// The build states more than it guarantees: a stable order in a segment.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, stable(group, search), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{has the type of a list whose segment 1 guarantees 'unique, minimum, removed, stable(group, search)', more than its build does}}
  %b = md_exec.build_segmented_pairs %t kinds(%kind : !k) at(%x, %cell) cutoff(0.9) skin(0.1) : !i, !x -> !sp
  return
}

// -----

// A build without the kind that its type takes.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{takes 0 kinds, but its type has 1}}
  %b = md_exec.build_segmented_pairs %t at(%x, %cell) cutoff(0.9) skin(0.1) : !i, !x -> !sp
  return
}

// -----

// A kind of three members a copy for a descriptor of four places.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp, %kind3: !k3) {
  // expected-error @+1 {{takes kind 0 with 3 members a copy, but its descriptor has 4 places}}
  %b = md_exec.empty_segmented_pairs %t kinds(%kind3 : !k3) cutoff(0.9) skin(0.1) : !i -> !sp
  return
}

// -----

// Anchors with a radius that is not positive.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp, %a: !md.relation<@atoms, 2, ordered>) {
  // expected-error @+1 {{expected a positive finite radius}}
  %b = md_exec.empty_segmented_pairs %t kinds(%kind : !k) anchor(%a : !md.relation<@atoms, 2, ordered>) radius(0.0) cutoff(0.9) skin(0.1) : !i -> !sp
  return
}

// -----

// A refresh whose inner skin exceeds its skin.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{expected finite skins with 0 <= prune_skin <= skin}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.2) : !sp, !x
  return
}

// -----

// A force that is not antisymmetric under the exchange of the units.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{adds at both ends of a pair, which needs every destination symmetric or antisymmetric and every sum symmetric; value 1 is antisymmetric}}
  %f, %u = md_exec.segment_for %l, %x, %cell ins(%q : !q) outs(%f0 : !x) reduce(%u0 : f64)
      cutoff(0.9) weights [0.5] exchange [antisymmetric, antisymmetric]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
      md_exec.yield %d, %r2 : vector<3xf64>, f64
    } : !sp, !x -> !x, f64
  return
}

// -----

// A loop over a segment whose exclusions are only marked.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, masked, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{takes a list whose segment(1, 1) does not meet its requirement}}
  %f, %u = md_exec.segment_for %l, %x, %cell ins(%q : !q) outs(%f0 : !x) reduce(%u0 : f64)
      cutoff(0.9) weights [0.5] exchange [antisymmetric, symmetric]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
      md_exec.yield %d, %r2 : vector<3xf64>, f64
    } : !sp, !x -> !x, f64
  return
}

// -----

// A region that yields one value fewer than its mask.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
md.tuple_set @kind on(@atoms) arity(4) orientation(ordered) disjoint
!x = !md.field<@atoms, 3 x f64>
!i = !md.field<@atoms, i32>
!q = !md.field<@atoms, f64>
!k = !md.relation<@atoms, 4, ordered, @kind>
!k3 = !md.relation<@atoms, 3, ordered, @kind>
func.func @f(%t: !i, %kind: !k, %x: !x, %cell: !md.cell, %q: !q, %f0: !x, %u0: f64, %l: !sp) {
  // expected-error @+1 {{expected the region of segment(1, 1) to yield the values of each mask}}
  %f, %u = md_exec.segment_for %l, %x, %cell ins(%q : !q) outs(%f0 : !x) reduce(%u0 : f64)
      cutoff(0.9) weights [0.5] exchange [antisymmetric, symmetric]
    segment(1, 1) sites [(0, 0) [0, 1]] {
    ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
      md_exec.yield %d : vector<3xf64>
    } : !sp, !x -> !x, f64
  return
}
