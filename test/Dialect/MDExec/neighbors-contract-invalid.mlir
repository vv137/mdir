// What the verifier refuses of the layout contract of a neighbor structure
// (D[neighbors-contract], #881): a consumer whose requirement the type does
// not meet, a build whose type states more than it guarantees, and a cast
// that strengthens. Each case is a bug that a pass could make.
//
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
md.particle_set @atoms

// A loop that takes each pair once over a directed list counts every pair
// twice: the matrix holds both orders.
func.func @unique_over_directed(%x: !vec, %cell: !md.cell) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{takes a neighbor structure that does not meet its requirement: the traversal 'unique' is required, and the structure is 'directed'}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> f64
  return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
md.particle_set @atoms

// A directed loop over groups misses half of each pair.
func.func @directed_over_groups(%x: !vec, %cell: !md.cell, %f0: !vec) -> !vec {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec -> !groups
  // expected-error@+1 {{the traversal 'directed' is required, and the structure is 'unique'}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !groups, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// Under owner_only each center adds its partners in its registers and
// stores once, so its partners must be its own row: grouped by owner. A
// list that is not grouped would lose all but one part of a row.
func.func @owner_only_over_ungrouped(%x: !vec, %cell: !md.cell, %f0: !vec) -> !vec {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, none>
  // expected-error@+1 {{the organization 'grouped_by(owner)' is required, and the structure guarantees 'none'}}
  %f = md_exec.pair_for %w, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, none>, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// The groups are grouped by group, not by owner: a list of 16 centers.
func.func @owner_only_over_groups(%x: !vec, %cell: !md.cell, %f0: !vec) -> !vec {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec
      -> !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  %w = md_exec.weaken_neighbors %nl
      : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
      to !mdrt.neighbors<@atoms, unique, minimum, removed, grouped_by(group)>
  // expected-error@+1 {{the traversal 'directed' is required, and the structure is 'unique'}}
  %f = md_exec.pair_for %w, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, unique, minimum, removed, grouped_by(group)>, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// The key of the grouping alone: a directed list grouped by group (no build
// makes it, a cast of a future build could) does not meet owner_only.
func.func @owner_only_over_group_key(%x: !vec, %cell: !md.cell, %f0: !vec,
    %nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(group, search)>) -> !vec {
  // expected-error@+1 {{the organization 'grouped_by(owner)' is required, and the structure guarantees 'stable(group, search)'}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(group, search)>, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A fixed_order sum (D301) must use the order the IR states: the list must
// be stable, and grouped is not enough. This is the refusal that a weakening
// of the deterministic mode's matrix would meet.
func.func @fixed_order_over_grouped(%x: !vec, %cell: !md.cell) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the organization 'stable(owner, search)' is required, and the structure guarantees 'grouped_by(owner)'}}
  %u = md_exec.pair_for %w, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } {md_exec.algebra = #md_exec.algebra<fixed_order>}
    : !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>, !vec -> f64
  return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// The same with the order by another key: the item is not the order of the
// search.
func.func @fixed_order_other_key(%x: !vec, %cell: !md.cell,
    %nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, item)>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the organization 'stable(owner, search)' is required, and the structure guarantees 'stable(owner, item)'}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } {md_exec.algebra = #md_exec.algebra<fixed_order>}
    : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, item)>, !vec -> f64
  return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A loop with plain differences over a list of minimum images (or the
// reverse) takes other pairs than the list holds.
func.func @images(%x: !vec, %cell: !md.cell) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the images 'explicit' are required, and the structure's are 'minimum'}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> f64
  return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// The loop over groups reads the bits of its entries and skips nothing: an
// entry that stands for an excluded pair, as a device's matrix enters it,
// would be computed.
func.func @exclusions(%x: !vec, %cell: !md.cell) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec
      -> !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  %w = md_exec.weaken_neighbors %nl
      : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
      to !mdrt.neighbors<@atoms, unique, minimum, masked, stable(group, search)>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the exclusions 'removed' are required, and the structure's are 'masked'}}
  %u = md_exec.pair_for %w, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms, unique, minimum, masked, stable(group, search)>, !vec -> f64
  return %u : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// The triplets are found in the rows of a matrix (D160).
func.func @triplets_over_groups(%x: !vec, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec
      -> !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  // expected-error@+1 {{the traversal 'directed' is required, and the structure is 'unique'}}
  %t = md_exec.build_triplets %nl, %x, %cell cutoff(1.0)
      : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>, !vec
      -> !md.relation<@atoms, 3, reversal>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A build of a matrix cannot claim each pair once.
func.func @matrix_typed_as_groups(%x: !vec, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  // expected-error@+1 {{has the type of a structure that guarantees 'unique, minimum, removed, stable(group, search)', more than its build does: 'directed, minimum, masked, stable(owner, search)'}}
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !mdrt.cells<@atoms>, !vec
      -> !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A pass that turns a matrix into groups and forgets the type.
func.func @groups_typed_as_matrix(%x: !vec, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  // expected-error@+1 {{has the type of a structure that guarantees 'directed, minimum, masked, stable(owner, search)', more than its build does: 'unique, minimum, removed, stable(group, search)'}}
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(groups) width(64) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  return
}

// -----

md.particle_set @atoms

// The same for an empty structure.
func.func @empty_groups_typed_as_matrix() {
  // expected-error@+1 {{has the type of a structure that guarantees 'directed, minimum, masked, stable(owner, search)', more than its build does: 'unique, minimum, removed, stable(group, search)'}}
  %nl = md_exec.empty_neighbors kind(groups) width(64) : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A build of a matrix with explicit images is not one of minimum images.
func.func @explicit_typed_as_minimum(%x: !vec, %cell: !md.cell, %n: index) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  // expected-error@+1 {{has the type of a structure that guarantees 'directed, minimum, masked, stable(owner, search)', more than its build does: 'directed, explicit, masked, stable(owner, search)'}}
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%n) cutoff(1.0)
      skin(0.0) kind(matrix) width(64) images(explicit)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A refresh builds with its images, which its type states.
func.func @refresh_images(%x: !vec, %cell: !md.cell) {
  %nl0 = md_exec.empty_neighbors kind(matrix) width(64) : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  // expected-error@+1 {{builds images(explicit), but its structure's type states 'minimum'}}
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell cutoff(1.0) skin(0.0)
      cell_width(1.0) images(explicit) policy(always) : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec
  return
}

// -----

md.particle_set @atoms

// The cast only weakens.
func.func @strengthen(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>) {
  // expected-error@+1 {{cannot strengthen 'directed, minimum, masked, grouped_by(owner)' to 'directed, minimum, masked, stable(owner, search)': the result must state less}}
  %w = md_exec.weaken_neighbors %nl
      : !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  return
}

// -----

md.particle_set @atoms

// Nor from masked exclusions to removed ones.
func.func @strengthen_exclusions(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>) {
  // expected-error@+1 {{cannot strengthen 'directed, minimum, masked, stable(owner, search)' to 'directed, minimum, removed, stable(owner, search)'}}
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, removed, stable(owner, search)>
  return
}

// -----

md.particle_set @atoms

// Another key is not weaker: the two are not comparable (D409).
func.func @other_key(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>) {
  // expected-error@+1 {{cannot strengthen 'directed, minimum, masked, stable(owner, search)' to 'directed, minimum, masked, grouped_by(group)'}}
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(group)>
  return
}

// -----

md.particle_set @atoms

// Nor is another traversal.
func.func @other_traversal(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>) {
  // expected-error@+1 {{cannot strengthen 'directed, minimum, masked, stable(owner, search)' to 'unique, minimum, masked, none'}}
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@atoms, unique, minimum, masked, none>
  return
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @other_set(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>) {
  // expected-error@+1 {{the structure is on @atoms, but the result is on @other}}
  %w = md_exec.weaken_neighbors %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      to !mdrt.neighbors<@other, directed, minimum, masked, none>
  return
}

// -----

md.particle_set @atoms

// The two branches of an scf.if yield one type: a structure of groups and a
// matrix are not one structure.
func.func @branches(%c: i1, %a: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>,
    %b: !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>) {
  // expected-error@+1 {{successor operand type #0 '!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>' should match successor input type #0 '!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>'}}
  %r = scf.if %c -> !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)> {
    // expected-note@+1 {{region branch point}}
    scf.yield %a : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  } else {
    scf.yield %b : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  }
  return
}

// -----

md.particle_set @atoms

// The same for the value that a loop carries.
func.func @carried(%n: index, %a: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>,
    %b: !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{successor operand type #0 '!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>' should match successor input type #0 '!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>'}}
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %a) -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)> {
    // expected-note@+1 {{region branch point}}
    scf.yield %b : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms

// A refresh's result is its operand: the type goes through unchanged.
func.func @refresh(%x: !vec, %cell: !md.cell) {
  // expected-note@+1 {{prior use here}}
  %nl0 = md_exec.empty_neighbors kind(groups) width(64)
      : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
  // expected-error@+1 {{use of value '%nl0' expects different type than prior uses}}
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell cutoff(1.0) skin(0.0)
      cell_width(1.0) policy(always) : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec
  return
}
