// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN:     -split-input-file -verify-diagnostics

// The lowering checks the precondition of host(lanes(W)) again
// (D402) and fails on an attribute that it refuses.

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @two_ended(%x: !vec, %cell: !md.cell) -> !vec {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %f0 = md_exec.zeros : !vec
  // expected-error @+1 {{cannot take host(lanes(8)): the policy is not (directed, owner_only): the lanes write at the centers alone}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, atomic) host(lanes(8)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @width(%x: !vec, %cell: !md.cell) -> !vec {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
  %f0 = md_exec.zeros : !vec
  // expected-error @+1 {{cannot take host(lanes(3)): the lanes are a power of two from 1 to 64, not 3}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, owner_only) host(lanes(3)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, !vec -> !vec
  return %f : !vec
}
