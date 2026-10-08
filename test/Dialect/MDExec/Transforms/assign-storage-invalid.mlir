// RUN: mdir-opt %s --md-exec-assign-storage -split-input-file \
// RUN:     -verify-diagnostics

md.particle_set @atoms

// The loop would overwrite positions that are read after it.
func.func @f(%x: !md.field<@atoms, 3 x f64>, %n: index)
    -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{needs a buffer of its own: the loop updates a field that is used after the loop or that belongs to an enclosing region}}
  %xe = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x) -> (!md.field<@atoms, 3 x f64>) {
    %x0 = md_exec.empty : !md.field<@atoms, 3 x f64>
    %xb = md_exec.particle_for ins(%xa : !md.field<@atoms, 3 x f64>)
        outs(%x0 : !md.field<@atoms, 3 x f64>) {
    ^bb0(%x_i: vector<3xf64>):
      %s = arith.addf %x_i, %x_i : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } -> !md.field<@atoms, 3 x f64>
    scf.yield %xb : !md.field<@atoms, 3 x f64>
  }
  return %x, %xe : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

// The loop would yield the same buffer for two fields.
func.func @f(%x: !md.field<@atoms, 3 x f64>, %y: !md.field<@atoms, 3 x f64>,
             %n: index) -> !md.field<@atoms, 3 x f64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %xe, %ye = scf.for %step = %c0 to %n step %c1 iter_args(%xa = %x, %ya = %y)
      -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) {
    // expected-error@+1 {{needs a buffer of its own: the loop yields one field twice}}
    scf.yield %xa, %xa
        : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>
  }
  return %xe : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}

func.func @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  // expected-error@+1 {{cannot be given storage; run 'md-differentiate', 'md-inline', and 'convert-md-to-md-exec' first}}
  %u = md.evaluate @u(%x, %cell) request [energy]
      : (!md.field<@atoms, 3 x f64>, !md.cell) -> f64
  return %u : f64
}

// -----

md.particle_set @atoms

// No field of the particle set has a buffer, so the number of particles is
// not known.
func.func @f() -> !md.field<@atoms, 3 x f64> {
  // expected-error@+1 {{the number of particles of @atoms is not known here: no field of the set has a buffer yet}}
  %x = md_exec.zeros : !md.field<@atoms, 3 x f64>
  return %x : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

// The buffer holds f32 and the field has f64: converting would be a copy.
func.func @f(%buffer: memref<?x3xf32>) {
  // expected-error@+1 {{the field has the type '!md.field<@atoms, 3 x f64>', which is not the type that the buffer stores; run 'md-exec-assign-precision' first}}
  %x = mdrt.from_buffer %buffer
      : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms

// The structure is refreshed where it is, so what it was before is gone.
func.func @f(%x: !md.field<@atoms, 3 x f64>, %y: !md.field<@atoms, 3 x f64>,
             %cell: !md.cell) -> (i64, i64) {
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nl1 = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  // expected-error@+1 {{needs storage of its own: the neighbor structure is used after it is refreshed}}
  %nl2 = md_exec.refresh_neighbors %nl1, %y, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  %a = md_exec.rebuild_count %nl1 : !mdrt.neighbors<@atoms>
  %b = md_exec.rebuild_count %nl2 : !mdrt.neighbors<@atoms>
  return %a, %b : i64, i64
}

// -----

md.particle_set @atoms

// A loop that the pass refuses inside a loop that carries fields, whose body
// uses a neighbor structure refreshed before it: the pass fails with its
// diagnostic alone. The body that it had built for the outer loop, which
// uses the storage of the structure, was left behind, and the process
// aborted at the end ("operation destroyed but still has uses", #208).
func.func @f(%x: !md.field<@atoms, 3 x f64>, %y: !md.field<@atoms, 3 x f64>,
             %cell: !md.cell, %n: index) -> !md.field<@atoms, 3 x f64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nl1 = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  %ye = scf.for %step = %c0 to %n step %c1 iter_args(%ya = %y)
      -> (!md.field<@atoms, 3 x f64>) {
    %a = md_exec.rebuild_count %nl1 : !mdrt.neighbors<@atoms>
    %g0 = md_exec.zeros : !md.field<@atoms, 3 x f64>
    // expected-error@+1 {{needs a buffer of its own: the loop writes to a field that it reads from other particles}}
    %g = md_exec.pair_for %nl1, %x, %cell
        ins(%g0 : !md.field<@atoms, 3 x f64>)
        outs(%g0 : !md.field<@atoms, 3 x f64>) cutoff(1.5)
        policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>, %g1: vector<3xf64>, %g2: vector<3xf64>):
      %s = arith.addf %g1, %g2 : vector<3xf64>
      md_exec.yield %s : vector<3xf64>
    } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
        -> !md.field<@atoms, 3 x f64>
    scf.yield %g : !md.field<@atoms, 3 x f64>
  }
  return %ye : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

// The same for a refusal of the end of the body.
func.func @f(%x: !md.field<@atoms, 3 x f64>, %y: !md.field<@atoms, 3 x f64>,
             %z: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %n: index)
    -> !md.field<@atoms, 3 x f64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  %nl1 = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  %ye, %ze = scf.for %step = %c0 to %n step %c1 iter_args(%ya = %y, %za = %z)
      -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>) {
    %a = md_exec.rebuild_count %nl1 : !mdrt.neighbors<@atoms>
    // expected-error@+1 {{needs a buffer of its own: the loop yields one field twice}}
    scf.yield %ya, %ya
        : !md.field<@atoms, 3 x f64>, !md.field<@atoms, 3 x f64>
  }
  return %ye : !md.field<@atoms, 3 x f64>
}
