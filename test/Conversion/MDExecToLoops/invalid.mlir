// RUN: mdir-opt %s --convert-md-exec-to-loops -split-input-file -verify-diagnostics

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
  // expected-error@+1 {{cannot be lowered; run 'md-differentiate', 'md-inline', and 'convert-md-to-md-exec' first}}
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
