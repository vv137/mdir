// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected an arity of at least 1, got 0}}
md.tuple_set @none on(@atoms) arity(0) orientation(ordered)

// -----

// expected-error@+1 {{expected the orientation 'unordered' with the arity 2 only, got the arity 3}}
md.tuple_set @angles on(@atoms) arity(3) orientation(unordered)

// -----

// expected-error@+1 {{expected the orientation 'reversal' with an arity of at least 2}}
md.tuple_set @restraints on(@atoms) arity(1) orientation(reversal)

// -----

// expected-error@+1 {{expected the orientation 'unordered' with the arity 2 only, got the arity 3}}
md.function @f(%a: !md.relation<@atoms, 3, unordered, @angles>) {
  md.return
}

// -----

// expected-error@+1 {{expected an arity of at least 2, got 1}}
md.function @f(%a: !md.relation<@atoms, 1, ordered>) {
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.5)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the relation of a tuple set, got '!md.relation<@atoms, 2, unordered>'}}
  %u = md.sum_tuples %n, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@ions, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>) -> f64 {
  // expected-error@+1 {{the relation is on @atoms, but the positions belong to @ions}}
  %u = md.sum_tuples %b, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@ions, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>,
               %k: !md.field<@angles, f64>) -> f64 {
  // expected-error@+1 {{the tuples are those of @bonds, but a field in 'tuple' belongs to @angles}}
  %u = md.sum_tuples %b, %x, %cell coordinates(distance(0, 1))
         tuple(%k : !md.field<@angles, f64>) {
  ^bb0(%r: f64, %k_t: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>,
               %q: !md.field<@ions, f64>) -> f64 {
  // expected-error@+1 {{the relation is on @atoms, but a gathered field belongs to @ions}}
  %u = md.sum_tuples %b, %x, %cell coordinates(distance(0, 1))
         gather(%q : !md.field<@ions, f64>) {
  ^bb0(%r: f64, %q0: f64, %q1: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>) -> f64 {
  // expected-error@+1 {{'distance' names member 2, but a tuple has the members 0 to 1}}
  %u = md.sum_tuples %b, %x, %cell coordinates(distance(0, 2)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %a: !md.relation<@atoms, 3, reversal, @angles>) -> f64 {
  // expected-error@+1 {{'cosine' names member 1 twice}}
  %u = md.sum_tuples %a, %x, %cell coordinates(cosine(1, 0, 1)) {
  ^bb0(%c: f64):
    md.yield %c : f64
  } : !md.relation<@atoms, 3, reversal, @angles>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %a: !md.relation<@atoms, 3, reversal, @angles>) -> f64 {
  // expected-error@+1 {{expected 3 members for 'angle', got 2}}
  %u = md.sum_tuples %a, %x, %cell coordinates(angle(0, 1)) {
  ^bb0(%c: f64):
    md.yield %c : f64
  } : !md.relation<@atoms, 3, reversal, @angles>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %a: !md.relation<@atoms, 3, reversal, @angles>) -> f64 {
  // expected-error@+1 {{expected 'distance', 'displacement', 'angle', 'cosine', or 'dihedral', got 'torsion'}}
  %u = md.sum_tuples %a, %x, %cell coordinates(torsion(0, 1, 2)) {
  ^bb0(%c: f64):
    md.yield %c : f64
  } : !md.relation<@atoms, 3, reversal, @angles>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>) -> f64 {
  // expected-error@+1 {{expected the kernel to have 2 arguments (one per coordinate, one per member for each gathered field, and one per field of the tuples), got 1}}
  %u = md.sum_tuples %b, %x, %cell
         coordinates(distance(0, 1), displacement(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>) -> f64 {
  // expected-error@+1 {{expected kernel argument 0 to have type 'vector<3xf64>', got 'f64'}}
  %u = md.sum_tuples %b, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>)
    -> !md.field<@atoms, 3 x f64> {
  %f = md.gather_tuples %b, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    // expected-error@+1 {{expected 2 values, got 1}}
    md.yield %d : vector<3xf64>
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>)
    -> !md.field<@atoms, 3 x f64> {
  %f = md.gather_tuples %b, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    // expected-error@+1 {{expected values of type 'vector<3xf64>', got 'f64'}}
    md.yield %r, %r : f64, f64
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %b: !md.relation<@atoms, 2, unordered, @bonds>)
    -> !md.field<@ions, 3 x f64> {
  // expected-error@+1 {{the relation is on @atoms, but the result belongs to @ions}}
  %f = md.gather_tuples %b, %x, %cell coordinates(displacement(0, 1)) {
  ^bb0(%d: vector<3xf64>):
    md.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !md.relation<@atoms, 2, unordered, @bonds>, !md.field<@atoms, 3 x f64>
      -> !md.field<@ions, 3 x f64>
  md.return %f : !md.field<@ions, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %e: !md.relation<@atoms, 3, reversal, @angles>) {
  // expected-error@+1 {{expected the excluded pairs to be the unordered relation of arity 2 of a tuple set, got '!md.relation<@atoms, 3, reversal, @angles>'}}
  %n = md.neighborhood %x, %cell cutoff(1.0)
         exclude(%e : !md.relation<@atoms, 3, reversal, @angles>)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %e: !md.relation<@ions, 2, unordered, @excluded>) {
  // expected-error@+1 {{the excluded pairs are on @ions, but the positions belong to @atoms}}
  %n = md.neighborhood %x, %cell cutoff(1.0)
         exclude(%e : !md.relation<@ions, 2, unordered, @excluded>)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  md.return
}

// -----

// expected-error@+1 {{only a table of rank 2 can be symmetric}}
md.function @f(%t: !md.table<1, f64, symmetric>) {
  md.return
}

// -----

// expected-error@+1 {{expected a table of rank 1 or 2, got 3}}
md.function @f(%t: !md.table<3, f64>) {
  md.return
}

// -----

md.function @f(%t: !md.table<2, f64, symmetric>, %i: i32) -> f64 {
  // expected-error@+1 {{expected 2 indices, one for each dimension of the table, got 1}}
  %v = md.lookup %t[%i] : !md.table<2, f64, symmetric>, i32 -> f64
  md.return %v : f64
}

// -----

func.func @f(%b: memref<?xf64>) {
  // expected-error@+1 {{expected the buffer of '!md.table<2, f64, symmetric>' to have type 'memref<?x?xf64>', got 'memref<?xf64>'}}
  %t = mdrt.from_buffer %b : memref<?xf64> to !md.table<2, f64, symmetric>
  return
}
