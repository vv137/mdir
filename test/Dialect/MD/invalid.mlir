// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected 1 or 3 components, got 2}}
md.function @f(%a: !md.field<@atoms, 2 x f64>) {
  md.return
}

// -----

// expected-error@+1 {{expected element type f64, i32, or i64, got 'f32'}}
md.function @f(%a: !md.field<@atoms, f32>) {
  md.return
}

// -----

// expected-error@+1 {{a field with 3 components must have element type f64, got 'i32'}}
md.function @f(%a: !md.field<@atoms, 3 x i32>) {
  md.return
}

// -----

// expected-error@+1 {{expected an arity of at least 2, got 1}}
md.function @f(%a: !md.relation<@atoms, 1, unordered>) {
  md.return
}

// -----

// expected-error@+1 {{expected argument 0 to be a position field}}
md.potential @u(%q: !md.field<@atoms, f64>, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}

// -----

// expected-error@+1 {{expected argument 1 to be a cell}}
md.potential @u(%x: !md.field<@atoms, 3 x f64>, %eps: f64) -> f64 {
  md.return %eps : f64
}

// -----

// expected-error@+1 {{expected exactly one result of type f64}}
md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  md.return %x : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>) -> f64 {
  // expected-error@+1 {{type of return value 0 ('!md.field<@atoms, 3 x f64>') does not match the result type ('f64')}}
  md.return %x : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected a positive cutoff, got 0.000000e+00}}
  %n = md.neighborhood %x, %cell cutoff(0.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{result is a relation on @ions, but the positions belong to @atoms}}
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@ions, 2, unordered>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected the result to be an unordered relation of arity 2}}
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, ordered>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{a sum over an unordered relation requires 'exchange(symmetric)'}}
  %u = md.sum_relation %n, %x, %cell exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %y: !md.field<@atoms, 3 x f64>,
               %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the positions that the neighborhood was built from}}
  %u = md.sum_relation %n, %y, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{a switching truncation requires 'from'}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(switch) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{'from' is allowed only with 'truncation(switch)' and 'truncation(force_switch)'}}
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(shift, from = 0.5) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the switching distance (1.500000e+00) to be smaller than the cutoff (1.000000e+00)}}
  %u = md.sum_relation %n, %x, %cell
         exchange(symmetric) truncation(switch, from = 1.5) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected the kernel to have 2 arguments (distance, displacement, and two per gathered field), got 1}}
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %q: !md.field<@atoms, f64>) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected kernel argument 3 to have type 'f64', got 'i32'}}
  %u = md.sum_relation %n, %x, %cell gather(%q : !md.field<@atoms, f64>)
         exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %q1: f64, %q2: i32):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  %f = md.gather_relation %n, %x, %cell exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    // expected-error@+1 {{expected a value of type 'vector<3xf64>', got 'f64'}}
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(1.0)
         : !md.field<@atoms, 3 x f64> -> !md.relation<@atoms, 2, unordered>
  // expected-error@+1 {{expected 'none', 'symmetric', or 'antisymmetric', got 'odd'}}
  %u = md.sum_relation %n, %x, %cell exchange(odd) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%a: !md.field<@atoms, f64>, %b: !md.field<@ions, f64>) -> f64 {
  // expected-error@+1 {{expected all gathered fields to belong to @atoms, but one belongs to @ions}}
  %s = md.sum_particles gather(%a, %b : !md.field<@atoms, f64>,
                                        !md.field<@ions, f64>) {
  ^bb0(%a_i: f64, %b_i: f64):
    md.yield %a_i : f64
  } : f64
  md.return %s : f64
}

// -----

md.function @f(%a: !md.field<@atoms, f64>) -> !md.field<@atoms, 3 x f64> {
  %s = md.map_particles gather(%a : !md.field<@atoms, f64>) {
  ^bb0(%a_i: f64):
    // expected-error@+1 {{expected a value of type 'vector<3xf64>', got 'f64'}}
    md.yield %a_i : f64
  } : !md.field<@atoms, 3 x f64>
  md.return %s : !md.field<@atoms, 3 x f64>
}

// -----

md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> f64 {
  md.return %a : f64
}

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> f64 {
  // expected-error@+1 {{expected 'derivative' to name a parameter, an argument index from 2 to 2, got 0}}
  %d = md.evaluate @u(%x, %cell, %a) request [derivative(0)]
      : (!md.field<@atoms, 3 x f64>, !md.cell, f64) -> f64
  md.return %d : f64
}

// -----

md.potential @u(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> f64 {
  md.return %a : f64
}

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> f64 {
  // expected-error@+1 {{expected result 0 to have type '!md.field<@atoms, 3 x f64>', got 'f64'}}
  %d = md.evaluate @u(%x, %cell, %a) request [forces]
      : (!md.field<@atoms, 3 x f64>, !md.cell, f64) -> f64
  md.return %d : f64
}

// -----

md.function @g(%a: f64) -> f64 {
  md.return %a : f64
}

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %a: f64)
    -> f64 {
  // expected-error@+1 {{'g' does not name a potential}}
  %d = md.evaluate @g(%x, %cell, %a) request [energy]
      : (!md.field<@atoms, 3 x f64>, !md.cell, f64) -> f64
  md.return %d : f64
}

// -----

md.function @g(%a: f64) -> f64 {
  md.return %a : f64
}

md.function @f(%a: f64) -> f64 {
  // expected-error@+1 {{'h' does not name a function or a potential}}
  %d = md.call @h(%a) : (f64) -> f64
  md.return %d : f64
}

// -----

md.function @g(%a: f64) -> f64 {
  md.return %a : f64
}

md.function @f(%x: !md.field<@atoms, 3 x f64>) -> f64 {
  // expected-error@+1 {{expected operand 0 to have type 'f64', got '!md.field<@atoms, 3 x f64>'}}
  %d = md.call @g(%x) : (!md.field<@atoms, 3 x f64>) -> f64
  md.return %d : f64
}
