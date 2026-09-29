// RUN: mdir-opt %s -split-input-file -verify-diagnostics

func.func @f(%m: memref<?x3xi32>) {
  // expected-error@+1 {{expected the number of particles in 'size'}}
  %inc = md_exec.build_incidence %m : memref<?x3xi32> -> memref<?x?xi32>
  return
}

// -----

func.func @f(%m: memref<?x3xi32>, %n: index) {
  // expected-error@+1 {{expected a relation and a structure, as in the value form, or buffers only}}
  %inc = md_exec.build_incidence %m size(%n)
      : memref<?x3xi32> -> !mdrt.incidence<@atoms, @angles, 3>
  return
}

// -----

func.func @f(%a: !md.relation<@atoms, 3, reversal, @angles>) {
  // expected-error@+1 {{expected the result to have type '!mdrt.incidence<@atoms, @angles, 3>', got '!mdrt.incidence<@atoms, @bonds, 3>'}}
  %inc = md_exec.build_incidence %a
      : !md.relation<@atoms, 3, reversal, @angles>
        -> !mdrt.incidence<@atoms, @bonds, 3>
  return
}

// -----

func.func @f(%m: memref<?x2xi32>) {
  // expected-error@+1 {{expected the relation of a tuple set}}
  %a = mdrt.from_buffer %m : memref<?x2xi32> to !md.relation<@atoms, 2, unordered>
  return
}

// -----

func.func @f(%m: memref<?x2xi32>) {
  // expected-error@+1 {{expected the buffer of '!md.relation<@atoms, 3, reversal, @angles>' to have type 'memref<?x3xi32>', got 'memref<?x2xi32>'}}
  %a = mdrt.from_buffer %m : memref<?x2xi32> to !md.relation<@atoms, 3, reversal, @angles>
  return
}

// -----

func.func @f(%inc: !mdrt.incidence<@atoms, @angles, 3>,
             %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the tuples have 3 members, but the arity is 2}}
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         reduce(%u0 : f64) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %c = arith.constant 0.0 : f64
    md_exec.yield %c : f64
  } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

func.func @f(%inc: !mdrt.incidence<@atoms, @angles, 3>,
             %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected displacements only, got 'distance'}}
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(distance(0, 1))
         reduce(%u0 : f64) arity(3) {
  ^bb0(%r: f64):
    md_exec.yield %r : f64
  } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

func.func @f(%inc: !mdrt.incidence<@atoms, @angles, 3>,
             %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
             %k: !md.field<@bonds, f64>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the tuples are those of @angles, but a field in 'tuple' belongs to @bonds}}
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         tuple(%k : !md.field<@bonds, f64>) reduce(%u0 : f64) arity(3) {
  ^bb0(%d: vector<3xf64>, %k_t: f64):
    md_exec.yield %k_t : f64
  } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

func.func @f(%inc: !mdrt.incidence<@atoms, @angles, 3>,
             %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell)
    -> !md.field<@atoms, 3 x f64> {
  %f0 = md_exec.zeros : !md.field<@atoms, 3 x f64>
  %f = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         outs(%f0 : !md.field<@atoms, 3 x f64>) arity(3) {
  ^bb0(%d: vector<3xf64>):
    // expected-error@+1 {{expected 3 values (one per member for each field in 'outs', and one per value in 'reduce'), got 1}}
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  return %f : !md.field<@atoms, 3 x f64>
}

// -----

func.func @f(%inc: !mdrt.incidence<@atoms, @angles, 3>,
             %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
             %q: !md.field<@atoms, f64>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected the kernel to have 4 arguments (one per displacement, one per member for each field in 'ins', and one per field in 'tuple'), got 2}}
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         ins(%q : !md.field<@atoms, f64>) reduce(%u0 : f64) arity(3) {
  ^bb0(%d: vector<3xf64>, %q0: f64):
    md_exec.yield %q0 : f64
  } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

func.func @f(%e: !mdrt.incidence<@atoms, @angles, 3>) {
  // expected-error@+1 {{expected the excluded pairs to have 2 members, got 3}}
  %nl = md_exec.empty_neighbors kind(matrix) width(32)
      exclude(%e : !mdrt.incidence<@atoms, @angles, 3>)
      : !mdrt.neighbors<@atoms>
  return
}
