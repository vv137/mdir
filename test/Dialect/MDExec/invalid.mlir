// RUN: mdir-opt %s -split-input-file -verify-diagnostics

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{the structure is on @ions, but the positions belong to @atoms}}
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@ions>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected a positive width, got 0.000000e+00}}
  %cells = md_exec.build_cells %x, %cell width(0.0)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(2.5)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  // expected-error@+1 {{the cells are 2.500000e+00 wide, which is less than the cutoff plus the skin, 2.800000e+00}}
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(96)
      : !mdrt.cells<@atoms>, !md.field<@atoms, 3 x f64>
      -> !mdrt.neighbors<@atoms>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(3.0)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  // expected-error@+1 {{expected a positive width, got 0}}
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(0)
      : !mdrt.cells<@atoms>, !md.field<@atoms, 3 x f64>
      -> !mdrt.neighbors<@atoms>
  md.return
}

// -----

md.function @f(%q: !md.field<@ions, f64>,
               %order: !mdrt.permutation<@atoms>) {
  // expected-error@+1 {{the order is on @atoms, but the field belongs to @ions}}
  %p = md_exec.permute %q, %order
      : !md.field<@ions, f64>, !mdrt.permutation<@atoms>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !md.field<@atoms, 3 x f64> -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(8)
      : !mdrt.cells<@atoms>, !md.field<@atoms, 3 x f64>
      -> !mdrt.neighbors<@atoms>
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{the cutoff, 1.500000e+00, exceeds the cutoff that the neighbor structure was built with, 1.000000e+00}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{only the policy (directed, owner_only) is supported}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      policy(unique, atomic) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected 1 weights, one per value in 'reduce', got 2}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      weights [0.5, 0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected the kernel to have 2 arguments, got 1}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  md.return %u : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) -> !md.field<@atoms, 3 x f64> {
  %f0 = md_exec.zeros : !md.field<@atoms, 3 x f64>
  %f = md_exec.pair_for %nl, %x, %cell
      outs(%f0 : !md.field<@atoms, 3 x f64>) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    // expected-error@+1 {{expected value 0 to have type 'vector<3xf64>', got 'f64'}}
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  md.return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.function @f(%v: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{expected at least 1 field in 'outs' or value in 'reduce'}}
  md_exec.particle_for ins(%v : !md.field<@atoms, 3 x f64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield
  }
  md.return
}

// -----

md.function @f(%v: !md.field<@atoms, 3 x f64>, %q: !md.field<@ions, f64>)
    -> f64 {
  %s0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected all fields to belong to @atoms, but one belongs to @ions}}
  %s = md_exec.particle_for
      ins(%v, %q : !md.field<@atoms, 3 x f64>, !md.field<@ions, f64>)
      reduce(%s0 : f64) {
  ^bb0(%v_i: vector<3xf64>, %q_i: f64):
    md_exec.yield %q_i : f64
  } -> f64
  md.return %s : f64
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) {
  // expected-error@+1 {{the cells are 2.500000e+00 wide, which is less than the cutoff plus the skin, 2.800000e+00}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.5) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  md.return
}

// -----

md.function @f() {
  // expected-error@+1 {{expected a positive width, got 0}}
  %nl = md_exec.empty_neighbors kind(matrix) width(0)
      : !mdrt.neighbors<@atoms>
  md.return
}
