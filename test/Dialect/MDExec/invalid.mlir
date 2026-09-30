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
      -> !md.field<@ions, f64>
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
  // expected-error@+1 {{expected cells of a positive width, got 0.000000e+00}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(0.0) policy(check)
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

// -----

md.particle_set @atoms

// The squared distance and the displacement have one type.
func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: !md.field<@atoms, 3 x f64>,
             %cell: !md.cell) -> f64 {
  %u0 = arith.constant 0.0 : f64
  // expected-error@+1 {{expected kernel argument 1 to have type 'vector<3xf32>', got 'vector<3xf64>'}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64)
      cutoff(2.5) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf64>):
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %e : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

md.particle_set @atoms

// A buffer of integers does not hold a field of floating-point values.
func.func @f(%buffer: memref<?xi32>) {
  // expected-error@+1 {{expected the buffer of '!md.field<@atoms, f64>' to have type 'memref<?xf64>', got 'memref<?xi32>'}}
  %q = mdrt.from_buffer %buffer : memref<?xi32> to !md.field<@atoms, f64>
  return
}

// -----

md.particle_set @atoms

// A loop has one form.
func.func @f(%v: !md.field<@atoms, 3 x f64>, %buffer: memref<?x3xf64>) {
  // expected-error@+1 {{expected fields only, as in the value form, or buffers only, as in the storage form}}
  md_exec.particle_for ins(%v : !md.field<@atoms, 3 x f64>)
      outs(%buffer : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  }
  return
}

// -----

md.particle_set @atoms

// In the storage form a destination has no result.
func.func @f(%v: memref<?x3xf64>) {
  // expected-error@+1 {{expected 0 results, one per value in 'reduce', got 1}}
  %w = md_exec.particle_for ins(%v : memref<?x3xf64>)
      outs(%v : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  } -> memref<?x3xf64>
  return
}

// -----

md.particle_set @atoms

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: !md.field<@atoms, 3 x f64>,
             %cell: !md.cell) -> !md.field<@atoms, 3 x f64> {
  %f0 = md_exec.zeros : !md.field<@atoms, 3 x f64>
  // expected-error@+1 {{'overwrite' belongs to the storage form; in the value form the destination tells what the loop adds to}}
  %f = md_exec.pair_for %nl, %x, %cell
      outs(%f0 : !md.field<@atoms, 3 x f64>) cutoff(2.5) overwrite [true]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d : vector<3xf64>
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
      -> !md.field<@atoms, 3 x f64>
  return %f : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

func.func @f(%n: index) {
  // expected-error@+1 {{expected the type of a buffer that holds positions, got 'memref<?xi32>'}}
  %nl = md_exec.empty_neighbors size(%n) positions(memref<?xi32>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  return
}

// -----

md.particle_set @atoms

// A buffer holds one value or three per particle.
func.func @f(%v: memref<?x2xf64>) {
  // expected-error@+1 {{operand #0 must be variadic of A per-particle field or buffer of a field, but got 'memref<?x2xf64>'}}
  md_exec.particle_for ins(%v : memref<?x2xf64>)
      outs(%v : memref<?x2xf64>) {
  ^bb0(%v_i: vector<2xf64>):
    md_exec.yield %v_i : vector<2xf64>
  }
  return
}

// -----

md.particle_set @atoms

func.func private @write_frame(i64, memref<?xf64>)

func.func @f(%x: !md.field<@atoms, 3 x f64>, %step: i64) {
  // expected-error@+1 {{argument 1 of 'write_frame' has type 'memref<?xf64>', which does not take '!md.field<@atoms, 3 x f64>'}}
  mdrt.host_call @write_frame(%step, %x)
      : (i64, !md.field<@atoms, 3 x f64>)
  return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>, %moved: i1) {
  // expected-error@+1 {{'moved' belongs to the policy 'check': with the policy 'always' the structure is built whatever has moved}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(always)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) {
  // expected-error@+1 {{expected a positive 'interval' with the policy 'interval'}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(interval)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) {
  // expected-error@+1 {{'interval' belongs to the policy 'interval'}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) interval(10) policy(check)
      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
  md.return
}

// -----

md.particle_set @atoms

func.func @f(%x: memref<?x3xf64>, %a: memref<?xf64>, %cell: !md.cell,
             %nl: !mdrt.neighbors<@atoms>, %moved: i1) {
  // expected-error@+1 {{expected no buffers in 'scratch': with 'moved' the op does not test the displacements}}
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell
      scratch(%a, %a : memref<?xf64>, memref<?xf64>) moved(%moved)
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  return
}

// -----

md.function @f(%nl: !mdrt.neighbors<@atoms>) {
  // expected-error@+1 {{the structure is on @atoms, but the positions belong to @ions}}
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> !md.field<@ions, 3 x f64>
  md.return
}

// -----

md.function @f(%nl: !mdrt.neighbors<@atoms>) {
  // expected-error@+1 {{expected a position field with 3 components of f32 or f64, got '!md.field<@atoms, f64>'}}
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> !md.field<@atoms, f64>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %nl: !mdrt.neighbors<@atoms>) {
  %no = arith.constant false
  // expected-error@+1 {{expected a value in 'reduce' to be f32, f64, or a fixed-size vector of one of them, got 'i1'}}
  %any = md_exec.pair_for %nl, %x, %cell reduce(%no : i1) cutoff(2.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %yes = arith.constant true
    md_exec.yield %yes : i1
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> i1
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>) {
  %none = arith.constant 0 : i32
  // expected-error@+1 {{expected a value in 'reduce' to be f32, f64, a fixed-size vector of one of them, or i1, got 'i32'}}
  %count = md_exec.particle_for ins(%x : !md.field<@atoms, 3 x f64>)
      reduce(%none : i32) {
  ^bb0(%x_i: vector<3xf64>):
    %one = arith.constant 1 : i32
    md_exec.yield %one : i32
  } -> i32
  md.return
}

// -----

md.particle_set @atoms

func.func @f(%x: memref<?x3xf64>, %order: memref<?xi32>) {
  // expected-error@+1 {{expected the buffer in 'outs' to be another buffer than that of the field}}
  md_exec.permute %x, %order outs(%x : memref<?x3xf64>)
      : memref<?x3xf64>, memref<?xi32>
  return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %ids: !md.field<@atoms, f64>) {
  // expected-error@+1 {{expected 'ids' to hold one i32 per particle, got '!md.field<@atoms, f64>'}}
  %order = md_exec.spatial_order %x, %cell, %ids width(1.4)
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64>
      -> !mdrt.permutation<@atoms>
  md.return
}

// -----

md.function @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
               %ids: !md.field<@atoms, i32>) {
  // expected-error@+1 {{expected a result in the value form, and a buffer in 'outs' in the storage form}}
  md_exec.spatial_order %x, %cell, %ids width(1.4)
      : !md.field<@atoms, 3 x f64>, !md.field<@atoms, i32>
  md.return
}

// -----

md.particle_set @atoms

func.func @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
             %nl: !mdrt.neighbors<@atoms>, %u0: f64) -> f64 {
  // expected-error@+1 {{expected 1 exchange contracts, one per value in 'outs' and 'reduce', got 2}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.5)
      exchange [symmetric, symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}

// -----

md.particle_set @atoms

func.func @f(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
             %nl: !mdrt.neighbors<@atoms>, %u0: f64) -> f64 {
  // expected-error@+2 {{expected 'none', 'symmetric', or 'antisymmetric', got 'odd'}}
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.5)
      exchange [odd] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> f64
  return %u : f64
}
