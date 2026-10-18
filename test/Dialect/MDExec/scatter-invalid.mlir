// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// The strategy of a scatter-accumulate (D285): each op takes
// the strategies that its lowering has, and the parameters are checked.

func.func @f(%inc: memref<?x?xi32>, %x: memref<?x3xf32>, %cell: !md.cell,
             %f: memref<?x3xf32>) {
  // expected-error@+1 {{cannot take the strategy 'tile': a tuple may have members anywhere, so the footprint of a block is not bounded}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) scatter(tile(8)) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// -----

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
md.particle_set @atoms
func.func @f(%x: !vec, %cell: !md.cell, %q: !real, %moduli: !md.table<2, f32>) {
  // expected-error@+1 {{cannot take the strategy 'hash': the template of the spreading has the kernels of tiles alone}}
  %u, %w, %f = md_exec.reciprocal %x, %q, %cell, %moduli
      grid([64, 64, 64]) order(4) beta(3.0) coulomb(138.9)
      scatter(hash(256, 8, 0))
      : !vec, !real, !md.table<2, f32> -> f64, vector<9xf64>, !vec
  return
}

// -----

func.func @f(%inc: memref<?x?xi32>, %x: memref<?x3xf32>, %cell: !md.cell,
             %f: memref<?x3xf32>) {
  // expected-error@+2 {{the strategy 'sorted' is reserved for #417 and not implemented}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) scatter(sorted) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// -----

func.func @f(%inc: memref<?x?xi32>, %x: memref<?x3xf32>, %cell: !md.cell,
             %f: memref<?x3xf32>) {
  // expected-error@+2 {{expected a table of a power of two slots, got 384}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) scatter(hash(384, 8, 0)) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// -----

func.func @f(%inc: memref<?x?xi32>, %x: memref<?x3xf32>, %cell: !md.cell,
             %f: memref<?x3xf32>) {
  // expected-error@+2 {{expected at least 1 slot tried, got 0}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) scatter(hash(256, 0, 0)) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// -----

func.func @f(%inc: memref<?x?xi32>, %x: memref<?x3xf32>, %cell: !md.cell,
             %f: memref<?x3xf32>) {
  // expected-error@+2 {{expected 'atomic', 'tile', 'hash', or 'sorted', got 'bricks'}}
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32>) arity(2) scatter(bricks) {
  ^bb0(%d: vector<3xf32>):
    md_exec.yield %d, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32>, memref<?x3xf32>
  return
}

// -----

// The attribute stands on its own as well.
// expected-error@+1 {{expected a tile of at least 1 point an edge, got 0}}
func.func private @g() attributes {scatter = #md_exec.scatter<tile(0)>}
