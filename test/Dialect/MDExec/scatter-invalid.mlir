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

// -----

// A loop over pairs takes the table where it adds at both ends of a pair
// (D291).
!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms
func.func @f(%x: !vec, %cell: !md.cell, %nl: !mdrt.neighbors<@atoms>,
             %f0: !vec) -> !vec {
  // expected-error@+1 {{cannot take the strategy 'hash' with the conflict 'owner_only': the loop adds only at its centers, each by its own thread}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) scatter(hash(512, 8, 0)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !mdrt.neighbors<@atoms>, !vec -> !vec
  return %f : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms
func.func @f(%x: !vec, %cell: !md.cell, %nl: !mdrt.neighbors<@atoms>,
             %f0: !vec) -> !vec {
  // expected-error@+1 {{cannot take the strategy 'tile': the entries of a list may be anywhere, so the footprint of a block is not bounded}}
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(unique, atomic) scatter(tile(8)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !mdrt.neighbors<@atoms>, !vec -> !vec
  return %f : !vec
}

// -----

// md_exec.scatter_add (D291).
md.particle_set @atoms
md.particle_set @ghosts
func.func @f(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, i32>,
             %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{cannot take the strategy 'tile': an index may name any row of the destination, so the footprint of a block is not bounded}}
  %a1 = md_exec.scatter_add %v at %i into %a scatter(tile(8))
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @f(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@atoms, i32>,
             %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{expected the values and the indices on one set, got @ghosts and @atoms}}
  %a1 = md_exec.scatter_add %v at %i into %a
      : !md.field<@ghosts, 3 x f64>, !md.field<@atoms, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @f(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, f64>,
             %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{expected one index of i32 a row, got '!md.field<@ghosts, f64>'}}
  %a1 = md_exec.scatter_add %v at %i into %a
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, f64>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @f(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, i32>,
             %a: !md.field<@atoms, 3 x f32>) {
  // expected-error@+1 {{expected the destination to have the components and the type of the values}}
  %a1 = md_exec.scatter_add %v at %i into %a
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f32>
  return
}

// -----

func.func @f(%v: memref<?x3xf64>, %i: memref<?xi32>, %a: memref<?x3xf64>) {
  // expected-error@+1 {{the strategy 'sorted' is reserved for #417 and not implemented}}
  md_exec.scatter_add %v at %i into %a scatter(sorted)
      : memref<?x3xf64>, memref<?xi32>, memref<?x3xf64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @f(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, i32>,
             %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{takes 'overwrite' in the storage form only}}
  %a1 = md_exec.scatter_add %v at %i into %a overwrite(true)
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

func.func @f(%v: memref<?x3xf64>, %i: memref<?xi32>,
             %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{expected the values, the indices, and the destination all fields or all buffers}}
  %a1 = md_exec.scatter_add %v at %i into %a
      : memref<?x3xf64>, memref<?xi32> -> !md.field<@atoms, 3 x f64>
  return
}
