// RUN: mdir-opt %s --convert-md-exec-to-loops -split-input-file \
// RUN:     -verify-diagnostics

md.particle_set @atoms

// The lowering takes the storage form only.
// expected-error@+1 {{in its signature, which is not in the storage form; run 'md-exec-assign-storage' first}}
func.func @f(%v: !md.field<@atoms, 3 x f64>) {
  return
}

// -----

md.particle_set @atoms

func.func @f(%buffer: memref<?x3xf64>) {
  // expected-error@+1 {{is not in the storage form; run 'md-exec-assign-storage' first}}
  %x = mdrt.from_buffer %buffer
      : memref<?x3xf64> to !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms

func.func @f() {
  // expected-error@+1 {{is not in the storage form; run 'md-exec-assign-storage' first}}
  %nl = md_exec.empty_neighbors kind(matrix) width(48)
      : !mdrt.neighbors<@atoms>
  return
}

// -----

md.particle_set @atoms

// The storage was allocated for positions of another type.
func.func @f(%x: memref<?x3xf32>, %cell: !md.cell, %n: index) {
  %nl0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  // expected-error@+1 {{the storage of the neighbor structure is for positions that are stored in 'memref<?x3xf64>', but these are stored in 'memref<?x3xf32>'}}
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf32>
  return
}

// -----

md.particle_set @atoms

func.func @f(%v: memref<?x3xf64, 1>) {
  // expected-error@+1 {{has its buffers on a device; use 'convert-md-exec-to-gpu'}}
  md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
      outs(%v : memref<?x3xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  }
  return
}
