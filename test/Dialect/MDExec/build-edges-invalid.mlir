// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// The forms of md_exec.build_edges (D277) and the host
// call that takes its pairs.

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: !vec, %cell: !md.cell) {
  // expected-error@+1 {{expected the result to have type '!md.edges<@atoms>'}}
  %e = md_exec.build_edges %nl, %x, %cell cutoff(0.7)
         : !mdrt.neighbors<@atoms>, !vec -> !md.edges<@water>
  return
}

// -----

md.particle_set @atoms

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: memref<?x3xf64, 1>,
             %cell: !md.cell) {
  // expected-error@+1 {{expected the slots, the counts, and the offsets on the device}}
  %s, %c, %o, %b = md_exec.build_edges %nl, %x, %cell cutoff(0.7)
         : !mdrt.neighbors<@atoms>, memref<?x3xf64, 1>
         -> memref<?x5xi32>, memref<?xi32, 1>, memref<?xi32, 1>, i64
  return
}

// -----

md.particle_set @atoms

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: memref<?x3xf32, 1>,
             %cell: !md.cell) {
  // expected-error@+1 {{expected the positions in f64: the cutoff is tested in f64}}
  %s, %c, %o, %b = md_exec.build_edges %nl, %x, %cell cutoff(0.7)
         : !mdrt.neighbors<@atoms>, memref<?x3xf32, 1>
         -> memref<?x5xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, i64
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func private @model(memref<?x3xf64>, memref<?x5xi32>, memref<?xi32>,
                         memref<?xi32>, i64)

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: !vec, %cell: !md.cell) {
  %e = md_exec.build_edges %nl, %x, %cell cutoff(0.7)
         : !mdrt.neighbors<@atoms>, !vec -> !md.edges<@atoms>
  // expected-error@+1 {{takes the pairs of images, which a host call takes only in place}}
  mdrt.host_call @model(%x, %e) : (!vec, !md.edges<@atoms>)
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>

md.particle_set @atoms

func.func private @model(memref<?x3xf64>, memref<?x5xi32>, memref<?xi32>,
                         memref<?xi32>)

func.func @f(%nl: !mdrt.neighbors<@atoms>, %x: !vec, %cell: !md.cell) {
  %e = md_exec.build_edges %nl, %x, %cell cutoff(0.7)
         : !mdrt.neighbors<@atoms>, !vec -> !md.edges<@atoms>
  // expected-error@+1 {{'model' takes 4 arguments, got 5}}
  mdrt.host_call @model(%x, %e) {in_place} : (!vec, !md.edges<@atoms>)
  return
}
