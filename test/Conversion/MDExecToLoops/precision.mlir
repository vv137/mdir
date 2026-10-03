// RUN: mdir-opt %s --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | FileCheck %s

!positions = !md.field<@atoms, 3 x f64>
!forces    = !md.field<@atoms, 3 x f32>
!single    = !md.field<@atoms, 3 x f32>

md.particle_set @atoms

// Positions in f64 and a kernel in f32, as in the mixed mode. The
// displacement is computed in the type of the positions and then converted:
// the subtraction is the step that loses precision.
//
// CHECK-LABEL: func.func @mixed(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>, %[[BOX:[a-z0-9]+]]: vector<3xf64>)
// CHECK-SAME:    -> (memref<?x3xf32>, f64)
func.func @mixed(%x: !positions, %cell: !md.cell) -> (!forces, f64) {
  // The structure remembers the positions in their type.
  //
  // CHECK:      %[[REFERENCE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf64>
  // CHECK:      call @mdrt.build_neighbors_matrix(%[[X]], %[[BOX]],
  %cells = md_exec.build_cells %x, %cell width(2.8)
      : !positions -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.5) skin(0.3) kind(matrix) width(96)
      : !mdrt.cells<@atoms>, !positions -> !mdrt.neighbors<@atoms>

  // The difference of the positions is taken in f64, then the minimum
  // image and the squared length in f32, the type of the kernel.
  //
  // CHECK:      %[[F:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf32>
  // CHECK:      scf.parallel
  // The square of the cutoff in f32 is pulled in so that its square root
  // is four units in the last place below the cutoff (D159).
  // CHECK:        %[[CUTOFF2:[a-z0-9_]+]] = arith.constant 6.24999619 : f32
  // CHECK:        scf.for
  // CHECK:          %[[RAW:[0-9]+]] = arith.subf %{{[0-9]+}}, %{{[0-9]+}} : vector<3xf64>
  // CHECK:          %[[RAWN:[0-9]+]] = arith.truncf %[[RAW]] : vector<3xf64> to vector<3xf32>
  // CHECK:          %[[D:[0-9]+]] = arith.subf %[[RAWN]], %{{[0-9]+}} : vector<3xf32>
  // CHECK:          %[[R2:[0-9]+]] = vector.reduction <add>, %{{[0-9]+}} : vector<3xf32> into f32
  // CHECK:          %[[S:[0-9]+]] = vector.broadcast %[[R2]] : f32 to vector<3xf32>
  // CHECK:          %[[K:[0-9]+]] = arith.mulf %[[S]], %[[D]] : vector<3xf32>
  // CHECK:          %[[E:[0-9]+]] = arith.extf %[[R2]] : f32 to f64
  // CHECK:          arith.cmpf olt, %[[R2]], %[[CUTOFF2]] : f32
  // CHECK:        memref.store %{{[0-9]+}}, %[[F]][
  %f0 = md_exec.zeros : !forces
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell outs(%f0 : !forces)
      reduce(%u0 : f64) cutoff(2.5) weights [0.5]
      policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %s = vector.broadcast %r2 : f32 to vector<3xf32>
    %k = arith.mulf %s, %d : vector<3xf32>
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %k, %e : vector<3xf32>, f64
  } : !mdrt.neighbors<@atoms>, !positions -> !forces, f64
  return %f, %u : !forces, f64
}

// Positions in f32. The neighbor structure is built by the instance of the
// template for f32, and the cell is converted.
//
// CHECK-LABEL: func.func @single(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf32>, %[[BOX:[a-z0-9]+]]: vector<3xf64>, %{{[a-z0-9]+}}: index)
func.func @single(%x0: !single, %cell: !md.cell, %steps: index) -> !single {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK:      %[[REFERENCE:[a-z0-9_]+]] = memref.alloc(%{{[a-z0-9_]+}}) : memref<?x3xf32>
  // CHECK:      scf.for
  %nl0 = md_exec.empty_neighbors kind(matrix) width(96)
      : !mdrt.neighbors<@atoms>
  %x, %nl = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0, %nla = %nl0)
      -> (!single, !mdrt.neighbors<@atoms>) {
    // The test of the displacement since the last build.
    //
    // CHECK:        memref.load %[[REFERENCE]][
    // CHECK:        arith.cmpf ole, %{{[0-9]+}}, %{{[0-9]+}} : f32
    // CHECK:        scf.if
    // CHECK:          %[[NARROW:[0-9]+]] = arith.truncf %[[BOX]] : vector<3xf64> to vector<3xf32>
    // CHECK:          call @mdrt.build_neighbors_matrix_f32(%{{[a-z0-9_]+}}, %[[NARROW]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}},
    // CHECK-SAME:       : (memref<?x3xf32>, vector<3xf32>, f32, f32, memref<?xi32>, memref<?x?xi32>) -> index
    %nlb = md_exec.refresh_neighbors %nla, %xa, %cell
        cutoff(2.5) skin(0.3) cell_width(2.8) policy(check)
        : !mdrt.neighbors<@atoms>, !single
    %e = md_exec.empty : !single
    %xb = md_exec.particle_for ins(%xa : !single) outs(%e : !single) {
    ^bb0(%x_i: vector<3xf32>):
      md_exec.yield %x_i : vector<3xf32>
    } -> !single
    scf.yield %xb, %nlb : !single, !mdrt.neighbors<@atoms>
  }
  return %x : !single
}

// Both instances of the template are in the module.
//
// CHECK-DAG: func.func private @mdrt.build_neighbors_matrix(%{{[a-z0-9]+}}: memref<?x3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64,
// CHECK-DAG: func.func private @mdrt.build_neighbors_matrix_f32(%{{[a-z0-9]+}}: memref<?x3xf32>, %{{[a-z0-9]+}}: vector<3xf32>, %{{[a-z0-9]+}}: f32, %{{[a-z0-9]+}}: f32,
// CHECK-DAG: func.func private @mdrt.minimum_image_f32(%{{[a-z0-9]+}}: f32, %{{[a-z0-9]+}}: f32, %{{[a-z0-9]+}}: f32) -> f32
