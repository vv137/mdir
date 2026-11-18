// The device build of the groups (D89) as its emitter writes it for each
// variant (GroupsBuild.h, D404): the cell, orthorhombic or
// triclinic, and the scan of the counting sorts, three launches or the
// decoupled look-back (D321). Each variant is a choice of the emitter: a
// variant has the code of its own choices and none of the others.
//
// RUN: mdir-opt %s --add-groups-build | FileCheck %s --check-prefixes=CHECK,ORTHO,TWO
// RUN: mdir-opt %s --add-groups-build="triclinic=true" | FileCheck %s --check-prefixes=CHECK,TRI,TWO
// RUN: mdir-opt %s --add-groups-build="single-pass-scan=true" | FileCheck %s --check-prefixes=CHECK,ORTHO,ONE
// RUN: mdir-opt %s --add-groups-build="triclinic=true single-pass-scan=true" | FileCheck %s --check-prefixes=CHECK,TRI,ONE
//
// The scans, scatters, sorts within bins, and reads of the ragged rows stay
// md_alg ops (D406) until they are lowered.
// RUN: mdir-opt %s --add-groups-build="triclinic=true single-pass-scan=true" \
// RUN:     --convert-md-alg-to-gpu | FileCheck %s --check-prefix=LOWERED

// The helpers; the partner of an excluded pair from its ragged row.
// CHECK-NOT: mdir.primitive
// CHECK: func.func private @mdrt_gpu_groups_grid(
// CHECK: func.func private @mdrt_gpu_groups_partner(
// CHECK: md_alg.ragged_entry %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] entry %{{.*}} : memref<?xi32, 1>
// CHECK: func.func private @mdrt_gpu_groups_image(
// ORTHO-NOT: triclinic
// TRI: func.func private @mdrt_gpu_groups_image_triclinic(
// TRI: func.func private @mdrt_gpu_groups_offset_triclinic(

// The build, of the cell of its variant: the tilts come from the last
// three elements of a triclinic cell.
// ORTHO: func.func private @mdrt_gpu_build_neighbors_groups(%{{.*}}: memref<?x3xf64, 1>, %{{.*}}: vector<3xf64>, %{{.*}}: f64,
// TRI: func.func private @mdrt_gpu_build_neighbors_groups_triclinic(%{{.*}}: memref<?x3xf64, 1>, %{{.*}}: vector<6xf64>, %{{.*}}: f64,
// TRI: vector.extract %{{.*}}[5] : f64 from vector<6xf64>

// The plans of the columns (about 64 particles in a cube) and of the grid
// of the candidates (cells of half the reach, 3 to 256 a side).
// CHECK: md_alg.grid_plan %{{.*}}, %{{.*}} : f64, f64 volume %{{.*}} : f64 occupancy 6.400000e+01 : f64 of %{{.*}} bounds[1.000000e+00 : f64, 2.560000e+02 : f64] -> i64, i64
// CHECK: md_alg.grid_plan %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64 width %{{.*}} : f64 bounds[3.000000e+00 : f64, 2.560000e+02 : f64] -> i64, i64, i64

// The counting sort by key, the scan of the chunks of the columns, and the
// counting sort by cell of the grid. The key of a particle writes its
// wrapped position and counts the positions without a bin; that of a
// place writes its position. The single-pass scan keeps the states of its
// tiles in i64, one more than the tiles.
// ONE: gpu.alloc ({{.*}}) : memref<?xi64, 1>
// CHECK: md_alg.sort_by_key %{{.*}} writes(%{{.*}} : memref<?x3xf32, 1>) adds(%{{.*}} : memref<5xi32, 1>) cleared : memref<?xi32, 1> launch(
// TWO-SAME: scan reduce_then_scan scratch %{{.*}} : memref<?xi32, 1> launch(
// ONE-SAME: scan single_pass scratch %{{.*}} : memref<?xi64, 1> launch(
// CHECK-SAME: sort insertion launch(
// CHECK: md_alg.no_bin
// CHECK: md_alg.bin %{{.*}} : index
// TWO: md_alg.scan %{{[^ ]*}} into %{{[^ ]*}} strategy reduce_then_scan
// ONE: md_alg.scan %{{[^ ]*}} into %{{[^ ]*}} strategy single_pass
// CHECK: md_alg.sort_by_key %{{.*}} writes(%{{.*}} : memref<?x4xf32, 1>) cleared : memref<?xi32, 1> launch(
// CHECK: md_alg.bin %{{.*}} : i32
// CHECK: md_alg.no_bin
// The excluded pairs of a particle from its ragged row.
// CHECK: md_alg.ragged_row %{{.*}} entry %{{.*}} checked : memref<?xi32, 1>

// The lists, then the images (27 codes): the shift of an entry in four
// bits a cell, (e + 4), or five a lattice vector, (e + 16), in a triclinic
// cell, whose images kernel takes the images by their offsets.
// ORTHO: arith.constant 4198404 : i32
// TRI: arith.constant 16793616 : i32
// The entries of a round of a warp, into blocks of 64 from the count of
// blocks: twice in the lists and twice in the images.
// CHECK-COUNT-2: md_alg.append %{{.*}} block 64 lane(
// CHECK: arith.constant 27 : i32
// TRI: func.call @mdrt_gpu_groups_offset_triclinic(
// CHECK-COUNT-2: md_alg.append %{{.*}} block 64 lane(
// CHECK: return
// ORTHO-NOT: triclinic
// ORTHO-NOT: 16793616
// TRI-NOT: 4198404
// TWO-NOT: single_pass
// ONE-NOT: reduce_then_scan

// LOWERED-NOT: md_alg
