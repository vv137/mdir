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
// The calls of the primitives stay calls until they are expanded.
// RUN: mdir-opt %s --add-groups-build="triclinic=true single-pass-scan=true" \
// RUN:     --expand-device-primitives | FileCheck %s --check-prefix=EXPANDED

// The declarations of the primitives, those of the single-pass scan only
// where it is taken.
// CHECK: func.func private @mdir.primitive.device_scan_cursor(
// TWO-NOT: single_pass_scan
// ONE: func.func private @mdir.primitive.single_pass_scan(memref<?xi32, 1>, memref<?xi32, 1>, index, memref<?xi64, 1>, index, index, index)
// ONE: func.func private @mdir.primitive.single_pass_scan_cursor(
// CHECK: func.func private @mdir.primitive.cursor_scatter(
// CHECK: func.func private @mdrt_gpu_groups_grid(
// CHECK: func.func private @mdrt_gpu_groups_partner(
// CHECK: func.func private @mdrt_gpu_groups_image(
// ORTHO-NOT: triclinic
// TRI: func.func private @mdrt_gpu_groups_image_triclinic(
// TRI: func.func private @mdrt_gpu_groups_offset_triclinic(

// The build, of the cell of its variant: the tilts come from the last
// three elements of a triclinic cell.
// ORTHO: func.func private @mdrt_gpu_build_neighbors_groups(%{{.*}}: memref<?x3xf64, 1>, %{{.*}}: vector<3xf64>, %{{.*}}: f64,
// TRI: func.func private @mdrt_gpu_build_neighbors_groups_triclinic(%{{.*}}: memref<?x3xf64, 1>, %{{.*}}: vector<6xf64>, %{{.*}}: f64,
// TRI: vector.extract %{{.*}}[5] : f64 from vector<6xf64>

// The three counting sorts: by key with a cursor, the chunks of the
// columns, by cell of the grid with a cursor. The single-pass scan keeps
// the states of its tiles in i64, one more than the tiles.
// TWO: call @mdir.primitive.device_scan_cursor(%{{.*}}) : (memref<?xi32, 1>, memref<?xi32, 1>, index, memref<?xi32, 1>, index, index, index, memref<?xi32, 1>) -> ()
// ONE: gpu.alloc ({{.*}}) : memref<?xi64, 1>
// ONE: call @mdir.primitive.single_pass_scan_cursor(
// TWO: call @mdir.primitive.device_scan(
// ONE: call @mdir.primitive.single_pass_scan(
// TWO: call @mdir.primitive.device_scan_cursor(
// ONE: call @mdir.primitive.single_pass_scan_cursor(

// The lists, then the images (27 codes): the shift of an entry in four
// bits a cell, (e + 4), or five a lattice vector, (e + 16), in a triclinic
// cell, whose images kernel takes the images by their offsets.
// ORTHO: arith.constant 4198404 : i32
// TRI: arith.constant 16793616 : i32
// CHECK: arith.constant 27 : i32
// TRI: func.call @mdrt_gpu_groups_offset_triclinic(
// CHECK: return
// ORTHO-NOT: triclinic
// ORTHO-NOT: 16793616
// TRI-NOT: 4198404
// TWO-NOT: single_pass_scan
// ONE-NOT: device_scan

// EXPANDED-NOT: mdir.primitive
