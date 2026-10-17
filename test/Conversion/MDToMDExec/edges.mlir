// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.05 width=48" \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.05 width=48" \
// RUN:     --md-exec-choose-neighbors=kind=groups \
// RUN: | FileCheck %s --check-prefix=CHOOSE
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.05 width=48" \
// RUN:     --md-exec-assign-storage="memory=device" \
// RUN: | FileCheck %s --check-prefix=STORAGE
// RUN: mdir-opt %s --convert-md-to-md-exec="skin=0.05 width=48" \
// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --check-prefix=GPU
// RUN: not mdir-opt %s --convert-md-to-md-exec="skin=0.05 width=48" \
// RUN:     --md-exec-assign-storage 2>&1 \
// RUN: | FileCheck %s --check-prefix=HOST

// The pair list of a learned potential (D277,
// docs/metatomic-ml1.md, Section 12): the pairs of images of a neighborhood
// at the model's cutoff, which a host call takes in place.

!vec   = !md.field<@atoms, 3 x f64>
!ids   = !md.field<@atoms, i32>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

func.func private @model(memref<?x3xf64> {mdrt.read_only},
                         memref<?x5xi32> {mdrt.read_only},
                         memref<?xi32> {mdrt.read_only},
                         memref<?xi32> {mdrt.read_only}, i64,
                         memref<?x3xf64>)

// The neighborhood becomes a matrix built with the skin, and the edges are
// found in it at the same positions.
//
// CHECK-LABEL: func.func @step(
// CHECK:         %[[NL:[0-9]+]] = md_exec.build_neighbors %{{[0-9]+}}, %[[X:[a-z0-9]+]], %[[C:[0-9]+]]
// CHECK-SAME:      cutoff(0.69999999999999996) skin(5.000000e-02) kind(matrix)
// CHECK-NEXT:    %[[E:[0-9]+]] = md_exec.build_edges %[[NL]], %[[X]], %[[C]] cutoff(0.69999999999999996)
// CHECK-SAME:      : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64> -> !md.edges<@atoms>
// CHECK:         mdrt.host_call @model(%[[X]], %[[E]]) outs(%{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>) {in_place}
//
// The edges keep their structure a matrix, whose rows hold every candidate
// of a particle.
// CHOOSE:        md_exec.build_neighbors {{.*}} kind(matrix)
// CHOOSE-NEXT:   md_exec.build_edges
//
// In the storage form the call takes the slots, the counts, the offsets,
// and the number of builds, the buffers of the device where they are.
// STORAGE-LABEL: func.func @step(
// STORAGE:         %{{[a-z0-9_]+}}:4 = md_exec.build_edges %{{[0-9]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} cutoff(0.69999999999999996)
// STORAGE-SAME:      -> memref<?x5xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>, i64
// STORAGE:         call @model(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}#3, %{{[a-z0-9_]+}})
// STORAGE-SAME:      : (memref<?x3xf64>, memref<?x5xi32>, memref<?xi32>, memref<?xi32>, i64, memref<?x3xf64>) -> ()
//
// The structure holds the slots of the runtime; each build counts the
// candidates, reads them on the host, reserves the slots, and writes the
// candidates; each step writes the slots.
// GPU-LABEL: func.func @step(
// GPU:         call @mdrtEdgesCreate()
// GPU:         call @mdrt_gpu_build_neighbors_matrix
// GPU:         gpu.launch
// GPU:         gpu.memcpy
// GPU:         call @mdrtEdgesReserve(
// GPU:         call @mdrtEdgesBuffer(
// GPU:         gpu.launch
// GPU:         call @mdrtEdgesBuffer(
// GPU:         call @mdrtEdgesBuffer(
// GPU:         gpu.launch
// GPU:         call @model(
//
// HOST: error: 'md_exec.build_edges' op finds the pairs of images on a device only: a learned potential runs on a GPU
func.func @step(%x: !vec, %f: !vec, %lx: f64, %ly: f64, %lz: f64) -> !vec {
  %cell = md.orthorhombic_cell %lx, %ly, %lz
  %n = md.neighborhood %x, %cell cutoff(0.7) : !vec -> !pairs
  %e = md.edges %n, %x, %cell cutoff(0.7) : !pairs, !vec -> !md.edges<@atoms>
  %f1 = mdrt.host_call @model(%x, %e) outs(%f : !vec) {in_place}
      : (!vec, !md.edges<@atoms>) -> (!vec)
  return %f1 : !vec
}
