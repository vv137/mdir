// The order of the lanes of the deterministic width in f32, 8, for the
// forces and the energy (D[host-pair-vector]): lane l with lane l + 4, then
// l + 2, then l + 1, the order of lane 0 of emitWarpReduce. Each force
// component is combined in f32; the energy is widened in each lane first
// and combined in f64.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @lj(
// CHECK:      %[[ROW:[0-9]+]] = vector.load %{{[0-9]+}}[%{{[a-z0-9]+}}, %{{[a-z0-9]+}}] : memref<?x?xi32>, vector<8xi32>
// CHECK:      %[[TAIL:[0-9]+]]:4 = scf.if
// CHECK:      } else {
// CHECK:      }
// CHECK:      %[[F0A:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#0 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F0B:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#0 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F04:[0-9]+]] = arith.addf %[[F0A]], %[[F0B]] : vector<4xf32>
// CHECK-NEXT: %[[F0C:[0-9]+]] = vector.extract_strided_slice %[[F04]] {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F0D:[0-9]+]] = vector.extract_strided_slice %[[F04]] {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F02:[0-9]+]] = arith.addf %[[F0C]], %[[F0D]] : vector<2xf32>
// CHECK-NEXT: %[[F0E:[0-9]+]] = vector.extract_strided_slice %[[F02]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F0F:[0-9]+]] = vector.extract_strided_slice %[[F02]] {offsets = [1], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F01:[0-9]+]] = arith.addf %[[F0E]], %[[F0F]] : vector<1xf32>
// CHECK-NEXT: %[[F0:[0-9]+]] = vector.extract %[[F01]][0] : f32 from vector<1xf32>
// CHECK:      %[[F1A:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#1 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F1B:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#1 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F14:[0-9]+]] = arith.addf %[[F1A]], %[[F1B]] : vector<4xf32>
// CHECK-NEXT: %[[F1C:[0-9]+]] = vector.extract_strided_slice %[[F14]] {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F1D:[0-9]+]] = vector.extract_strided_slice %[[F14]] {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F12:[0-9]+]] = arith.addf %[[F1C]], %[[F1D]] : vector<2xf32>
// CHECK-NEXT: %[[F1E:[0-9]+]] = vector.extract_strided_slice %[[F12]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F1F:[0-9]+]] = vector.extract_strided_slice %[[F12]] {offsets = [1], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F11:[0-9]+]] = arith.addf %[[F1E]], %[[F1F]] : vector<1xf32>
// CHECK-NEXT: %[[F1:[0-9]+]] = vector.extract %[[F11]][0] : f32 from vector<1xf32>
// CHECK:      %[[F2A:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#2 {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F2B:[0-9]+]] = vector.extract_strided_slice %[[TAIL]]#2 {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
// CHECK-NEXT: %[[F24:[0-9]+]] = arith.addf %[[F2A]], %[[F2B]] : vector<4xf32>
// CHECK-NEXT: %[[F2C:[0-9]+]] = vector.extract_strided_slice %[[F24]] {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F2D:[0-9]+]] = vector.extract_strided_slice %[[F24]] {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[F22:[0-9]+]] = arith.addf %[[F2C]], %[[F2D]] : vector<2xf32>
// CHECK-NEXT: %[[F2E:[0-9]+]] = vector.extract_strided_slice %[[F22]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F2F:[0-9]+]] = vector.extract_strided_slice %[[F22]] {offsets = [1], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[F21:[0-9]+]] = arith.addf %[[F2E]], %[[F2F]] : vector<1xf32>
// CHECK-NEXT: %[[F2:[0-9]+]] = vector.extract %[[F21]][0] : f32 from vector<1xf32>
// CHECK-NEXT: %[[FORCE:[0-9]+]] = vector.from_elements %[[F0]], %[[F1]], %[[F2]] : vector<3xf32>
// CHECK-NEXT: %[[WIDE:[0-9]+]] = arith.extf %[[TAIL]]#3 : vector<8xf32> to vector<8xf64>
// CHECK:      %[[UA:[0-9]+]] = vector.extract_strided_slice %[[WIDE]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf64> to vector<4xf64>
// CHECK-NEXT: %[[UB:[0-9]+]] = vector.extract_strided_slice %[[WIDE]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf64> to vector<4xf64>
// CHECK-NEXT: %[[U4:[0-9]+]] = arith.addf %[[UA]], %[[UB]] : vector<4xf64>
// CHECK-NEXT: %[[UC:[0-9]+]] = vector.extract_strided_slice %[[U4]] {offsets = [0], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[UD:[0-9]+]] = vector.extract_strided_slice %[[U4]] {offsets = [2], sizes = [2], strides = [1]}
// CHECK-NEXT: %[[U2:[0-9]+]] = arith.addf %[[UC]], %[[UD]] : vector<2xf64>
// CHECK-NEXT: %[[UE:[0-9]+]] = vector.extract_strided_slice %[[U2]] {offsets = [0], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[UF:[0-9]+]] = vector.extract_strided_slice %[[U2]] {offsets = [1], sizes = [1], strides = [1]}
// CHECK-NEXT: %[[U1:[0-9]+]] = arith.addf %[[UE]], %[[UF]] : vector<1xf64>
// CHECK-NEXT: %[[U:[0-9]+]] = vector.extract %[[U1]][0] : f64 from vector<1xf64>
// CHECK:      memref.store %{{[0-9]+}}, %{{[a-z0-9_]+}}[%{{[a-z0-9]+}}, %{{[a-z0-9_]+}}] : memref<?x3xf32>

!vec  = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>

md.particle_set @atoms

func.func @lj(%x: !vec, %q: !real, %cell: !md.cell) -> (!vec, f64) {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell ins(%q : !real) outs(%f0 : !vec)
      reduce(%u0 : f64) cutoff(1.0) weights [0.5]
      exchange [antisymmetric, symmetric] policy(directed, owner_only)
      host(lanes(8)) {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
    %one = arith.constant 1.0 : f32
    %c3 = arith.constant 3 : i32
    %inv = arith.divf %one, %r2 : f32
    %s6 = math.fpowi %inv, %c3 : f32, i32
    %qq = arith.mulf %qi, %qj : f32
    %e = arith.mulf %s6, %qq : f32
    %b = vector.broadcast %e : f32 to vector<3xf32>
    %fd = arith.mulf %b, %d : vector<3xf32>
    %wide = arith.extf %e : f32 to f64
    md_exec.yield %fd, %wide : vector<3xf32>, f64
  } : !mdrt.neighbors<@atoms>, !vec -> !vec, f64
  return %f, %u : !vec, f64
}
