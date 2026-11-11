// The rows of a loop over pairs in lanes on the host (D[host-pair-vector]):
// full groups of the row read by vector.load, one masked tail whose padding
// repeats the row's last candidate and whose mask enters every sum, a read a lane of the positions and the
// fields, the kernel on vectors of the lanes (llvm.intr.powi for a constant
// power), the energy summed in f32 in each lane and widened there, and the
// lanes combined by halving.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | FileCheck %s

!vec  = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>

md.particle_set @atoms

// CHECK-LABEL: func.func @lj(
func.func @lj(%x: !vec, %q: !real, %cell: !md.cell) -> (!vec, f64) {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  // CHECK:      scf.parallel (%[[I:[a-z0-9]+]]) =
  // CHECK:        %[[REST:[0-9]+]] = arith.remui %{{[0-9]+}}, %[[FOUR:[a-z0-9_]+]] : index
  // CHECK:        %[[FULL:[0-9]+]] = arith.subi %{{[0-9]+}}, %[[REST]] : index
  // CHECK:        %[[GROUPS:[0-9]+]]:4 = scf.for %[[K:[a-z0-9]+]] = %{{[a-z0-9_]+}} to %[[FULL]] step %[[FOUR]]
  // CHECK-SAME:     -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  // CHECK:          %[[ROW:[0-9]+]] = vector.load %[[ENTRIES:[0-9]+]][%[[I]], %[[K]]] : memref<?x?xi32>, vector<4xi32>
  // CHECK:          %[[OTHERS:[0-9]+]] = arith.index_cast %[[ROW]] : vector<4xi32> to vector<4xindex>
  // CHECK:          vector.extract %[[OTHERS]][3] : index from vector<4xindex>
  // CHECK:          math.roundeven %{{[0-9]+}} : vector<4xf32>
  // CHECK:          llvm.intr.powi(%{{[0-9]+}}, %{{[a-z0-9_]+}}) : (vector<4xf32>, i32) -> vector<4xf32>
  // CHECK:          arith.cmpf olt, %{{[0-9]+}}, %{{[0-9]+}} : vector<4xf32>
  // CHECK:          arith.cmpi ne, %[[OTHERS]], %{{[0-9]+}} : vector<4xindex>
  // CHECK:        %[[TAIL:[0-9]+]]:4 = scf.if %{{[0-9]+}} -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // CHECK:          %[[LAST:[0-9]+]] = memref.load %[[ENTRIES]][%[[I]], %{{[0-9]+}}] : memref<?x?xi32>
  // CHECK:          %[[PAD:[0-9]+]] = vector.broadcast %[[LAST]] : i32 to vector<4xi32>
  // CHECK:          %[[MASK:[0-9]+]] = vector.create_mask %[[REST]] : vector<4xi1>
  // CHECK:          vector.maskedload %[[ENTRIES]][%[[I]], %[[FULL]]], %[[MASK]], %[[PAD]]
  // The lanes past the row (the padding, which repeats the last candidate)
  // are masked out of every sum.
  // CHECK:          %[[WITHIN:[0-9]+]] = arith.cmpf olt, %{{[0-9]+}}, %{{[0-9]+}} : vector<4xf32>
  // CHECK:          %[[DISTINCT:[0-9]+]] = arith.cmpi ne, %{{[0-9]+}}, %{{[0-9]+}} : vector<4xindex>
  // CHECK:          %[[PAIR:[0-9]+]] = arith.andi %[[WITHIN]], %[[DISTINCT]] : vector<4xi1>
  // CHECK:          %[[KEEP:[0-9]+]] = arith.andi %[[PAIR]], %[[MASK]] : vector<4xi1>
  // CHECK:          arith.select %[[KEEP]],
  // CHECK:          arith.select %[[KEEP]],
  // CHECK:          arith.select %[[KEEP]],
  // CHECK:          arith.select %[[KEEP]],
  // CHECK-NOT:      arith.select
  // CHECK:        } else {
  // CHECK:          scf.yield %[[GROUPS]]#0, %[[GROUPS]]#1, %[[GROUPS]]#2, %[[GROUPS]]#3
  // CHECK:        vector.extract_strided_slice %[[TAIL]]#0 {offsets = [0], sizes = [2], strides = [1]}
  // CHECK:        %[[WIDE:[0-9]+]] = arith.extf %[[TAIL]]#3 : vector<4xf32> to vector<4xf64>
  // CHECK:        %[[LOW:[0-9]+]] = vector.extract_strided_slice %[[WIDE]] {offsets = [0], sizes = [2], strides = [1]}
  // CHECK:        %[[HIGH:[0-9]+]] = vector.extract_strided_slice %[[WIDE]] {offsets = [2], sizes = [2], strides = [1]}
  // CHECK:        %[[HALF:[0-9]+]] = arith.addf %[[LOW]], %[[HIGH]] : vector<2xf64>
  // CHECK:        vector.extract_strided_slice %[[HALF]] {offsets = [1], sizes = [1], strides = [1]}
  // CHECK:        memref.store %{{[0-9]+}}, %{{[a-z0-9_]+}}[%[[I]], %{{[a-z0-9_]+}}] : memref<?x3xf32>
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell ins(%q : !real) outs(%f0 : !vec)
      reduce(%u0 : f64) cutoff(1.0) weights [0.5]
      exchange [antisymmetric, symmetric] policy(directed, owner_only)
      host(lanes(4)) {
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
