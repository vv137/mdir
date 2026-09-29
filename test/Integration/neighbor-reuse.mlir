// A neighbor structure that a loop carries and refreshes: it is rebuilt only
// when it is no longer valid, and it never misses a pair.
//
// RUN: mdir-opt %s --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: mdir-opt %s --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s

// 200 particles move at constant velocity for 100 steps in a periodic cube
// with the edge 6. At every step the pairs within the cutoff 1.5 are
// counted through the neighbor structure, which has the skin 0.3.
//
// The reference values come from Inputs/neighbor_reuse_reference.py, which
// tests all pairs at every step.

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()

// Fills `x` with values in [offset, offset + scale), continuing the
// generator from `seed`. Returns the state of the generator.
func.func @fill(%x: memref<?x3xf64>, %scale: f64, %offset: f64, %seed: i64)
    -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %scaled = arith.mulf %f, %scale : f64
      %value = arith.addf %scaled, %offset : f64
      memref.store %value, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return %last : i64
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 200 : index
  %steps = arith.constant 100 : index

  %zero = arith.constant 0.0 : f64
  %edge = arith.constant 6.0 : f64
  %two = arith.constant 2.0 : f64
  %minus = arith.constant -1.0 : f64
  %dt = arith.constant 0.01 : f64
  %seed = arith.constant 42 : i64

  %positions = memref.alloc(%count) : memref<?x3xf64>
  %velocities = memref.alloc(%count) : memref<?x3xf64>
  %state = call @fill(%positions, %edge, %zero, %seed)
      : (memref<?x3xf64>, f64, f64, i64) -> i64
  %unused = call @fill(%velocities, %two, %minus, %state)
      : (memref<?x3xf64>, f64, f64, i64) -> i64

  %x0 = mdrt.from_buffer %positions : memref<?x3xf64> to !vec
  %v = mdrt.from_buffer %velocities : memref<?x3xf64> to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %nl0 = md_exec.empty_neighbors kind(matrix) width(64) : !nl
  %x, %nl, %total = scf.for %step = %c0 to %steps step %c1
      iter_args(%xa = %x0, %na = %nl0, %sum = %zero) -> (!vec, !nl, f64) {
    %destination = md_exec.empty : !vec
    %xb = md_exec.particle_for ins(%xa, %v : !vec, !vec)
        outs(%destination : !vec) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %dtv = vector.broadcast %dt : f64 to vector<3xf64>
      %change = arith.mulf %dtv, %v_i : vector<3xf64>
      %moved = arith.addf %x_i, %change : vector<3xf64>
      md_exec.yield %moved : vector<3xf64>
    } -> !vec

    %nb = md_exec.refresh_neighbors %na, %xb, %cell
        cutoff(1.5) skin(0.3) cell_width(1.8) policy(check) : !nl, !vec

    %none = arith.constant 0.0 : f64
    %pairs = md_exec.pair_for %nb, %xb, %cell reduce(%none : f64)
        cutoff(1.5) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      %one = arith.constant 1.0 : f64
      md_exec.yield %one : f64
    } : !nl, !vec -> f64

    %next = arith.addf %sum, %pairs : f64
    scf.yield %xb, %nb, %next : !vec, !nl, f64
  }

  // The pairs within the cutoff, summed over the steps.
  // CHECK:      128826
  call @printF64(%total) : (f64) -> ()
  call @printNewline() : () -> ()

  // The builds of the neighbor structure: 10 in 100 steps.
  // CHECK-NEXT: 10
  %builds = md_exec.rebuild_count %nl : !nl
  call @printI64(%builds) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}
