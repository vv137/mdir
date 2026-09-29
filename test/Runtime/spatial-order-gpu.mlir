// The template that orders the particles by cell on a device, compared
// with the template for the host, entry by entry.
//
// REQUIRES: cuda
//
// RUN: cat %S/../../lib/Runtime/Templates/NeighborsMatrix.mlir \
// RUN:     %S/../../lib/Runtime/Templates/NeighborsMatrixGPU.mlir %s \
// RUN: | mlir-opt --gpu-lower-to-nvvm-pipeline="cubin-format=isa" \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt_cuda \
// RUN: | FileCheck %s

// Each case places 2000 particles, as spatial-order.mlir does, and prints
// the number of places in which the two orders differ.

func.func private @printI64(i64)
func.func private @printNewline()

func.func @fill(%x: memref<?x3xf64>, %length: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %two = arith.constant 2.0 : f64
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %image = arith.remui %i, %c5 : index
    %wide = arith.index_cast %image : index to i64
    %real = arith.sitofp %wide : i64 to f64
    %away = arith.subf %real, %two : f64
    %shift = arith.mulf %away, %length : f64
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %v = arith.mulf %f, %length : f64
      %moved = arith.addf %v, %shift : f64
      memref.store %moved, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @run(%length: f64, %width: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 2000 : index
  %prime = arith.constant 7919 : index
  %zero = arith.constant 0 : i64

  %x = memref.alloc(%count) : memref<?x3xf64>
  call @fill(%x, %length) : (memref<?x3xf64>, f64) -> ()
  %box = vector.broadcast %length : f64 to vector<3xf64>
  %ids = memref.alloc(%count) : memref<?xi32>
  scf.for %i = %c0 to %count step %c1 {
    %scattered = arith.muli %i, %prime : index
    %id = arith.remui %scattered, %count : index
    %narrow = arith.index_cast %id : index to i32
    memref.store %narrow, %ids[%i] : memref<?xi32>
  }

  // On the host.
  %order = memref.alloc(%count) : memref<?xi32>
  call @mdrt.spatial_order(%x, %box, %width, %ids, %order)
      : (memref<?x3xf64>, vector<3xf64>, f64, memref<?xi32>, memref<?xi32>)
        -> ()

  // On the device.
  %xd = gpu.alloc (%count) : memref<?x3xf64, 1>
  %idsd = gpu.alloc (%count) : memref<?xi32, 1>
  %orderd = gpu.alloc (%count) : memref<?xi32, 1>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %xd, %x : memref<?x3xf64, 1>, memref<?x3xf64>
  %t2 = gpu.memcpy async [%t1] %idsd, %ids : memref<?xi32, 1>, memref<?xi32>
  gpu.wait [%t2]
  call @mdrt_gpu_spatial_order(%xd, %box, %width, %idsd, %orderd)
      : (memref<?x3xf64, 1>, vector<3xf64>, f64, memref<?xi32, 1>,
         memref<?xi32, 1>) -> ()
  %order2 = memref.alloc(%count) : memref<?xi32>
  %t3 = gpu.wait async
  %t4 = gpu.memcpy async [%t3] %order2, %orderd
      : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%t4]

  %wrong = scf.for %k = %c0 to %count step %c1
      iter_args(%w = %zero) -> (i64) {
    %a = memref.load %order[%k] : memref<?xi32>
    %b = memref.load %order2[%k] : memref<?xi32>
    %differs = arith.cmpi ne, %a, %b : i32
    %d = arith.extui %differs : i1 to i64
    %next = arith.addi %w, %d : i64
    scf.yield %next : i64
  }
  call @printI64(%wrong) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  // Eight cells along each direction, with 4 particles in a cell.
  // CHECK:      {{^0$}}
  %l0 = arith.constant 12.0 : f64
  %w0 = arith.constant 1.4 : f64
  call @run(%l0, %w0) : (f64, f64) -> ()

  // Two cells along each direction, with 250 particles in a cell.
  // CHECK-NEXT: {{^0$}}
  %w1 = arith.constant 5.0 : f64
  call @run(%l0, %w1) : (f64, f64) -> ()

  // More cells than particles.
  // CHECK-NEXT: {{^0$}}
  %w2 = arith.constant 0.5 : f64
  call @run(%l0, %w2) : (f64, f64) -> ()
  return
}
