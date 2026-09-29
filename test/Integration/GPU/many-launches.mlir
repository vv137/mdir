// A loop of the host that launches a kernel in every iteration. The
// arguments of a launch are put on the stack, so the loop must release the
// stack in every iteration: 200000 launches would not fit otherwise.
//
// The host does not wait for a kernel to end. It waits where it reads what
// the device has computed, so the copy at the end sees all the launches.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %s

// With MDRT_WAIT, the host waits after every launch.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu \
// RUN: | mlir-opt %lower_gpu_to_llvm \
// RUN: | env MDRT_WAIT=1 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \
// RUN: | FileCheck %s

md.particle_set @atoms

func.func private @printF64(f64)
func.func private @printNewline()

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %n = arith.constant 300 : index
  %steps = arith.constant 200000 : index

  %host = memref.alloc(%n) : memref<?x3xf64>
  %start = arith.constant 1.0 : f64
  scf.for %i = %c0 to %n step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      memref.store %start, %host[%i, %k] : memref<?x3xf64>
    }
  }

  %device = gpu.alloc (%n) : memref<?x3xf64, 1>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %device, %host
      : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%t1]

  %change = arith.constant 0.5 : f64
  scf.for %step = %c0 to %steps step %c1 {
    md_exec.particle_for ins(%device : memref<?x3xf64, 1>)
        outs(%device : memref<?x3xf64, 1>) {
    ^bb0(%x_i: vector<3xf64>):
      %d = vector.broadcast %change : f64 to vector<3xf64>
      %new = arith.addf %x_i, %d : vector<3xf64>
      md_exec.yield %new : vector<3xf64>
    }
  }

  %t2 = gpu.wait async
  %t3 = gpu.memcpy async [%t2] %host, %device
      : memref<?x3xf64>, memref<?x3xf64, 1>
  gpu.wait [%t3]

  // 1 + 200000 / 2, for the first and for the last particle.
  // CHECK:      100001
  // CHECK-NEXT: 100001
  %last = arith.subi %n, %c1 : index
  %first = memref.load %host[%c0, %c0] : memref<?x3xf64>
  %other = memref.load %host[%last, %c2] : memref<?x3xf64>
  call @printF64(%first) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%other) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
