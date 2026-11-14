// RUN: mdir-opt %s --md-exec-release-dead-storage | FileCheck %s

// The release of the buffers of the storage that the loop over steps and
// the code after it never use (D[md-exec-release-dead], #885), by the
// liveness of D412 with kills=proven. The first function and the four with
// a wait, a join, a loop, or a branch of the setup are released, after the
// op that holds the last occupied point. Every other function breaks one
// condition of the release and must keep its buffer.

// A buffer that only the setup uses is released after its last use, before
// the loop; B, which the loop uses, keeps its storage.
// CHECK-LABEL: func.func @setup_only
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: %[[B:.*]] = gpu.alloc
// CHECK: memref.load %[[A]]
// CHECK: gpu.terminator
// CHECK-NEXT: }
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @setup_only(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Read after the loop: kept.
// CHECK-LABEL: func.func @after_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  return
}

// Written whole by a launch after the loop and then read: kept. The
// buffer is dead from the end of the setup to that write under kills=stores,
// and so is a release that reads the liveness alone, not the occupancy.
// CHECK-LABEL: func.func @overwritten_after_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @overwritten_after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  return
}

// Filled after the loop, a proven kill, and then read: kept. A release that
// reads the liveness alone frees it under kills=proven as well.
// CHECK-LABEL: func.func @filled_after_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @filled_after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  %t = gpu.memset async %A, %f : memref<4xf32, 1>, f32
  gpu.wait [%t]
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %A[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  return
}

// Used in the loop (read at every step): kept.
// CHECK-LABEL: func.func @in_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @in_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Used in the loop through a view: kept.
// CHECK-LABEL: func.func @view_in_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @view_in_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  %V = memref.subview %A[0] [2] [1] : memref<4xf32, 1> to memref<2xf32, strided<[1]>, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %V[%c0] : memref<2xf32, strided<[1]>, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A raw pointer in the host's code pins it: kept.
// CHECK-LABEL: func.func @raw_pointer
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @raw_pointer(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  %p = memref.extract_aligned_pointer_as_index %A : memref<4xf32, 1> -> index
  func.call @keep(%p) : (index) -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Returned: kept.
// CHECK-LABEL: func.func @returned
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @returned(%out: memref<4xf32, 1>, %n: index, %c: i1) -> memref<4xf32, 1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return %A : memref<4xf32, 1>
}

// Given to an external function that the analysis does not know, which may
// keep the pointer: kept.
// CHECK-LABEL: func.func @unknown_external
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @unknown_external(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  func.call @keep_buffer(%A) : (memref<4xf32, 1>) -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Given to a function of the module that gives it to such an external
// function: kept.
// CHECK-LABEL: func.func @unknown_through_callee
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @unknown_through_callee(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  func.call @pass_on(%A) : (memref<4xf32, 1>) -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Given to an external function that enqueues work on it: released after
// the next wait, not at the call.
// CHECK-LABEL: func.func @enqueued
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: call @mdrtDeviceTuplesOrderEnd(%[[A]])
// CHECK-NOT: gpu.dealloc
// CHECK: gpu.wait
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @enqueued(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc (%n) : memref<?xi32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  call @mdrtDeviceTuplesOrderEnd(%A) : (memref<?xi32, 1>) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %B[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.wait
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// The same with the next wait after the loop: kept.
// CHECK-LABEL: func.func @enqueued_wait_after_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @enqueued_wait_after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc (%n) : memref<?xi32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  call @mdrtDeviceTuplesOrderEnd(%A) : (memref<?xi32, 1>) -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  gpu.wait
  return
}

// The same with no wait at all: pinned, kept.
// CHECK-LABEL: func.func @enqueued_no_wait
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @enqueued_no_wait(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc (%n) : memref<?xi32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  call @mdrtDeviceTuplesOrderEnd(%A) : (memref<?xi32, 1>) -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// An asynchronous fill: released after the wait of its token.
// CHECK-LABEL: func.func @async_wait
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: gpu.memset async %[[A]]
// CHECK-NOT: gpu.dealloc
// CHECK: gpu.wait [
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
func.func @async_wait(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  %t = gpu.memset async %A, %f : memref<4xf32, 1>, f32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %B[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.wait [%t]
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Its token waited for in the loop: kept.
// CHECK-LABEL: func.func @async_wait_in_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @async_wait_in_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  %t = gpu.memset async %A, %f : memref<4xf32, 1>, f32
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.wait [%t]
  }
  return
}

// Its token never waited for: pinned, kept.
// CHECK-LABEL: func.func @async_no_wait
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @async_no_wait(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  %t = gpu.memset async %A, %f : memref<4xf32, 1>, f32
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Used on a side stream that is joined before the loop: released after the
// join.
// CHECK-LABEL: func.func @side_joined
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: call @mdrtSideEnd()
// CHECK-NOT: gpu.dealloc
// CHECK: call @mdrtSideJoin()
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
func.func @side_joined(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  func.call @mdrtSideBegin() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  func.call @mdrtSideEnd() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %B[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  func.call @mdrtSideJoin() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Joined only after the loop: kept.
// CHECK-LABEL: func.func @side_joined_after_loop
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @side_joined_after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  func.call @mdrtSideBegin() : () -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  func.call @mdrtSideEnd() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  func.call @mdrtSideJoin() : () -> ()
  return
}

// Last used in a loop of the setup: released after that loop, not in it.
// CHECK-LABEL: func.func @setup_loop
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: scf.for
// CHECK-NOT: gpu.dealloc
// CHECK: memref.load %[[A]]
// CHECK-NEXT: memref.store
// CHECK-NEXT: gpu.terminator
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
func.func @setup_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %j = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Last used in a branch of the setup: released after the branch.
// CHECK-LABEL: func.func @setup_branch
// CHECK: %[[A:.*]] = gpu.alloc
// CHECK: scf.if
// CHECK-NOT: gpu.dealloc
// CHECK: memref.load %[[A]]
// CHECK-NEXT: memref.store
// CHECK-NEXT: gpu.terminator
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: %[[P:.*]] = memref.memory_space_cast %[[A]]
// CHECK-NEXT: gpu.dealloc %[[P]] :
// CHECK-NEXT: scf.for
func.func @setup_branch(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.if %c {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Released by the code already: not released again.
// CHECK-LABEL: func.func @released
// CHECK: gpu.dealloc
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @released(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  gpu.dealloc %A : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// Memory shared with the host, and a buffer nothing uses: kept.
// CHECK-LABEL: func.func @not_the_pool
// CHECK-NOT: gpu.dealloc
// CHECK: return
func.func @not_the_pool(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc host_shared () : memref<4xf32, 1>
  %U = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %A[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

func.func private @keep(index)
func.func private @keep_buffer(memref<4xf32, 1>)
func.func private @pass_on(%x: memref<4xf32, 1>) {
  func.call @keep_buffer(%x) : (memref<4xf32, 1>) -> ()
  return
}
func.func private @mdrtDeviceTuplesOrderEnd(memref<?xi32, 1>)
func.func private @mdrtSideBegin()
func.func private @mdrtSideEnd()
func.func private @mdrtSideJoin()
