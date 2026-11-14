// RUN: mdir-opt %s --test-md-exec-print-storage-liveness -o /dev/null | FileCheck %s
// RUN: mdir-opt %s --test-md-exec-print-storage-liveness=kills=proven -o /dev/null | FileCheck %s --check-prefix=PROVEN

// The liveness of the storage of md_exec within the loop over steps
// (D412, #880). Each function is one rule, with the
// bug it must catch. A buffer must interfere with one whose life lies
// inside its own when it is read after the loop, carried by the loop,
// used by an asynchronous launch until its wait or by a side stream until
// its join, given to an external function that enqueues work on it (until
// the next wait, or the join in a forked region), or reached through a
// view. A write that is partial, conditional, or not proven
// (kills=proven) must not end a life. An external function that the
// analysis does not know may keep its argument, which pins it. The points
// of a step are numbered from the loop's head (0); each launch, call, copy,
// fill, or wait that touches a buffer is one point.

// Two scratch buffers of one step whose uses do not overlap.
// CHECK-LABEL: storage liveness of @disjoint (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-2) first 1 last 2{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(3-4) first 3 last 4{{$}}
// CHECK-NEXT: s0 interferes in the step with none{{$}}
// CHECK-NEXT: s1 interferes in the step with none{{$}}
// PROVEN-LABEL: storage liveness of @disjoint (kills=proven): 2 buffers, 5 points in the step
// PROVEN-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 2 carried{{$}}
// PROVEN-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 3 last 4 carried{{$}}
// PROVEN-NEXT: s0 interferes in the step with s1{{$}}
// PROVEN-NEXT: s1 interferes in the step with s0{{$}}
func.func @disjoint(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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

// The same, each written by a fill: a kill in both modes.
// CHECK-LABEL: storage liveness of @memset (kills=stores): 2 buffers, 7 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-3) first 1 last 3{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(4-6) first 4 last 6{{$}}
// CHECK-NEXT: s0 interferes in the step with none{{$}}
// CHECK-NEXT: s1 interferes in the step with none{{$}}
// PROVEN-LABEL: storage liveness of @memset (kills=proven): 2 buffers, 7 points in the step
// PROVEN-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-3) first 1 last 3{{$}}
// PROVEN-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(4-6) first 4 last 6{{$}}
// PROVEN-NEXT: s0 interferes in the step with none{{$}}
// PROVEN-NEXT: s1 interferes in the step with none{{$}}
func.func @memset(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %t0 = gpu.memset async %A, %f : memref<4xf32, 1>, f32
    gpu.wait [%t0]
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    %t1 = gpu.memset async %B, %f : memref<4xf32, 1>, f32
    gpu.wait [%t1]
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A buffer read after the loop is live from its write to the end of the
// step: it interferes with B, used after its last use in the step.
// CHECK-LABEL: storage liveness of @after_loop (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 2 after-loop{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(3-4) first 3 last 4{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @after_loop(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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

// Buffers carried by the loop (swapped by its yield), and one that a step
// reads before it overwrites it (F), are live through the whole step.
// CHECK-LABEL: storage liveness of @carried (kills=stores): 4 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 1 carried{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 1 carried{{$}}
// CHECK-NEXT: s2 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-1,4) first 1 last 4 carried{{$}}
// CHECK-NEXT: s3 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1 s2 s3{{$}}
// CHECK-NEXT: s1 interferes in the step with s0 s2 s3{{$}}
// CHECK-NEXT: s2 interferes in the step with s0 s1{{$}}
// CHECK-NEXT: s3 interferes in the step with s0 s1{{$}}
func.func @carried(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %P = gpu.alloc () : memref<4xf32, 1>
  %Q = gpu.alloc () : memref<4xf32, 1>
  %F = gpu.alloc () : memref<4xf32, 1>
  %S = gpu.alloc () : memref<4xf32, 1>
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%P0 = %P, %Q1 = %Q) -> (memref<4xf32, 1>, memref<4xf32, 1>) {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %F[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      %u = memref.load %P0[%c0] : memref<4xf32, 1>
      memref.store %u, %Q1[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %S[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %S[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %F[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    scf.yield %Q1, %P0 : memref<4xf32, 1>, memref<4xf32, 1>
  }
  return
}

// A launch whose token is waited for later uses A until the wait, which
// B's whole life precedes.
// CHECK-LABEL: storage liveness of @async (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @async(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %t0 = gpu.wait async
    %t1 = gpu.launch async [%t0] blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    %t2 = gpu.wait async [%t1]
    gpu.wait [%t2]
  }
  return
}

// A is read through a view (subview) after B's life: A interferes with B.
// CHECK-LABEL: storage liveness of @view_subview (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @view_subview(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %V = memref.subview %A[0] [2] [1] : memref<4xf32, 1> to memref<2xf32, strided<[1]>, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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
      %v = memref.load %V[%c0] : memref<2xf32, strided<[1]>, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A is read through a view (cast) after B's life: A interferes with B.
// CHECK-LABEL: storage liveness of @view_cast (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @view_cast(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %V = memref.cast %A : memref<4xf32, 1> to memref<?xf32, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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
      %v = memref.load %V[%c0] : memref<?xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A is read through a view (reinterpret_cast) after B's life: A interferes with B.
// CHECK-LABEL: storage liveness of @view_reinterpret_cast (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @view_reinterpret_cast(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %V = memref.reinterpret_cast %A to offset: [0], sizes: [4], strides: [1] : memref<4xf32, 1> to memref<4xf32, strided<[1]>, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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
      %v = memref.load %V[%c0] : memref<4xf32, strided<[1]>, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A is read through a view (memory_space_cast) after B's life: A interferes with B.
// CHECK-LABEL: storage liveness of @view_memory_space_cast (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @view_memory_space_cast(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %V = memref.memory_space_cast %A : memref<4xf32, 1> to memref<4xf32>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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
      %v = memref.load %V[%c0] : memref<4xf32>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// The same with the view made in the body of the launch.
// CHECK-LABEL: storage liveness of @view_in_kernel (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @view_in_kernel(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
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
      %V = memref.subview %A[1] [2] [1] : memref<4xf32, 1> to memref<2xf32, strided<[1], offset: 1>, 1>
      %v = memref.load %V[%c0] : memref<2xf32, strided<[1], offset: 1>, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A write through a subview is not a kill: A is carried.
// CHECK-LABEL: storage liveness of @partial_write (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 4 carried{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @partial_write(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %V = memref.subview %A[0] [2] [1] : memref<4xf32, 1> to memref<2xf32, strided<[1]>, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %V[%c0] : memref<2xf32, strided<[1]>, 1>
      gpu.terminator
    }
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
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A raw pointer of A given to an external function in the host's code pins A.
// CHECK-LABEL: storage liveness of @pointer (kills=stores): 2 buffers, 4 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-3) first 1 last 1 carried pinned(a raw pointer in the host's code){{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @pointer(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %p = memref.extract_aligned_pointer_as_index %A : memref<4xf32, 1> -> index
    func.call @keep(%p) : (index) -> ()
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

// A launch on a side stream uses A until the stream is joined.
// CHECK-LABEL: storage liveness of @side (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @side(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
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
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @mdrtSideJoin() : () -> ()
  }
  return
}

// A write in one branch of an if is not a kill where the branches join:
// A is carried.
// CHECK-LABEL: storage liveness of @branch (kills=stores): 2 buffers, 7 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-6) first 2 last 6 carried{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(4-5) first 4 last 5{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @branch(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    scf.if %c {
      gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
        memref.store %f, %A[%c0] : memref<4xf32, 1>
        gpu.terminator
      }
    }
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
      %v = memref.load %A[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// A function of the module that writes all of its argument and reads none
// of it kills it at the call; one that reads it uses it there.
// CHECK-LABEL: storage liveness of @callee (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-4) first 1 last 4{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(2-3) first 2 last 3{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @callee(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    func.call @fill(%A) : (memref<4xf32, 1>) -> ()
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @use(%A, %out) : (memref<4xf32, 1>, memref<4xf32, 1>) -> ()
  }
  return
}

// An allocation nothing uses, and one of a dynamic size.
// CHECK-LABEL: storage liveness of @unused (kills=stores): 2 buffers, 3 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-2) first 1 last 2{{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<?xf32, 1> sizes(?) step(none) untouched unused{{$}}
// CHECK-NEXT: s0 interferes in the step with none{{$}}
// CHECK-NEXT: s1 interferes in the step with none{{$}}
func.func @unused(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %D = gpu.alloc (%n) : memref<?xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
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

// An external function that enqueues work on its buffers and returns (a
// transform of cuFFT) uses them until the next wait: A and C interfere
// with B, whose life lies between the call and the wait.
// CHECK-LABEL: storage liveness of @enqueued (kills=stores): 3 buffers, 7 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-6) first 1 last 6 external(@mdrtCudaFFTForward3DF32){{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(3-4) first 3 last 4{{$}}
// CHECK-NEXT: s2 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-6) first 2 last 6 carried external(@mdrtCudaFFTForward3DF32){{$}}
// CHECK-NEXT: s0 interferes in the step with s1 s2{{$}}
// CHECK-NEXT: s1 interferes in the step with s0 s2{{$}}
// CHECK-NEXT: s2 interferes in the step with s0 s1{{$}}
func.func @enqueued(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  %C = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %Av = memref.cast %A : memref<4xf32, 1> to memref<?xf32, 1>
    %Cv = memref.cast %C : memref<4xf32, 1> to memref<?xf32, 1>
    %k = arith.constant 1 : i64
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @mdrtCudaFFTForward3DF32(%Av, %Cv, %k, %k, %k) : (memref<?xf32, 1>, memref<?xf32, 1>, i64, i64, i64) -> ()
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
      %v = memref.load %C[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    %t0 = gpu.wait async
    gpu.wait [%t0]
  }
  return
}

// The same call in a forked region uses its buffers until the join: A
// interferes with B, used on the shared stream before the join.
// CHECK-LABEL: storage liveness of @enqueued_forked (kills=stores): 3 buffers, 7 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(1-5) first 1 last 5 external(@mdrtCudaFFTForward3DF32){{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(3-4) first 3 last 4{{$}}
// CHECK-NEXT: s2 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-6) first 2 last 6 carried external(@mdrtCudaFFTForward3DF32){{$}}
// CHECK-NEXT: s0 interferes in the step with s1 s2{{$}}
// CHECK-NEXT: s1 interferes in the step with s0 s2{{$}}
// CHECK-NEXT: s2 interferes in the step with s0 s1{{$}}
func.func @enqueued_forked(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  %C = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    %Av = memref.cast %A : memref<4xf32, 1> to memref<?xf32, 1>
    %Cv = memref.cast %C : memref<4xf32, 1> to memref<?xf32, 1>
    %k = arith.constant 1 : i64
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @mdrtSideBegin() : () -> ()
    func.call @mdrtCudaFFTForward3DF32(%Av, %Cv, %k, %k, %k) : (memref<?xf32, 1>, memref<?xf32, 1>, i64, i64, i64) -> ()
    func.call @mdrtSideEnd() : () -> ()
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %B[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %B[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @mdrtSideJoin() : () -> ()
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      %v = memref.load %C[%c0] : memref<4xf32, 1>
      memref.store %v, %out[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
  }
  return
}

// An external function the analysis does not know may keep the pointer it
// is given and use it in a later call: A is pinned.
// CHECK-LABEL: storage liveness of @kept (kills=stores): 2 buffers, 5 points in the step
// CHECK-NEXT: s0 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(0-4) first 1 last 2 carried external(@keep_buffer) pinned(given to @keep_buffer, which may keep it){{$}}
// CHECK-NEXT: s1 {{%[a-z_0-9]+}} : memref<4xf32, 1> step(3-4) first 3 last 4{{$}}
// CHECK-NEXT: s0 interferes in the step with s1{{$}}
// CHECK-NEXT: s1 interferes in the step with s0{{$}}
func.func @kept(%out: memref<4xf32, 1>, %n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  %A = gpu.alloc () : memref<4xf32, 1>
  %B = gpu.alloc () : memref<4xf32, 1>
  scf.for %i = %c0 to %n step %c1 {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
      memref.store %f, %A[%c0] : memref<4xf32, 1>
      gpu.terminator
    }
    func.call @keep_buffer(%A) : (memref<4xf32, 1>) -> ()
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

func.func private @mdrtCudaFFTForward3DF32(memref<?xf32, 1>, memref<?xf32, 1>, i64, i64, i64)
func.func private @keep_buffer(memref<4xf32, 1>)
func.func private @fill(%x: memref<4xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = arith.constant 1.0 : f32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %f, %x[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  return
}
func.func private @use(%x: memref<4xf32, 1>, %out: memref<4xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %v = memref.load %x[%c0] : memref<4xf32, 1>
    memref.store %v, %out[%c0] : memref<4xf32, 1>
    gpu.terminator
  }
  return
}
func.func private @keep(index)
func.func private @mdrtSideBegin()
func.func private @mdrtSideEnd()
func.func private @mdrtSideJoin()
