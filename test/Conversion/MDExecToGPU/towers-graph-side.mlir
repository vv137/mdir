// RUN: mdir-opt %s --mdir-towers-step-graph -split-input-file -verify-diagnostics | FileCheck %s

// D449: the side streams of term_streams in the plain TOWERS
// step graph (D439). A minimal ordinary step: the drift check
// (mdrtFlagStart), a PME stage, the force publication (towers_finish), the
// certificate (mdrtFlagFinish) and the two step counters. The capture
// splits at the drift check, so a side-stream window must open and close
// within one phase, and a fork or join must not hide inside a region.

// A window within the force phase is captured: the fork, the side work and
// the join follow the capture of the force phase.
// CHECK-LABEL: func.func @mdir_run(
// CHECK:         call @mdrtStepDrift(
// CHECK:         call @mdrtSideBegin()
// CHECK:         call @mdrtSideEnd()
// CHECK:         call @mdrtSideJoin()
// CHECK:         call @mdrtStepEnd(
module attributes {gpu.container_module, mdir.towers_grid = array<i64: 4, 1, 1, 1, 1, 1, 1>, mdir.towers_dt = 0.002 : f64} {
  gpu.module @towers_site {
    gpu.func @towers_finish_dynamic(%a: memref<?xf32, 1>, %b: memref<?xf32, 1>, %c: memref<?xf32, 1>, %d: memref<?xf32, 1>, %e: memref<?xf32, 1>, %cert: memref<?xi32, 1>, %out: memref<?x3xf32, 1>) kernel {
      llvm.inline_asm has_side_effects "trap;", "" : () -> ()
      gpu.return
    }
  }
  func.func private @mdrtFlagStart(!llvm.ptr) -> i64
  func.func private @mdrtFlagFinish(i64) -> i32
  func.func private @mdrtStopConstraintFailure(i64, i64)
  func.func private @mdrt_gpu_pme_weights_p(memref<?x3xf64, 1>)
  func.func private @mdrtSideBegin()
  func.func private @mdrtSideEnd()
  func.func private @mdrtSideJoin()
  func.func @mdir_run(%n: index, %flag: memref<1xi32>, %steps: memref<1xi64>, %builds: memref<1xi64>,
                      %x0: memref<?x3xf64, 1>, %x1: memref<?x3xf64, 1>, %f0: memref<?x3xf32, 1>,
                      %s: memref<?xf32, 1>, %cert: memref<?xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    memref.store %zero, %steps[%c0] : memref<1xi64>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %x0, %y = %x1) -> (memref<?x3xf64, 1>, memref<?x3xf64, 1>) {
      %p = memref.extract_aligned_pointer_as_index %flag : memref<1xi32> -> index
      %pi = arith.index_cast %p : index to i64
      %ptr = llvm.inttoptr %pi : i64 to !llvm.ptr
      %t = func.call @mdrtFlagStart(%ptr) : (!llvm.ptr) -> i64
      %bad = arith.cmpi ne, %t, %zero : i64
      scf.if %bad {
        %set = arith.constant 0 : i64
        func.call @mdrtStopConstraintFailure(%zero, %set) : (i64, i64) -> ()
      }
      func.call @mdrt_gpu_pme_weights_p(%y) : (memref<?x3xf64, 1>) -> ()
      gpu.launch_func @towers_site::@towers_finish_dynamic blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %cert : memref<?xi32, 1>, %f0 : memref<?x3xf32, 1>)
      func.call @mdrtSideBegin() : () -> ()
      func.call @mdrt_gpu_pme_weights_p(%x) : (memref<?x3xf64, 1>) -> ()
      func.call @mdrtSideEnd() : () -> ()
      func.call @mdrtSideJoin() : () -> ()
      %u = func.call @mdrtFlagFinish(%t) : (i64) -> i32
      %a = memref.load %steps[%c0] : memref<1xi64>
      %a1 = arith.addi %a, %one : i64
      memref.store %a1, %steps[%c0] : memref<1xi64>
      %b = memref.load %builds[%c0] : memref<1xi64>
      %b1 = arith.addi %b, %one : i64
      memref.store %b1, %builds[%c0] : memref<1xi64>
      scf.yield %y, %x : memref<?x3xf64, 1>, memref<?x3xf64, 1>
    }
    return
  }
}

// -----

// A window opened in the drift phase and joined in the force phase.
module attributes {gpu.container_module, mdir.towers_grid = array<i64: 4, 1, 1, 1, 1, 1, 1>, mdir.towers_dt = 0.002 : f64} {
  gpu.module @towers_site {
    gpu.func @towers_finish_dynamic(%a: memref<?xf32, 1>, %b: memref<?xf32, 1>, %c: memref<?xf32, 1>, %d: memref<?xf32, 1>, %e: memref<?xf32, 1>, %cert: memref<?xi32, 1>, %out: memref<?x3xf32, 1>) kernel {
      llvm.inline_asm has_side_effects "trap;", "" : () -> ()
      gpu.return
    }
  }
  func.func private @mdrtFlagStart(!llvm.ptr) -> i64
  func.func private @mdrtFlagFinish(i64) -> i32
  func.func private @mdrtStopConstraintFailure(i64, i64)
  func.func private @mdrt_gpu_pme_weights_p(memref<?x3xf64, 1>)
  func.func private @mdrtSideBegin()
  func.func private @mdrtSideEnd()
  func.func private @mdrtSideJoin()
  func.func @mdir_run(%n: index, %flag: memref<1xi32>, %steps: memref<1xi64>, %builds: memref<1xi64>,
                      %x0: memref<?x3xf64, 1>, %x1: memref<?x3xf64, 1>, %f0: memref<?x3xf32, 1>,
                      %s: memref<?xf32, 1>, %cert: memref<?xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    memref.store %zero, %steps[%c0] : memref<1xi64>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %x0, %y = %x1) -> (memref<?x3xf64, 1>, memref<?x3xf64, 1>) {
      %p = memref.extract_aligned_pointer_as_index %flag : memref<1xi32> -> index
      %pi = arith.index_cast %p : index to i64
      %ptr = llvm.inttoptr %pi : i64 to !llvm.ptr
      func.call @mdrtSideBegin() : () -> ()
      func.call @mdrt_gpu_pme_weights_p(%x) : (memref<?x3xf64, 1>) -> ()
      func.call @mdrtSideEnd() : () -> ()
      // expected-error @+1 {{step_graph: a side-stream window crosses the end of the drift phase}}
      %t = func.call @mdrtFlagStart(%ptr) : (!llvm.ptr) -> i64
      %bad = arith.cmpi ne, %t, %zero : i64
      scf.if %bad {
        %set = arith.constant 0 : i64
        func.call @mdrtStopConstraintFailure(%zero, %set) : (i64, i64) -> ()
      }
      func.call @mdrt_gpu_pme_weights_p(%y) : (memref<?x3xf64, 1>) -> ()
      gpu.launch_func @towers_site::@towers_finish_dynamic blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %cert : memref<?xi32, 1>, %f0 : memref<?x3xf32, 1>)
      func.call @mdrtSideJoin() : () -> ()
      %u = func.call @mdrtFlagFinish(%t) : (i64) -> i32
      %a = memref.load %steps[%c0] : memref<1xi64>
      %a1 = arith.addi %a, %one : i64
      memref.store %a1, %steps[%c0] : memref<1xi64>
      %b = memref.load %builds[%c0] : memref<1xi64>
      %b1 = arith.addi %b, %one : i64
      memref.store %b1, %builds[%c0] : memref<1xi64>
      scf.yield %y, %x : memref<?x3xf64, 1>, memref<?x3xf64, 1>
    }
    return
  }
}

// -----

// A window that the step never joins.
module attributes {gpu.container_module, mdir.towers_grid = array<i64: 4, 1, 1, 1, 1, 1, 1>, mdir.towers_dt = 0.002 : f64} {
  gpu.module @towers_site {
    gpu.func @towers_finish_dynamic(%a: memref<?xf32, 1>, %b: memref<?xf32, 1>, %c: memref<?xf32, 1>, %d: memref<?xf32, 1>, %e: memref<?xf32, 1>, %cert: memref<?xi32, 1>, %out: memref<?x3xf32, 1>) kernel {
      llvm.inline_asm has_side_effects "trap;", "" : () -> ()
      gpu.return
    }
  }
  func.func private @mdrtFlagStart(!llvm.ptr) -> i64
  func.func private @mdrtFlagFinish(i64) -> i32
  func.func private @mdrtStopConstraintFailure(i64, i64)
  func.func private @mdrt_gpu_pme_weights_p(memref<?x3xf64, 1>)
  func.func private @mdrtSideBegin()
  func.func private @mdrtSideEnd()
  func.func private @mdrtSideJoin()
  func.func @mdir_run(%n: index, %flag: memref<1xi32>, %steps: memref<1xi64>, %builds: memref<1xi64>,
                      %x0: memref<?x3xf64, 1>, %x1: memref<?x3xf64, 1>, %f0: memref<?x3xf32, 1>,
                      %s: memref<?xf32, 1>, %cert: memref<?xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    memref.store %zero, %steps[%c0] : memref<1xi64>
    // expected-error @+1 {{step_graph: a side-stream window is not joined within the step}}
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %x0, %y = %x1) -> (memref<?x3xf64, 1>, memref<?x3xf64, 1>) {
      %p = memref.extract_aligned_pointer_as_index %flag : memref<1xi32> -> index
      %pi = arith.index_cast %p : index to i64
      %ptr = llvm.inttoptr %pi : i64 to !llvm.ptr
      %t = func.call @mdrtFlagStart(%ptr) : (!llvm.ptr) -> i64
      %bad = arith.cmpi ne, %t, %zero : i64
      scf.if %bad {
        %set = arith.constant 0 : i64
        func.call @mdrtStopConstraintFailure(%zero, %set) : (i64, i64) -> ()
      }
      func.call @mdrt_gpu_pme_weights_p(%y) : (memref<?x3xf64, 1>) -> ()
      gpu.launch_func @towers_site::@towers_finish_dynamic blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %cert : memref<?xi32, 1>, %f0 : memref<?x3xf32, 1>)
      func.call @mdrtSideBegin() : () -> ()
      func.call @mdrt_gpu_pme_weights_p(%x) : (memref<?x3xf64, 1>) -> ()
      func.call @mdrtSideEnd() : () -> ()
      %u = func.call @mdrtFlagFinish(%t) : (i64) -> i32
      %a = memref.load %steps[%c0] : memref<1xi64>
      %a1 = arith.addi %a, %one : i64
      memref.store %a1, %steps[%c0] : memref<1xi64>
      %b = memref.load %builds[%c0] : memref<1xi64>
      %b1 = arith.addi %b, %one : i64
      memref.store %b1, %builds[%c0] : memref<1xi64>
      scf.yield %y, %x : memref<?x3xf64, 1>, memref<?x3xf64, 1>
    }
    return
  }
}

// -----

// A fork inside a region, which the phase check would not see.
module attributes {gpu.container_module, mdir.towers_grid = array<i64: 4, 1, 1, 1, 1, 1, 1>, mdir.towers_dt = 0.002 : f64} {
  gpu.module @towers_site {
    gpu.func @towers_finish_dynamic(%a: memref<?xf32, 1>, %b: memref<?xf32, 1>, %c: memref<?xf32, 1>, %d: memref<?xf32, 1>, %e: memref<?xf32, 1>, %cert: memref<?xi32, 1>, %out: memref<?x3xf32, 1>) kernel {
      llvm.inline_asm has_side_effects "trap;", "" : () -> ()
      gpu.return
    }
  }
  func.func private @mdrtFlagStart(!llvm.ptr) -> i64
  func.func private @mdrtFlagFinish(i64) -> i32
  func.func private @mdrtStopConstraintFailure(i64, i64)
  func.func private @mdrt_gpu_pme_weights_p(memref<?x3xf64, 1>)
  func.func private @mdrtSideBegin()
  func.func private @mdrtSideEnd()
  func.func private @mdrtSideJoin()
  func.func @mdir_run(%n: index, %flag: memref<1xi32>, %steps: memref<1xi64>, %builds: memref<1xi64>,
                      %x0: memref<?x3xf64, 1>, %x1: memref<?x3xf64, 1>, %f0: memref<?x3xf32, 1>,
                      %s: memref<?xf32, 1>, %cert: memref<?xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    memref.store %zero, %steps[%c0] : memref<1xi64>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %x0, %y = %x1) -> (memref<?x3xf64, 1>, memref<?x3xf64, 1>) {
      %p = memref.extract_aligned_pointer_as_index %flag : memref<1xi32> -> index
      %pi = arith.index_cast %p : index to i64
      %ptr = llvm.inttoptr %pi : i64 to !llvm.ptr
      %t = func.call @mdrtFlagStart(%ptr) : (!llvm.ptr) -> i64
      %bad = arith.cmpi ne, %t, %zero : i64
      scf.if %bad {
        %set = arith.constant 0 : i64
        func.call @mdrtStopConstraintFailure(%zero, %set) : (i64, i64) -> ()
      }
      func.call @mdrt_gpu_pme_weights_p(%y) : (memref<?x3xf64, 1>) -> ()
      gpu.launch_func @towers_site::@towers_finish_dynamic blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %s : memref<?xf32, 1>, %cert : memref<?xi32, 1>, %f0 : memref<?x3xf32, 1>)
      %go = arith.cmpi eq, %t, %zero : i64
      // expected-error @+1 {{step_graph: a side-stream call inside a region of the plain TOWERS step}}
      scf.if %go {
        func.call @mdrtSideBegin() : () -> ()
      }
      func.call @mdrtSideEnd() : () -> ()
      func.call @mdrtSideJoin() : () -> ()
      %u = func.call @mdrtFlagFinish(%t) : (i64) -> i32
      %a = memref.load %steps[%c0] : memref<1xi64>
      %a1 = arith.addi %a, %one : i64
      memref.store %a1, %steps[%c0] : memref<1xi64>
      %b = memref.load %builds[%c0] : memref<1xi64>
      %b1 = arith.addi %b, %one : i64
      memref.store %b1, %builds[%c0] : memref<1xi64>
      scf.yield %y, %x : memref<?x3xf64, 1>, memref<?x3xf64, 1>
    }
    return
  }
}
