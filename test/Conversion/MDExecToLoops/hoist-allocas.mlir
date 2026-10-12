// RUN: mdir-opt --hoist-static-allocas %s | FileCheck %s
// RUN: not mdir-opt --hoist-static-allocas=refuse-rest %s 2>&1 | FileCheck %s --check-prefix=REFUSED

// The variable of a reduction that `convert-scf-to-openmp` allocates in a
// loop moves to the entry block, once for every iteration (D117); an
// alloca inside a region of OpenMP, one for each thread, and one of a size
// that is not a constant stay.

// With `refuse-rest`, as the lowering of a program of frames runs the pass
// where it ends (#309), the alloca that stays in the loop is an error that
// names it: a module that the pass accepts has no allocation of the stack
// outside the first block of a function, apart from those of the threads of
// OpenMP, and one that it cannot prove so does not compile
// (hoist-allocas-refuse-rest.mlir has the module that it accepts).

// REFUSED: error: 'llvm.alloca' op takes memory of the stack of a size that is not a constant outside the first block of its function
// REFUSED-NOT: error:

// CHECK-LABEL: llvm.func @steps
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[SUM:.*]] = llvm.alloca %[[ONE]] x f64
// CHECK:         llvm.br ^[[LOOP:.*]](
// CHECK:       ^[[LOOP]](
// CHECK-NOT:     llvm.alloca %{{.*}} x f64
// CHECK:         llvm.store %{{.*}}, %[[SUM]]
// CHECK:         omp.parallel
// CHECK:           llvm.alloca %{{.*}} x i32
// CHECK:         llvm.alloca %arg0 x i64
llvm.func @steps(%n: i64) {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64
  llvm.br ^loop(%zero : i64)
^loop(%i: i64):
  %one = llvm.mlir.constant(1 : i64) : i64
  %sum = llvm.alloca %one x f64 : (i64) -> !llvm.ptr
  %init = llvm.mlir.constant(0.0 : f64) : f64
  llvm.store %init, %sum : f64, !llvm.ptr
  omp.parallel {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %private = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
    omp.terminator
  }
  %dynamic = llvm.alloca %n x i64 : (i64) -> !llvm.ptr
  %next = llvm.add %i, %step : i64
  %more = llvm.icmp "slt" %next, %n : i64
  llvm.cond_br %more, ^loop(%next : i64), ^done
^done:
  llvm.return
}
