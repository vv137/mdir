// RUN: mdir-opt --hoist-static-allocas=refuse-rest %s | FileCheck %s

// What `hoist-static-allocas` accepts with `refuse-rest`, as the lowering of
// a program of frames runs it where it ends (#309): in a loop that never
// ends an alloca of a constant size, which moves to the first block, and one
// for each thread inside a region of OpenMP, which stays; and in the first
// block an alloca of a size that is not a constant, which is taken once.
// hoist-allocas.mlir has the module that it refuses.

// CHECK-LABEL: llvm.func @frames
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[ROW:.*]] = llvm.alloca %[[ONE]] x f64
// CHECK:         llvm.alloca %arg0 x i64
// CHECK:         llvm.br ^[[LOOP:.*]](
// CHECK:       ^[[LOOP]](
// CHECK-NOT:     llvm.alloca %{{.*}} x f64
// CHECK:         llvm.store %{{.*}}, %[[ROW]]
// CHECK:         omp.parallel
// CHECK:           llvm.alloca %{{.*}} x i32
llvm.func @frames(%n: i64) {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64
  %once = llvm.alloca %n x i64 : (i64) -> !llvm.ptr
  llvm.br ^loop(%zero : i64)
^loop(%i: i64):
  %one = llvm.mlir.constant(1 : i64) : i64
  %row = llvm.alloca %one x f64 : (i64) -> !llvm.ptr
  %init = llvm.mlir.constant(0.0 : f64) : f64
  llvm.store %init, %row : f64, !llvm.ptr
  omp.parallel {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %private = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
    omp.terminator
  }
  %next = llvm.add %i, %step : i64
  llvm.br ^loop(%next : i64)
}
