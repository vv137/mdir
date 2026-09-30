// The approximation of erfc that md-exec-approximate writes into f32
// kernels under fast_math, against erfc in f64, on the host: the relative
// error on [0, 6] is below 5e-7.
//
// RUN: mdir-opt %s --md-exec-approximate --md-exec-assign-storage \
// RUN:     --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The kernel takes y = x² in f32 and computes erfc(sqrt(y)), which the pass
// rewrites to exp(-y) P(t): the argument of the exponential is y itself,
// exact, so the error is that of the polynomial, of the square root in t,
// and of the exponential in f32. The reference is erfc(sqrt(y)) in f64, of
// the same y. y runs over [0, 36] in steps of 0.001.
//
// CHECK: largest relative error
// CHECK-NEXT: 1

!y = !md.field<@atoms, f32>
!e = !md.field<@atoms, f64>

md.particle_set @atoms

func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printString(!llvm.ptr)

llvm.mlir.global internal constant @label("largest relative error\0A\00")

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 36001 : index
  %step = arith.constant 0.001 : f64
  %ys = memref.alloc(%count) : memref<?xf32>
  scf.for %i = %c0 to %count step %c1 {
    %i64 = arith.index_cast %i : index to i64
    %if = arith.sitofp %i64 : i64 to f64
    %y64 = arith.mulf %if, %step : f64
    %y32 = arith.truncf %y64 : f64 to f32
    memref.store %y32, %ys[%i] : memref<?xf32>
  }
  %y = mdrt.from_buffer %ys : memref<?xf32> to !y

  %d = md_exec.empty : !e
  %approximate = md_exec.particle_for ins(%y : !y) outs(%d : !e) {
  ^bb0(%yi: f32):
    %root = math.sqrt %yi : f32
    %one = arith.constant 1.0 : f32
    %x = arith.mulf %root, %one : f32
    %erfc = math.erfc %x : f32
    %wide = arith.extf %erfc : f32 to f64
    md_exec.yield %wide : f64
  } -> !e
  %es = mdrt.to_buffer %approximate : !e to memref<?xf64>

  %zero = arith.constant 0.0 : f64
  %largest = scf.for %i = %c0 to %count step %c1 iter_args(%m = %zero)
      -> (f64) {
    %y32 = memref.load %ys[%i] : memref<?xf32>
    %y64 = arith.extf %y32 : f32 to f64
    %x = math.sqrt %y64 : f64
    %reference = math.erfc %x : f64
    %value = memref.load %es[%i] : memref<?xf64>
    %difference = arith.subf %value, %reference : f64
    %absolute = math.absf %difference : f64
    %relative = arith.divf %absolute, %reference : f64
    %next = arith.maximumf %m, %relative : f64
    scf.yield %next : f64
  }
  %label = llvm.mlir.addressof @label : !llvm.ptr
  call @printString(%label) : (!llvm.ptr) -> ()
  %bound = arith.constant 5.0e-7 : f64
  %below = arith.cmpf olt, %largest, %bound : f64
  %flag = arith.extui %below : i1 to i64
  call @printI64(%flag) : (i64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%largest) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
