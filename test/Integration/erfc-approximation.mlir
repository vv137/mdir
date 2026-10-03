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
// The kernel of a loop over pairs of particles takes y = x² in f32 and
// computes erfc(sqrt(y)), which the pass
// rewrites to exp(-y) P(t): the argument of the exponential is y itself,
// exact, so the error is that of the polynomial, of the square root in t,
// and of the exponential in f32. The reference is erfc(sqrt(y)) in f64, of
// the same y. y runs over [0, 36] in steps of 0.001.
//
// CHECK: largest relative error
// CHECK-NEXT: 1

!x = !md.field<@atoms, 3 x f32>
!y = !md.field<@pairs, f64>
!e = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered, @pairs>
!inc = !mdrt.incidence<@atoms, 2, @pairs>

md.particle_set @atoms
md.tuple_set @pairs on(@atoms) arity(2) orientation(unordered)

func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printString(!llvm.ptr)

llvm.mlir.global internal constant @label("largest relative error\0A\00")

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
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
  // A tuple (2 i, 2 i + 1) for each y, which it takes as its parameter:
  // the pass approximates the kernels of the terms of the potential, not
  // those of loops over particles (D112).
  %c2 = arith.constant 2 : index
  %particles = arith.muli %count, %c2 : index
  %members = memref.alloc(%count) : memref<?x2xi32>
  %parameters = memref.alloc(%count) : memref<?xf64>
  %places = memref.alloc(%particles) : memref<?x3xf32>
  %origin = arith.constant 0.0 : f32
  scf.for %i = %c0 to %count step %c1 {
    %first = arith.muli %i, %c2 : index
    %second = arith.addi %first, %c1 : index
    %first32 = arith.index_cast %first : index to i32
    %second32 = arith.index_cast %second : index to i32
    memref.store %first32, %members[%i, %c0] : memref<?x2xi32>
    memref.store %second32, %members[%i, %c1] : memref<?x2xi32>
    %y32 = memref.load %ys[%i] : memref<?xf32>
    %y64 = arith.extf %y32 : f32 to f64
    memref.store %y64, %parameters[%i] : memref<?xf64>
  }
  scf.for %p = %c0 to %particles step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      memref.store %origin, %places[%p, %k] : memref<?x3xf32>
    }
  }
  %positions = mdrt.from_buffer %places : memref<?x3xf32> to !x
  %pairs = mdrt.from_buffer %members : memref<?x2xi32> to !pairs
  %inc = md_exec.build_incidence %pairs : !pairs -> !inc
  %y = mdrt.from_buffer %parameters : memref<?xf64> to !y
  %edge = arith.constant 10.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %d = md_exec.zeros : !e
  %approximate = md_exec.tuple_for %inc, %positions, %cell
      coordinates(displacement(1, 0)) tuple(%y : !y) outs(%d : !e) arity(2) {
  ^bb0(%r: vector<3xf32>, %yt: f64):
    %yi = arith.truncf %yt : f64 to f32
    %root = math.sqrt %yi : f32
    %one = arith.constant 1.0 : f32
    %x0 = arith.mulf %root, %one : f32
    %erfc = math.erfc %x0 : f32
    %wide = arith.extf %erfc : f32 to f64
    %none = arith.constant 0.0 : f64
    md_exec.yield %wide, %none : f64, f64
  } : !inc, !x -> !e
  %es = mdrt.to_buffer %approximate : !e to memref<?xf64>

  %zero = arith.constant 0.0 : f64
  %largest = scf.for %i = %c0 to %count step %c1 iter_args(%m = %zero)
      -> (f64) {
    %y32 = memref.load %ys[%i] : memref<?xf32>
    %y64 = arith.extf %y32 : f32 to f64
    %x = math.sqrt %y64 : f64
    %reference = math.erfc %x : f64
    %place = arith.muli %i, %c2 : index
    %value = memref.load %es[%place] : memref<?xf64>
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
