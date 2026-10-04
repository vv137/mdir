// The documented branch conventions (D185), evaluated at
// the singular/tie points, rather than compared with a difference
// that crosses a branch. Each check prints 1.
// RUN: mdir-opt %s %md_passes --convert-md-to-md-exec %md_exec_passes | mlir-opt %lower_loops_to_llvm | mlir-runner -e main --entry-point-result=void --shared-libs=%mdrt,%mlir_c_runner_utils | FileCheck %s
!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
md.potential @abs(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = math.absf %a : f64
  md.return %v : f64
}
md.potential @sqrt(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = math.sqrt %a : f64
  md.return %v : f64
}
md.potential @min(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = arith.minimumf %a, %zero : f64
  md.return %v : f64
}
md.potential @max(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = arith.maximumf %a, %zero : f64
  md.return %v : f64
}
md.potential @select(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %condition = arith.cmpf oge, %a, %zero : f64
  %minus = arith.negf %a : f64
  %v = arith.select %condition, %a, %minus : f64
  md.return %v : f64
}
md.potential @floor(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = math.floor %a : f64
  md.return %v : f64
}
md.potential @ceil(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = math.ceil %a : f64
  md.return %v : f64
}
md.potential @inactive_sqrt(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %zero = arith.constant 0.0 : f64
  %v = math.sqrt %zero : f64
  md.return %v : f64
}
// At r=0 the radial conversion divides by r; it has no geometric
// extension, and the force is nonfinite rather than a proved zero.
md.potential @radial(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.3) : !vec -> !md.relation<@atoms, 2, unordered>
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %r : f64
  } : !md.relation<@atoms, 2, unordered>, !vec -> f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()
func.func @main() {
  %buf = memref.alloc() : memref<2x3xf64>
  %dynamic = memref.cast %buf : memref<2x3xf64> to memref<?x3xf64>
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %z = arith.constant 0.0 : f64
  memref.store %z, %buf[%i0, %i0] : memref<2x3xf64>
  memref.store %z, %buf[%i0, %i1] : memref<2x3xf64>
  memref.store %z, %buf[%i0, %i2] : memref<2x3xf64>
  memref.store %z, %buf[%i1, %i0] : memref<2x3xf64>
  memref.store %z, %buf[%i1, %i1] : memref<2x3xf64>
  memref.store %z, %buf[%i1, %i2] : memref<2x3xf64>
  %x = mdrt.from_buffer %dynamic : memref<?x3xf64> to !vec
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %inf = arith.constant 0x7FF0000000000000 : f64
  %cell = md.orthorhombic_cell %one, %one, %one
  %d_abs = md.evaluate @abs(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_abs = arith.cmpf oeq, %d_abs, %one : f64
  %flag_abs = arith.uitofp %ok_abs : i1 to f64
  call @printF64(%flag_abs) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_sqrt = md.evaluate @sqrt(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_sqrt = arith.cmpf oeq, %d_sqrt, %inf : f64
  %flag_sqrt = arith.uitofp %ok_sqrt : i1 to f64
  call @printF64(%flag_sqrt) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_min = md.evaluate @min(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_min = arith.cmpf oeq, %d_min, %zero : f64
  %flag_min = arith.uitofp %ok_min : i1 to f64
  call @printF64(%flag_min) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_max = md.evaluate @max(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_max = arith.cmpf oeq, %d_max, %zero : f64
  %flag_max = arith.uitofp %ok_max : i1 to f64
  call @printF64(%flag_max) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_select = md.evaluate @select(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_select = arith.cmpf oeq, %d_select, %one : f64
  %flag_select = arith.uitofp %ok_select : i1 to f64
  call @printF64(%flag_select) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_floor = md.evaluate @floor(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_floor = arith.cmpf oeq, %d_floor, %zero : f64
  %flag_floor = arith.uitofp %ok_floor : i1 to f64
  call @printF64(%flag_floor) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_ceil = md.evaluate @ceil(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_ceil = arith.cmpf oeq, %d_ceil, %zero : f64
  %flag_ceil = arith.uitofp %ok_ceil : i1 to f64
  call @printF64(%flag_ceil) : (f64) -> ()
  call @printNewline() : () -> ()
  %d_inactive_sqrt = md.evaluate @inactive_sqrt(%x, %cell, %zero) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  %ok_inactive_sqrt = arith.cmpf oeq, %d_inactive_sqrt, %zero : f64
  %flag_inactive_sqrt = arith.uitofp %ok_inactive_sqrt : i1 to f64
  call @printF64(%flag_inactive_sqrt) : (f64) -> ()
  call @printNewline() : () -> ()
  %force = md.evaluate @radial(%x, %cell) request [forces] : (!vec, !md.cell) -> !vec
  %forces = mdrt.to_buffer %force : !vec to memref<?x3xf64>
  %fx = memref.load %forces[%i0, %i0] : memref<?x3xf64>
  %undefined = arith.cmpf uno, %fx, %fx : f64
  %flag_radial = arith.uitofp %undefined : i1 to f64
  call @printF64(%flag_radial) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
// CHECK: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

// CHECK-NEXT: 1
