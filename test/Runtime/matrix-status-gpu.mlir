// REQUIRES: cuda
// RUN: mlir-opt %s %lower_gpu_to_llvm > %t.mlir
// RUN: mlir-runner %t.mlir -e growth --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda | FileCheck %s
// RUN: not mlir-runner %t.mlir -e invalid --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda 2>&1 | FileCheck %s --check-prefix=INVALID

func.func private @mdrtMatrixCreate(i64, i64) -> i64
func.func private @mdrtMatrixSpill(i64) -> memref<?x?xi32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtMatrixEntries(i64) -> memref<?x?xi32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtMatrixPublish(i64) -> i32
func.func private @mdrtMatrixPrepare(i64)
func.func private @mdrtDeviceWaitAll()
func.func private @mdrtDeviceActivationLeave()
func.func private @printI64(i64)
func.func private @printNewline()

func.func @publish(%handle: i64, %width: i32, %bad: i32) {
  %buffer = func.call @mdrtMatrixSpill(%handle) : (i64) -> memref<?x?xi32, 1>
  %one = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
             threads(%tx, %ty, %tz) in (%dx = %one, %dy = %one, %dz = %one) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0 : i32
    memref.store %zero, %buffer[%c0, %c0] : memref<?x?xi32, 1>
    memref.store %zero, %buffer[%c0, %c1] : memref<?x?xi32, 1>
    memref.store %width, %buffer[%c0, %c2] : memref<?x?xi32, 1>
    memref.store %bad, %buffer[%c0, %c3] : memref<?x?xi32, 1>
    gpu.terminator
  }
  %retry = func.call @mdrtMatrixPublish(%handle) : (i64) -> i32
  return
}

func.func @show_width(%handle: i64) {
  %matrix = func.call @mdrtMatrixEntries(%handle) : (i64) -> memref<?x?xi32, 1>
  %one = arith.constant 1 : index
  %width = memref.dim %matrix, %one : memref<?x?xi32, 1>
  %wide = arith.index_cast %width : index to i64
  func.call @printI64(%wide) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}

// The first publish sizes the matrix. Reading status cannot relocate
// storage that a pair kernel is still using. The next prepare uses the maximum.
// CHECK:      {{^}}8{{$}}
// CHECK-NEXT: {{^}}55{{$}}
func.func @growth() {
  %n = arith.constant 64 : i64
  %width = arith.constant 8 : i64
  %large = arith.constant 44 : i32
  %small = arith.constant 10 : i32
  %good = arith.constant 0 : i32
  %handle = func.call @mdrtMatrixCreate(%n, %width) : (i64, i64) -> i64
  func.call @publish(%handle, %large, %good) : (i64, i32, i32) -> ()
  func.call @publish(%handle, %small, %good) : (i64, i32, i32) -> ()
  func.call @mdrtDeviceWaitAll() : () -> ()
  func.call @show_width(%handle) : (i64) -> ()
  func.call @mdrtMatrixPrepare(%handle) : (i64) -> ()
  func.call @show_width(%handle) : (i64) -> ()
  return
}

// The activation hands results out immediately after the last build.
// INVALID: a position is not a number or is far outside the cell
// INVALID-NOT: 123456
func.func @invalid() {
  %n = arith.constant 64 : i64
  %width = arith.constant 8 : i64
  %required = arith.constant 4 : i32
  %bad = arith.constant 1 : i32
  %sentinel = arith.constant 123456 : i64
  %handle = func.call @mdrtMatrixCreate(%n, %width) : (i64, i64) -> i64
  func.call @publish(%handle, %required, %bad) : (i64, i32, i32) -> ()
  func.call @mdrtDeviceActivationLeave() : () -> ()
  func.call @printI64(%sentinel) : (i64) -> ()
  return
}
