// RUN: mdir-opt %s --check-read-only-arguments -split-input-file -verify-diagnostics

// A buffer that a function declares read-only and that its code may write,
// or that the pass cannot prove it does not, fails the compile with an error
// that names the argument and the op (D258).

func.func @store(%field: memref<?xf32> {mdrt.read_only}, %x: f32) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{the argument 0 of 'store' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  memref.store %x, %field[%c0] : memref<?xf32>
  return
}

// -----

// Written through a view.
func.func @view(%table: memref<?x?xf64> {mdrt.read_only}, %x: f64) {
  %c0 = arith.constant 0 : index
  %flat = memref.cast %table : memref<?x?xf64> to memref<?x?xf64>
  // expected-error @below {{the argument 0 of 'view' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  memref.store %x, %flat[%c0, %c0] : memref<?x?xf64>
  return
}

// -----

// The target of a copy, from the host or from a device.
func.func @copy(%from: memref<?xf32>, %field: memref<?xf32> {mdrt.read_only}) {
  // expected-error @below {{the argument 1 of 'copy' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  memref.copy %from, %field : memref<?xf32> to memref<?xf32>
  return
}

// -----

func.func @back(%device: memref<?xf32, 1>, %field: memref<?xf32> {mdrt.read_only}) {
  // expected-error @below {{the argument 1 of 'back' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  gpu.memcpy %field, %device : memref<?xf32>, memref<?xf32, 1>
  return
}

// -----

// Released.
func.func @release(%field: memref<?xf32> {mdrt.read_only}) {
  // expected-error @below {{the argument 0 of 'release' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  memref.dealloc %field : memref<?xf32>
  return
}

// -----

// Passed to a function of the runtime that does not declare that it only
// reads it.
func.func private @mdrtWrites(memref<?x?xi32>)

func.func @runtime(%members: memref<?x?xi32> {mdrt.read_only}) {
  // expected-error @below {{the argument 0 of 'runtime' is declared read-only ('mdrt.read_only'), but it is passed to 'mdrtWrites', which does not declare its parameter 0 read-only}}
  func.call @mdrtWrites(%members) : (memref<?x?xi32>) -> ()
  return
}

// -----

// Written by a function of the module that it is passed to.
func.func @writes(%buffer: memref<?xf32>, %x: f32) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{the argument 0 of 'callee' is declared read-only ('mdrt.read_only'), but it may be written or released here}}
  memref.store %x, %buffer[%c0] : memref<?xf32>
  return
}

func.func @callee(%field: memref<?xf32> {mdrt.read_only}, %x: f32) {
  func.call @writes(%field, %x) : (memref<?xf32>, f32) -> ()
  return
}

// -----

// Carried by a loop: the pass does not follow it, and says so.
func.func @carried(%field: memref<?xf32> {mdrt.read_only}, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{the argument 0 of 'carried' is declared read-only ('mdrt.read_only'), but the op takes it and its memory effects are not known}}
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%b = %field) -> (memref<?xf32>) {
    scf.yield %b : memref<?xf32>
  }
  return
}

// -----

// Returned to the caller.
func.func @returned(%field: memref<?xf32> {mdrt.read_only}) -> memref<?xf32> {
  // expected-error @below {{the argument 0 of 'returned' is declared read-only ('mdrt.read_only'), but the op takes it without saying how}}
  return %field : memref<?xf32>
}

// -----

// Its address taken out of it.
func.func @address(%field: memref<?xf32> {mdrt.read_only}) -> index {
  // expected-error @below {{the argument 0 of 'address' is declared read-only ('mdrt.read_only'), but the op takes it without saying how}}
  %p = memref.extract_aligned_pointer_as_index %field : memref<?xf32> -> index
  return %p : index
}

// -----

// expected-error @below {{declares its argument 0 read-only, which is not a buffer}}
func.func @scalar(%x: f64 {mdrt.read_only}) {
  return
}
