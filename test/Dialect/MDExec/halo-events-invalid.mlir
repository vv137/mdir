// RUN: mdir-opt %s -split-input-file -verify-diagnostics

func.func @missing_wait(%halo: i64, %image: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  // expected-error@+1 {{event must have exactly one use by mdrt.halo_wait}}
  %event = mdrt.halo_copy_start %halo, %source, %destination, %image
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  return
}

// -----

func.func @early_read(%halo: i64, %image: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  %c0 = arith.constant 0 : index
  %event = mdrt.halo_copy_start %halo, %source, %destination, %image
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  // expected-error@+1 {{uses a halo transfer buffer or handle before completion}}
  %early = memref.load %destination[%c0, %c0, %c0] : memref<2x2x2xf64>
  %ready = mdrt.halo_wait %event, %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  return
}

// -----

func.func @early_free(%halo: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  %event = mdrt.halo_sum_start %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  // expected-error@+1 {{uses a halo transfer buffer or handle before completion}}
  memref.dealloc %source : memref<2x2x2xf64>
  %ready = mdrt.halo_wait %event, %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  return
}

// -----

func.func @wrong_handle(%halo: i64, %other: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  %event = mdrt.halo_sum_start %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  // expected-error@+1 {{halo handle and transfer buffers do not match the start}}
  %ready = mdrt.halo_wait %event, %other, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  return
}

// -----

func.func @unproved_alias(%halo: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  // expected-error@+1 {{creates an unproved alias of a halo transfer buffer}}
  %view = memref.subview %destination[0, 0, 0] [2, 2, 2] [1, 1, 1]
      : memref<2x2x2xf64> to memref<2x2x2xf64, strided<[4, 2, 1]>>
  %event = mdrt.halo_sum_start %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  %ready = mdrt.halo_wait %event, %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  return
}

// -----

func.func @freed_source(%halo: i64) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  // expected-error@+1 {{halo transfer buffer is deallocated before completion}}
  memref.dealloc %source : memref<2x2x2xf64>
  %event = mdrt.halo_sum_start %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  %ready = mdrt.halo_wait %event, %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  return
}

// -----

func.func @cross_block_wait(%halo: i64, %condition: i1) {
  %source = memref.alloc() : memref<2x2x2xf64>
  %destination = memref.alloc() : memref<2x2x2xf64>
  // expected-error@+1 {{event must be consumed by a later mdrt.halo_wait in the same block}}
  %event = mdrt.halo_sum_start %halo, %source, %destination
      : memref<2x2x2xf64>, memref<2x2x2xf64>
  scf.if %condition {
    %ready = mdrt.halo_wait %event, %halo, %source, %destination
        : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
  }
  return
}
