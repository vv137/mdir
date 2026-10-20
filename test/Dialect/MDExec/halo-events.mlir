// RUN: mdir-opt %s | FileCheck %s --check-prefix=EVENT
// RUN: mdir-opt %s --lower-mdrt-halo-events | FileCheck %s --check-prefix=SYNC

module {
  func.func @copy(%halo: i64, %image: i64) {
    %source = memref.alloc() : memref<2x2x2xf64>
    %destination = memref.alloc() : memref<2x2x2xf64>
    %event = mdrt.halo_copy_start %halo, %source, %destination, %image
        : memref<2x2x2xf64>, memref<2x2x2xf64>
    %independent = arith.constant 1 : index
    %ready = mdrt.halo_wait %event, %halo, %source, %destination
        : memref<2x2x2xf64>, memref<2x2x2xf64> -> memref<2x2x2xf64>
    %value = memref.load %ready[%independent, %independent, %independent]
        : memref<2x2x2xf64>
    memref.dealloc %destination : memref<2x2x2xf64>
    memref.dealloc %source : memref<2x2x2xf64>
    return
  }
}

// EVENT: mdrt.halo_copy_start
// EVENT: mdrt.halo_wait
// SYNC: call @mdrtDistHaloCopy
// SYNC-NOT: mdrt.halo_wait
