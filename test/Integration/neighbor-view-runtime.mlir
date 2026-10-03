// Duplicate entries are rejected before a force kernel can double count them.
// RUN: mdir-opt %s --convert-md-exec-to-loops --convert-scf-to-cf \
// RUN:   --expand-strided-metadata --finalize-memref-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts > %t
// RUN: not --crash stdbuf -o0 mlir-runner %t -e main --entry-point-result=void 2>&1 | FileCheck %s
md.particle_set @atoms
func.func @main() {
  %z = arith.constant 0 : index
  %o = arith.constant 1 : index
  %two = arith.constant 2 : i32
  %neighbor = arith.constant 1 : i32
  %local = arith.constant 3 : index
  %counts = memref.alloc() : memref<1xi32>
  %entries = memref.alloc() : memref<1x2xi32>
  memref.store %two, %counts[%z] : memref<1xi32>
  memref.store %neighbor, %entries[%z, %z] : memref<1x2xi32>
  memref.store %neighbor, %entries[%z, %o] : memref<1x2xi32>
  %nl = md_exec.neighbor_view %counts, %entries local_size(%local)
      : memref<1xi32>, memref<1x2xi32> -> !mdrt.neighbors<@atoms>
  memref.dealloc %entries : memref<1x2xi32>
  memref.dealloc %counts : memref<1xi32>
  return
}
// CHECK: neighbor indices must be sorted, unique, and nonnegative
