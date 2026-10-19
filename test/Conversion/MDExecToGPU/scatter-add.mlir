// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter=scatter-hash=on \
// RUN:     --convert-md-exec-to-gpu | FileCheck %s --check-prefix=TABLE
// RUN: mdir-opt %s --md-exec-choose-scatter --convert-md-exec-to-gpu \
// RUN: | FileCheck %s --check-prefix=LEAST
// RUN: not mdir-opt %s --md-exec-choose-scatter=scatter-hash=on \
// RUN:     --convert-md-exec-to-gpu=deterministic=true 2>&1 \
// RUN: | FileCheck %s --check-prefix=DET

// The lowering of md_exec.scatter_add for a device (D291).
//
// Without a strategy: a destination that is overwritten is cleared, then a
// thread a value adds it with an atomic of the device.
// CHECK-LABEL: func.func @add(
// CHECK:         gpu.launch
// CHECK:           memref.store %{{.*}}, %arg2[
// CHECK:         gpu.launch
// CHECK:           memref.load %arg1[
// CHECK-COUNT-3:   "red.relaxed.gpu.global.add.f32 [$0], $1;"
// CHECK-NOT:     atom.shared.cas
// CHECK:         return
//
// With `hash(128, 8, 0)`: a table of 128 slots for 128 threads of one
// value each, its keys and the rows of 3 components, cleared, filled
// through compare-and-swap with the device atomic as the fallback, and
// flushed after a barrier.
// TABLE-LABEL: func.func @add(
// TABLE:         gpu.launch {{.*}} workgroup(%[[KEYS:[a-z0-9]+]] : memref<128xi32, #gpu.address_space<workgroup>>, %[[SUMS:[a-z0-9]+]] : memref<384xf32, #gpu.address_space<workgroup>>)
// TABLE:           memref.store %c-1_i32, %[[KEYS]][
// TABLE:           gpu.barrier
// TABLE:           "atom.shared.cas.b32 $0, [$1], $2, $3;"
// TABLE-COUNT-3:   "red.shared.add.f32 [$0], $1;"
// TABLE:           } else {
// TABLE-COUNT-3:   "red.relaxed.gpu.global.add.f32 [$0], $1;"
// TABLE:           gpu.barrier
// TABLE:           memref.load %[[KEYS]][
// TABLE-COUNT-3:   memref.load %[[SUMS]][
// TABLE-COUNT-3:   "red.relaxed.gpu.global.add.f32 [$0], $1;"
// TABLE:           gpu.terminator
//
// By default, from 100,000 values on: the host compares the count.
// LEAST-LABEL: func.func @add(
// LEAST:         arith.cmpi uge, %{{.*}}, %c100000
// LEAST:         scf.if
// LEAST:           gpu.launch {{.*}} workgroup(
// LEAST:         } else {
// LEAST:           gpu.launch
//
// DET: error: 'md_exec.scatter_add' op takes the strategy 'hash', whose precondition does not hold: the deterministic mode adds floating-point values in a fixed order, and a sum of a block first would round otherwise

func.func @add(%v: memref<?x3xf32, 1>, %i: memref<?xi32, 1>,
               %a: memref<?x3xf32, 1>) {
  md_exec.scatter_add %v at %i into %a overwrite(true)
      : memref<?x3xf32, 1>, memref<?xi32, 1>, memref<?x3xf32, 1>
  return
}
