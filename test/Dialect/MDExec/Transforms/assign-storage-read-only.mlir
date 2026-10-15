// RUN: mdir-opt %s --md-exec-assign-storage --check-read-only-arguments | FileCheck %s --check-prefix=HOST
// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" --check-read-only-arguments | FileCheck %s --check-prefix=DEVICE
// RUN: sed 's/ {mdrt.read_only}//' %s | mdir-opt --md-exec-assign-storage | FileCheck %s --check-prefix=PLAIN-HOST
// RUN: sed 's/ {mdrt.read_only}//' %s | mdir-opt --md-exec-assign-storage="memory=device" | FileCheck %s --check-prefix=PLAIN-DEVICE

!real = !md.field<@atoms, f64>

md.particle_set @atoms

// A buffer that the function declares read-only (D258)
// is never taken for another field: on the host a loop that could write in
// place writes to a buffer of its own, and on a device the copy back goes
// to a buffer of the host of its own. Without the declaration the pass takes
// the buffer that it was given, in both.
//
// HOST-LABEL: func.func @scale(
// HOST-SAME:    %[[M:[a-z0-9]+]]: memref<?xf64> {mdrt.read_only}
// HOST:         %[[OWN:[a-z0-9_]+]] = memref.alloc
// HOST:         md_exec.particle_for ins(%[[M]] : memref<?xf64>) outs(%[[OWN]] : memref<?xf64>)
// HOST:         return %[[OWN]]
//
// PLAIN-HOST-LABEL: func.func @scale(
// PLAIN-HOST-SAME:    %[[M:[a-z0-9]+]]: memref<?xf64>
// PLAIN-HOST-NOT:     memref.alloc
// PLAIN-HOST:         md_exec.particle_for ins(%[[M]] : memref<?xf64>) outs(%[[M]] : memref<?xf64>)
// PLAIN-HOST:         return %[[M]]
//
// DEVICE-LABEL: func.func @scale(
// DEVICE-SAME:    %[[M:[a-z0-9]+]]: memref<?xf64> {mdrt.read_only}
// DEVICE:         gpu.memcpy {{.*}} %{{[a-z0-9_]+}}, %[[M]] : memref<?xf64, 1>, memref<?xf64>
// DEVICE:         %[[OWN:[a-z0-9_]+]] = memref.alloc
// DEVICE:         gpu.memcpy {{.*}} %[[OWN]], %{{[a-z0-9_]+}} : memref<?xf64>, memref<?xf64, 1>
// DEVICE:         return %[[OWN]]
//
// PLAIN-DEVICE-LABEL: func.func @scale(
// PLAIN-DEVICE-SAME:    %[[M:[a-z0-9]+]]: memref<?xf64>
// PLAIN-DEVICE-NOT:     memref.alloc
// PLAIN-DEVICE:         gpu.memcpy {{.*}} %[[M]], %{{[a-z0-9_]+}} : memref<?xf64>, memref<?xf64, 1>
// PLAIN-DEVICE:         return %[[M]]
func.func @scale(%masses: memref<?xf64> {mdrt.read_only}, %s: f64)
    -> memref<?xf64> {
  %m = mdrt.from_buffer %masses : memref<?xf64> to !real
  %m0 = md_exec.empty : !real
  %m1 = md_exec.particle_for ins(%m : !real) outs(%m0 : !real) {
  ^bb0(%m_i: f64):
    %scaled = arith.mulf %m_i, %s : f64
    md_exec.yield %scaled : f64
  } -> !real
  %result = mdrt.to_buffer %m1 : !real to memref<?xf64>
  return %result : memref<?xf64>
}
