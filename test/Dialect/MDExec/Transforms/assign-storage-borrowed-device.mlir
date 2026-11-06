// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" | FileCheck %s --implicit-check-not=gpu.memcpy
// RUN: not mdir-opt %s --md-exec-assign-storage="memory=host" 2>&1 | FileCheck %s --check-prefix=HOST
// HOST: a borrowed device field requires memory=device
//
// The input remains caller-owned and read-only even at its last use. Its
// result must use a distinct allocation, with no host/device copy.
// CHECK-LABEL: func.func @borrowed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<?x3xf64, 1>)
// CHECK: %[[BORROWED:[a-zA-Z0-9_]+]] = memref.cast %[[INPUT]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = gpu.alloc
// CHECK: md_exec.particle_for ins(%[[BORROWED]]
// CHECK-SAME: outs(%[[OUTPUT]]
// CHECK-NOT: gpu.dealloc %[[BORROWED]]
// CHECK: call @consume
// CHECK-NOT: gpu.dealloc %[[BORROWED]]
// CHECK: return

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
func.func private @consume(memref<?x3xf64>)
func.func @borrowed(%input: memref<?x3xf64, 1>) {
  %field = mdrt.from_buffer %input : memref<?x3xf64, 1> to !vec
  %empty = md_exec.empty : !vec
  %output = md_exec.particle_for ins(%field : !vec) outs(%empty : !vec) {
  ^bb0(%x: vector<3xf64>):
    %one = arith.constant dense<1.0> : vector<3xf64>
    %y = arith.addf %x, %one : vector<3xf64>
    md_exec.yield %y : vector<3xf64>
  } -> !vec
  mdrt.host_call @consume(%output) {in_place} : (!vec)
  return
}
