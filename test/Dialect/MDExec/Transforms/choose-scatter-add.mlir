// RUN: mdir-opt %s --md-exec-choose-scatter | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter="scatter-hash=on block-size=256 hash-probes=4" \
// RUN: | FileCheck %s --check-prefix=ON
// RUN: mdir-opt %s --md-exec-choose-scatter=scatter-hash=off | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-choose-scatter=deterministic=true | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-choose-scatter="block-size=4096 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHY

// The strategy of md_exec.scatter_add (D[scatter-op]): `hash(S, P, L)`
// with S the slots for `block-size` threads of one value each, a power of
// two, and L = `hash-least` (100,000 by default, 0 with scatter-hash=on);
// none with scatter-hash=off, in the deterministic mode, or where the
// table would not fit 48 KiB.

md.particle_set @atoms
md.particle_set @ghosts

// CHECK-LABEL: func.func @add(
// CHECK: md_exec.scatter_add {{.*}} scatter(hash(128, 8, 100000))
// CHECK: md_exec.scatter_add {{.*}} scatter(hash(128, 8, 100000))
// ON: md_exec.scatter_add {{.*}} scatter(hash(256, 4, 0))
// ON: md_exec.scatter_add {{.*}} scatter(hash(256, 4, 0))
// OFF-NOT: scatter(
// 4096 slots of 3 x f64: 4096 x 28 bytes; of one f32: 4096 x 8.
// WHY: remark: scatter: no hash: a table of 4096 slots takes 114688 bytes of block memory, more than 49152
// WHY: remark: scatter: hash(4096, 8, 100000)
func.func @add(%v: !md.field<@ghosts, 3 x f64>, %i: !md.field<@ghosts, i32>,
               %a: !md.field<@atoms, 3 x f64>, %w: !md.field<@ghosts, f32>,
               %b: !md.field<@atoms, f32>)
    -> (!md.field<@atoms, 3 x f64>, !md.field<@atoms, f32>) {
  %a1 = md_exec.scatter_add %v at %i into %a
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  %b1 = md_exec.scatter_add %w at %i into %b
      : !md.field<@ghosts, f32>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, f32>
  return %a1, %b1 : !md.field<@atoms, 3 x f64>, !md.field<@atoms, f32>
}
