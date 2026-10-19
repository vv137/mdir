// RUN: mdir-opt %s --md-exec-choose-scatter | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-choose-scatter=pair-hash=auto | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-choose-scatter="pair-hash=auto explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO
// RUN: mdir-opt %s --md-exec-choose-scatter=pair-hash=on | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter=pair-hash=on | mdir-opt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-exec-choose-scatter="pair-hash=on block-size=256 hash-probes=4 pair-hash-least=1000" \
// RUN: | FileCheck %s --check-prefix=WIDE
// RUN: mdir-opt %s --md-exec-choose-scatter="pair-hash=on deterministic=true" \
// RUN: | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-choose-scatter="pair-hash=on explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHY
// RUN: mdir-opt %s --md-exec-choose-scatter="explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHYOFF
// RUN: mdir-opt %s --md-exec-choose-scatter="pair-hash=on block-size=1024 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=BIG
// RUN: not mdir-opt %s --md-exec-choose-scatter=pair-hash=yes 2>&1 \
// RUN: | FileCheck %s --check-prefix=SWITCH

// The loops over pairs that add the value of a pair at both of its
// particles (the conflict `atomic`: the loops over groups, each pair once)
// take the table of a block, `hash(S, P, L)` (D291), with S the
// slots for `block-size` threads that each add to 3 particles a unit of
// work, rounded up to a power of two: 512 for 128 threads, 1024 for 256.
// `pair-hash=off` is the default, and `pair-hash=auto` chooses the atomics
// as well (measured faster at every loop over pairs). A loop that writes only at its centers,
// one with only global sums, one whose table would not fit 48 KiB, and the
// deterministic mode take none.
//
// With 1024 threads, 4096 slots: the forces' table, 4096 x (4 + 12)
// bytes, exceeds 48 KiB; the density's, 4096 x (4 + 4), fits.
// BIG: remark: scatter: no hash: a table of 4096 slots takes 65536 bytes of block memory, more than 49152
// BIG: remark: scatter: hash(4096, 8, 0)

!vec = !md.field<@atoms, 3 x f32>
!real = !md.field<@atoms, f32>
!nl = !mdrt.neighbors<@atoms>
md.particle_set @atoms

// The forces of a loop over groups, and EAM's density.
// CHECK-LABEL: func.func @groups(
// CHECK: md_exec.pair_for {{.*}} policy(unique, atomic) scatter(hash(512, 8, 0))
// CHECK: md_exec.pair_for {{.*}} policy(unique, atomic) scatter(hash(512, 8, 0))
// WIDE-LABEL: func.func @groups(
// WIDE: md_exec.pair_for {{.*}} policy(unique, atomic) scatter(hash(1024, 4, 1000))
// OFF-LABEL: func.func @groups(
// OFF-NOT: scatter(
// OFF: return
// WHY: remark: scatter: hash(512, 8, 0)
// WHY: remark: scatter: hash(512, 8, 0)
// WHYOFF: remark: scatter: no hash: pair-hash=off
// AUTO: remark: scatter: no hash: the table was measured slower at every loop over pairs, 1.18 to 2.06 times the atomics (pair-hash=auto)
func.func @groups(%x: !vec, %cell: !md.cell, %nl: !nl, %f0: !vec,
                  %rho0: !real) -> (!vec, !real) {
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(unique, atomic) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !vec -> !vec
  %rho = md_exec.pair_for %nl, %x, %cell outs(%rho0 : !real) cutoff(1.0)
      exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !nl, !vec -> !real
  return %f, %rho : !vec, !real
}

// A loop over the matrix writes at its centers, each by its own thread; a
// loop with only a global sum adds to no field.
// CHECK-LABEL: func.func @none(
// CHECK-NOT: scatter(
// CHECK: return
func.func @none(%x: !vec, %cell: !md.cell, %nl: !nl, %f0: !vec) -> (!vec, f64) {
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !vec -> !vec
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell reduce(%u0 : f64) cutoff(1.0)
      exchange [symmetric] policy(unique, atomic) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_exec.yield %e : f64
  } : !nl, !vec -> f64
  return %f, %u : !vec, f64
}

// SWITCH: expected 'auto', 'on', or 'off' for 'tile', 'hash', 'pair-hash', and 'scatter-hash', got 'auto', 'auto', 'yes', and 'auto'
