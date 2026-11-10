// D399: a step loop on domains carries one
// md_dist.state; it prints and parses back, and the three witnesses of
// coverage are accepted: a halo built at the positions, one unpack, and the
// then-region of a branch on md_dist.covered.
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!own   = !md_dist.field<@dd, owned, 3 x f64>
!gho   = !md_dist.field<@dd, ghost, 3 x f64>
!halo  = !md_dist.halo<@dd, 2.4>
!lay   = !md_dist.layout<@dd>
!state = !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 2, 2])

// CHECK-LABEL: func.func @loop
// CHECK: md_dist.pack %{{.*}}, %{{.*}}, %{{.*}} fields(%{{.*}}, %{{.*}} : !md_dist.field<@dd, owned, 3 x f64>, !md_dist.field<@dd, owned, 3 x f64>) : !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
// CHECK: scf.for
// CHECK: md_dist.unpack %{{.*}}, %{{.*}} : !md_dist.state<@dd, reach = 2.4, cutoff = 2.0, fields = [3 x f64, 3 x f64]>
// CHECK: md_dist.covered
// CHECK: scf.if
func.func @loop(%x: !vec, %v: !vec, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %l0 = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l0 image(%cell) : !vec -> !own
  %v0 = md_dist.distribute %v, %l0 : !vec -> !own
  %h0 = md_dist.halo %l0, %x0, %cell : !own -> !halo
  %s0 = md_dist.pack %l0, %h0, %cell fields(%x0, %v0 : !own, !own) : !state
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%sa = %s0) -> (!state) {
    %l, %h, %xr, %xa, %va = md_dist.unpack %sa, %cell : !state
    %x1 = md_dist.map_particles gather(%xa, %va : !own, !own) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %y = arith.addf %x_i, %v_i : vector<3xf64>
      md_dist.yield %y : vector<3xf64>
    } : !own
    %ok = md_dist.covered %h, %xr, %x1, %cell cutoff(2.0) : !halo, !own -> i1
    %next = scf.if %ok -> (!state) {
      %kept = md_dist.pack %l, %h, %cell fields(%x1, %va : !own, !own) : !state
      scf.yield %kept : !state
    } else {
      %l2 = md_dist.migrate %l, %x1, %cell : !lay, !own -> !lay
      %x2 = md_dist.redistribute %x1, %l, %l2 image(%cell) : !own, !lay, !lay -> !own
      %v2 = md_dist.redistribute %va, %l, %l2 : !own, !lay, !lay -> !own
      %h2 = md_dist.halo %l2, %x2, %cell : !own -> !halo
      %rebuilt = md_dist.pack %l2, %h2, %cell fields(%x2, %v2 : !own, !own) : !state
      scf.yield %rebuilt : !state
    }
    scf.yield %next : !state
  }
  %lf, %hf, %xrf, %xf, %vf = md_dist.unpack %s, %cell : !state
  %xg = md_dist.halo_copy %xf, %hf image(%cell) : !own, !halo -> !gho
  %r = md_dist.neighborhood %xf, %xg, %hf cutoff(2.0)
      : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}
