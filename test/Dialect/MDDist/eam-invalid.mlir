// The forces of EAM on domains read the density's adjoint, F'(ρ), at both
// particles of a pair: through a copy to the ghosts of exactly the version
// computed in this step (D[ml2-eam], docs/distributed-ml2.md, Section 15.2).
// A copy of the previous step's F'(ρ), or no copy, is an error of the
// verifier (V2, V1), not of a run.
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!s    = !md_dist.field<@dd, owned, f64>
!sg   = !md_dist.field<@dd, ghost, f64>
!rel  = !md_dist.relation<@dd, directed>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V2: the ghosts hold F'(ρ) of the previous step, from the density the step
// kept, the owners that of this step.
func.func @stale_embedding(%x: !vec, %cell: !md.cell, %rho_prev: !md.field<@atoms, f64>) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0) : !own, !gho, !halo -> !rel
  %rho_old = md_dist.distribute %rho_prev, %l : !md.field<@atoms, f64> -> !s
  %rho = md_dist.gather_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md_dist.yield %f : f64
  } : !rel, !own, !gho -> !s
  // expected-note @+1 {{the version that the op reads}}
  %fp = md_dist.map_particles gather(%rho : !s) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md_dist.yield %s : f64
  } : !s
  // expected-note @+1 {{the version that was copied}}
  %fp_old = md_dist.map_particles gather(%rho_old : !s) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md_dist.yield %s : f64
  } : !s
  // expected-note @+1 {{the copy to the ghosts}}
  %fpg = md_dist.halo_copy %fp_old, %h : !s, !halo -> !sg
  // expected-error @+1 {{stale halo: the ghost values of gathered field #0}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg gather(%fp, %fpg : !s, !sg) exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %ai: f64, %ak: f64):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %df = arith.mulf %c2, %t : f64
    %a = arith.addf %ai, %ak : f64
    %p = arith.mulf %a, %df : f64
    %q = arith.divf %p, %r : f64
    %qv = vector.broadcast %q : f64 to vector<3xf64>
    %k = arith.mulf %qv, %d : vector<3xf64>
    md_dist.yield %k : vector<3xf64>
  } : !rel, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %f = md_dist.halo_sum %fo, %fg, %h
         : !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!s    = !md_dist.field<@dd, owned, f64>
!rel  = !md_dist.relation<@dd, directed>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V1: F'(ρ) is read at the neighbors without its copy to the ghosts.
func.func @missing_embedding_transfer(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0) : !own, !gho, !halo -> !rel
  %rho = md_dist.gather_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md_dist.yield %f : f64
  } : !rel, !own, !gho -> !s
  %fp = md_dist.map_particles gather(%rho : !s) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md_dist.yield %s : f64
  } : !s
  // expected-error @+1 {{missing transfer: 'gather' takes each field as its owned values and their copy to the ghosts}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg gather(%fp : !s) exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %ai: f64):
    %qv = vector.broadcast %ai : f64 to vector<3xf64>
    %k = arith.mulf %qv, %d : vector<3xf64>
    md_dist.yield %k : vector<3xf64>
  } : !rel, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  return
}
