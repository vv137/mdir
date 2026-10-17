// The verifier of md_dist rejects what would compute a wrong result:
// V1 to V9 of docs/distributed-ml2.md, Section 4.9 (D275).
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V1: the positions are read at neighbors, but their ghost values are not
// there: the owned field stands where the copy should.
func.func @missing_transfer(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  // expected-error @+1 {{missing transfer: the positions is read at neighbors}}
  %n = md_dist.neighborhood %xo, %xo, %h cutoff(2.0)
         : !own, !own, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!s    = !md_dist.field<@dd, owned, f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V1: a gathered field without its copy to the ghosts.
func.func @missing_field_transfer(%x: !vec, %q: !md.field<@atoms, f64>, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %qo = md_dist.distribute %q, %l : !md.field<@atoms, f64> -> !s
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{missing transfer: 'gather' takes each field as its owned values and their copy to the ghosts}}
  %f = md_dist.gather_pairs %n, %xo, %xg gather(%qo : !s) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V2: the ghosts hold a copy of the positions before they moved, and the
// op reads the positions after: a stale halo.
func.func @stale_halo(%x0: !vec, %x1: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  // expected-note @+1 {{the version that was copied}}
  %xo0 = md_dist.distribute %x0, %l image(%cell) : !vec -> !own
  // expected-note @+1 {{the version that the op reads}}
  %xo1 = md_dist.distribute %x1, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo0, %cell : !own -> !halo
  // expected-note @+1 {{the copy to the ghosts}}
  %xg0 = md_dist.halo_copy %xo0, %h image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{stale halo: the ghost values of the positions are a copy of another version of the field than the owned values that the op reads}}
  %n = md_dist.neighborhood %xo1, %xg0, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!s    = !md_dist.field<@dd, owned, f64>
!sg   = !md_dist.field<@dd, ghost, f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V2 for an intermediate field, as the embedding derivative of EAM: the
// ghosts hold the version of another evaluation.
func.func @stale_intermediate(%x: !vec, %p0: !md.field<@atoms, f64>, %p1: !md.field<@atoms, f64>, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  // expected-note @+1 {{the version that was copied}}
  %po0 = md_dist.distribute %p0, %l : !md.field<@atoms, f64> -> !s
  // expected-note @+1 {{the version that the op reads}}
  %po1 = md_dist.distribute %p1, %l : !md.field<@atoms, f64> -> !s
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // expected-note @+1 {{the copy to the ghosts}}
  %pg0 = md_dist.halo_copy %po0, %h : !s, !halo -> !sg
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{stale halo: the ghost values of gathered field #0}}
  %f = md_dist.gather_pairs %n, %xo, %xg gather(%po1, %pg0 : !s, !sg) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pk: f64):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V3: the positions were copied through another halo than the relation's.
func.func @wrong_halo(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h1 = md_dist.halo %l, %xo, %cell : !own -> !md_dist.halo<@dd, 2.0>
  %h2 = md_dist.halo %l, %xo, %cell : !own -> !md_dist.halo<@dd, 2.5>
  // expected-note @+1 {{the copy to the ghosts}}
  %xg = md_dist.halo_copy %xo, %h2 image(%cell) : !own, !md_dist.halo<@dd, 2.5> -> !gho
  // expected-error @+1 {{were copied through another halo than the one the op traverses}}
  %n = md_dist.neighborhood %xo, %xg, %h1 cutoff(2.0)
         : !own, !gho, !md_dist.halo<@dd, 2.0> -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V3: ghost positions without their images.
func.func @no_images(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h : !own, !halo -> !gho
  // expected-error @+1 {{the ghost positions must be copied with their images}}
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 1.5>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V4: the halo reaches less far than the cutoff.
func.func @short_reach(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{the reach of the halo, 1.500000e+00, is less than the cutoff, 2.000000e+00: the ghosts do not cover the pairs}}
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V5: a field of one layout with the halo of another: the ownership of an
// earlier epoch.
func.func @stale_layout(%x0: !vec, %x1: !vec, %cell: !md.cell) {
  // expected-note @+1 {{the layout of the op}}
  %l0 = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  // expected-note @+1 {{the layout of the field}}
  %l1 = md_dist.partition %x1, %cell : !vec -> !md_dist.layout<@dd>
  %xo0 = md_dist.distribute %x0, %l0 image(%cell) : !vec -> !own
  %xo1 = md_dist.distribute %x1, %l1 image(%cell) : !vec -> !own
  %h = md_dist.halo %l0, %xo0, %cell : !own -> !halo
  // expected-error @+1 {{the field is of another layout (epoch of ownership) than the op}}
  %xg = md_dist.halo_copy %xo1, %h image(%cell)
          : !own, !halo -> !md_dist.field<@dd, ghost, 3 x f64>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V7: the contributions at the ghosts are never returned to their owners.
func.func @lost_contributions(%x: !vec, %cell: !md.cell) -> !vec {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error @+1 {{result #0 is partial and must have exactly one use, which completes it; it has 0: contributions would be lost}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  return %x : !vec
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V7: the contributions of a traversal summed twice.
func.func @counted_twice(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error @+1 {{it has 2: contributions would be counted more than once}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  %f1 = md_dist.halo_sum %fo, %fg, %h : !op, !gp, !halo -> !own
  %f2 = md_dist.halo_sum %fo, %fg, %h : !op, !gp, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V7: the owned contributions of one traversal with the ghost
// contributions of another.
func.func @mixed_traversals(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  %ao, %ag = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  %bo, %bg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %e = arith.negf %d : vector<3xf64>
    md_dist.yield %e : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  // expected-error @+1 {{the partial operands are not the two results of one traversal}}
  %f1 = md_dist.halo_sum %ao, %bg, %h : !op, !gp, !halo -> !own
  %f2 = md_dist.halo_sum %bo, %ag, %h : !op, !gp, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V6: a partial field read as if it were complete.
func.func @partial_read(%x: !vec, %cell: !md.cell) -> !vec {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error @+1 {{result #0 is a partial field, which only md_dist.halo_sum may consume, not 'md_dist.collect'}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  %f = md_dist.collect %fo, %l : !op -> !vec
  %z = md_dist.zeros %l : !md_dist.layout<@dd> -> !own
  %s = md_dist.halo_sum %z, %fg, %h : !own, !gp, !halo -> !own
  return %f : !vec
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V8: a pair computed once needs the value at its second particle from an
// exchange contract.
func.func @unique_without_exchange(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error @+1 {{the contributions at the second particle of a pair come from the exchange contract}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  %f = md_dist.halo_sum %fo, %fg, %h : !op, !gp, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V8: the stated contract is not what the kernel computes: d times r is
// antisymmetric, d itself added to a constant vector is neither.
func.func @false_contract(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // expected-error @+1 {{cannot prove that the kernel is antisymmetric under exchange of the two particles}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %one = arith.constant dense<1.0> : vector<3xf64>
    %e = arith.addf %d, %one : vector<3xf64>
    md_dist.yield %e : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !op, !gp
  %f = md_dist.halo_sum %fo, %fg, %h : !op, !gp, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// The weight one half of a directed sum is right for a symmetric kernel
// only.
func.func @sum_not_symmetric(%x: !vec, %cell: !md.cell) -> f64 {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{a sum over pairs needs a symmetric kernel}}
  %p = md_dist.sum_pairs %n, %xo, %xg exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %r : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %u = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  return %u : f64
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V7 for a number: a partial sum never reduced over the domains.
func.func @unreduced(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{result #0 is partial and must have exactly one use}}
  %p = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %r : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V9: a relation found at one configuration, traversed at another, with
// nothing that shows it still covers the pairs.
func.func @other_configuration(%x0: !vec, %x1: !vec, %cell: !md.cell) -> f64 {
  %l = md_dist.partition %x0, %cell : !vec -> !md_dist.layout<@dd>
  %xo0 = md_dist.distribute %x0, %l image(%cell) : !vec -> !own
  %xo1 = md_dist.distribute %x1, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo0, %cell : !own -> !halo
  %xg0 = md_dist.halo_copy %xo0, %h image(%cell) : !own, !halo -> !gho
  %xg1 = md_dist.halo_copy %xo1, %h image(%cell) : !own, !halo -> !gho
  // expected-note @+1 {{the relation}}
  %n = md_dist.neighborhood %xo0, %xg0, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{the relation was found at other positions than those the op reads}}
  %p = md_dist.sum_pairs %n, %xo1, %xg1 exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %r : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %u = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  return %u : f64
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
md.particle_set @other
md_dist.decomposition @dd on(@other) grid([2, 1, 1])

// A decomposition of another particle set than the positions'.
func.func @other_set(%x: !vec, %cell: !md.cell) {
  // expected-error @+1 {{the positions are of @atoms, the decomposition of @other}}
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  return
}

// -----

md.particle_set @atoms
// expected-error @+1 {{expected divisions of at least 1, got 0}}
md_dist.decomposition @dd on(@atoms) grid([2, 0, 1])

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// The two-ended gather over a directed relation (D276) adds
// half of each pair at its second particle from the exchange contract, too.
func.func @directed_two_ended_without_exchange(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{the contributions at the second particle of a pair come from the exchange contract}}
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !op, !gp
  %f = md_dist.halo_sum %fo, %fg, %h : !op, !gp, !halo -> !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// One result over a directed relation is the complete field of owned rows,
// not values at the ghosts.
func.func @directed_one_ghost(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // expected-error @+1 {{over a directed relation every pair of a particle is at its owner: expected one complete field of owned rows, or two partial fields of owned and of ghost rows}}
  %fo = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md_dist.yield %d : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !gho
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// A map over owned and ghost rows at once: the rows are not the same.
func.func @map_two_roles(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // expected-error @+1 {{expected fields of one role}}
  %m = md_dist.map_particles gather(%xo, %xg : !own, !gho) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %s = arith.addf %a, %b : vector<3xf64>
    md_dist.yield %s : vector<3xf64>
  } : !own
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// A map gives values of the rows it reads.
func.func @map_other_rows(%x: !vec, %cell: !md.cell) {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  // expected-error @+1 {{expected a complete field of the rows of the fields it reads}}
  %m = md_dist.map_particles gather(%xo : !own) {
  ^bb0(%a: vector<3xf64>):
    md_dist.yield %a : vector<3xf64>
  } : !gho
  return
}

// -----

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!op   = !md_dist.field<@dd, owned, partial, 3 x f64>
!gp   = !md_dist.field<@dd, ghost, partial, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// V6: a map does not read a partial field.
func.func @map_partial(%x: !vec, %cell: !md.cell, %p: !op) {
  // expected-error @+1 {{a partial field cannot be read}}
  %m = md_dist.map_particles gather(%p : !op) {
  ^bb0(%a: vector<3xf64>):
    md_dist.yield %a : vector<3xf64>
  } : !own
  return
}
