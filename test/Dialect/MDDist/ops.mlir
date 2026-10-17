// The ops of md_dist parse, verify, and print back (D275).
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!gho = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.5>

md.particle_set @atoms
// CHECK: md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// CHECK-LABEL: func.func @directed
func.func @directed(%x: !vec, %l: f64) -> (f64, !vec, vector<9xf64>) {
  %cell = md.orthorhombic_cell %l, %l, %l
  // CHECK: md_dist.partition %{{.*}}, %{{.*}} : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  // CHECK: md_dist.distribute %{{.*}}, %{{.*}} image(%{{.*}}) : !md.field<@atoms, 3 x f64> -> !md_dist.field<@dd, owned, 3 x f64>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  // CHECK: md_dist.halo %{{.*}}, %{{.*}}, %{{.*}} : !md_dist.field<@dd, owned, 3 x f64> -> !md_dist.halo<@dd, 2.5>
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  // CHECK: md_dist.halo_copy %{{.*}}, %{{.*}} image(%{{.*}})
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // CHECK: md_dist.neighborhood {{.*}} cutoff(2.000000e+00) : {{.*}} -> !md_dist.relation<@dd, directed>
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
         : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  // CHECK: md_dist.sum_pairs {{.*}} exchange(symmetric)
  %up = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %i6 = arith.constant 6 : i32
    %one = arith.constant 1.0 : f64
    %sr = arith.divf %one, %r : f64
    %s6 = math.fpowi %sr, %i6 : f64, i32
    md_dist.yield %s6 : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  // CHECK: md_dist.all_reduce %{{.*}} : !md_dist.partial<f64> -> f64
  %u = md_dist.all_reduce %up : !md_dist.partial<f64> -> f64
  // CHECK: md_dist.gather_pairs {{.*}} exchange(antisymmetric)
  %fo = md_dist.gather_pairs %n, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c = arith.constant 2.0 : f64
    %s = vector.broadcast %c : f64 to vector<3xf64>
    %f = arith.mulf %s, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  %wp = md_dist.sum_pairs %n, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %w = vector.broadcast %r : f64 to vector<9xf64>
    md_dist.yield %w : vector<9xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<vector<9xf64>>
  %w = md_dist.all_reduce %wp : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
  // CHECK: md_dist.collect %{{.*}}, %{{.*}} : !md_dist.field<@dd, owned, 3 x f64> -> !md.field<@atoms, 3 x f64>
  %f = md_dist.collect %fo, %lay : !own -> !vec
  return %u, %f, %w : f64, !vec, vector<9xf64>
}

// CHECK-LABEL: func.func @unique
func.func @unique(%x: !vec, %q: !md.field<@atoms, f64>, %l: f64) -> !vec {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %qo = md_dist.distribute %q, %lay : !md.field<@atoms, f64> -> !md_dist.field<@dd, owned, f64>
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %qg = md_dist.halo_copy %qo, %h : !md_dist.field<@dd, owned, f64>, !halo -> !md_dist.field<@dd, ghost, f64>
  %n = md_dist.neighborhood %xo, %xg, %h cutoff(2.5)
         : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  // CHECK: md_dist.gather_pairs {{.*}} gather({{.*}}) exchange(antisymmetric)
  // CHECK: } : {{.*}} -> !md_dist.field<@dd, owned, partial, 3 x f64>, !md_dist.field<@dd, ghost, partial, 3 x f64>
  %fo, %fg = md_dist.gather_pairs %n, %xo, %xg
      gather(%qo, %qg : !md_dist.field<@dd, owned, f64>, !md_dist.field<@dd, ghost, f64>)
      exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qk: f64):
    %qq = arith.mulf %qi, %qk : f64
    %s = vector.broadcast %qq : f64 to vector<3xf64>
    %f = arith.mulf %s, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  // CHECK: md_dist.halo_sum
  %f = md_dist.halo_sum %fo, %fg, %h
         : !md_dist.field<@dd, owned, partial, 3 x f64>,
           !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %c = md_dist.collect %f, %lay : !own -> !vec
  return %c : !vec
}

// The transpose of the copy alone, and the inner product of the ghosts.
// CHECK-LABEL: func.func @transpose
func.func @transpose(%x: !vec, %l: f64) -> (f64, !vec) {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  // CHECK: md_dist.zeros %{{.*}} : !md_dist.layout<@dd> -> !md_dist.field<@dd, owned, 3 x f64>
  %z = md_dist.zeros %lay : !md_dist.layout<@dd> -> !own
  %t = md_dist.halo_sum %z, %xg, %h : !own, !gho, !halo -> !own
  // CHECK: md_dist.sum_particles gather(%{{.*}} : !md_dist.field<@dd, ghost, 3 x f64>)
  %p = md_dist.sum_particles gather(%xg : !gho) {
  ^bb0(%v: vector<3xf64>):
    %s = vector.reduction <add>, %v : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %s = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  %c = md_dist.collect %t, %lay : !own -> !vec
  return %s, %c : f64, !vec
}
