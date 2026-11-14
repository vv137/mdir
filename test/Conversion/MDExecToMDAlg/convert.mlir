// convert-md-exec-to-md-alg (D[md-alg-traverse]): each loop over pairs over
// groups becomes md_alg.traverse_pairs in its structure form, with the
// visit, the combination, the algebra, and the exchange contracts of the
// loop, and its region. Of a dual list (D114) the op reads the inner list.
// The first loop after the refresh at the same positions and in the same
// cell takes the frames that the pruning left (D268), and with
// `fuse-pruning` prunes for the refresh (D284). The second loop follows one
// that writes a buffer that may be the positions (the forces, which the
// loop carries), so it takes neither, as before the op existed.
//
// RUN: mdir-opt %s --md-exec-expose-validity --md-exec-choose-neighbors \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device \
// RUN:     --convert-md-exec-to-md-alg \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-exec-expose-validity --md-exec-choose-neighbors \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage=memory=device \
// RUN:     --convert-md-exec-to-md-alg="fuse-pruning=true fuse-pruning-groups=7" \
// RUN: | FileCheck %s --check-prefix=FUSED
// In the deterministic mode every sum is fixed_order, and the op states a
// stable visit, which its verifier refuses with an atomic combination.
// RUN: not mdir-opt %s --md-exec-expose-validity --md-exec-choose-neighbors \
// RUN:     --md-exec-assign-algebra=deterministic=true \
// RUN:     --md-exec-assign-storage=memory=device --convert-md-exec-to-md-alg \
// RUN:     2>&1 | FileCheck %s --check-prefix=FIXED

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>

md.particle_set @atoms

// CHECK-LABEL: func.func @dual(
// CHECK:         %[[S:[0-9]+]] = md_exec.refresh_neighbors {{.*}} prune_skin(
// CHECK:         md_alg.traverse_pairs structure %[[S]] : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)> inner
// CHECK-SAME:      unique visit unordered combine atomic algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
// CHECK-SAME:      cutoff(2.500000e+00) weights [5.000000e-01] frames {
// CHECK-NOT:       md_exec.yield
// CHECK:           md_alg.yield
// CHECK:         md_alg.traverse_pairs structure %[[S]] : {{.*}} inner
// CHECK-SAME:      algebra ["any_order"] exchange ["symmetric"] cutoff(2.500000e+00) weights [5.000000e-01] {
// CHECK-NOT:     md_exec.pair_for {{.*}} policy(unique
// FUSED-LABEL: func.func @dual(
// FUSED:         md_alg.traverse_pairs {{.*}} frames prune(7, 2.600000e+00) {
// FUSED:         md_alg.traverse_pairs {{.*}} weights [5.000000e-01] {
// FIXED: 'md_alg.traverse_pairs' op adds destination 0, 'fixed_order', by the combination 'atomic', whose order the threads decide
func.func @dual(%x: !vec, %cell: !md.cell, %n: index) -> (!vec, f64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %u0 = arith.constant 0.0 : f64
  %f0 = md_exec.zeros : !vec
  %g0 = md_exec.empty_neighbors kind(matrix) width(96) : !nl
  %g, %f, %a, %b = scf.for %i = %c0 to %n step %c1
      iter_args(%s = %g0, %h = %f0, %p = %u0, %q = %u0)
      -> (!nl, !vec, f64, f64) {
    %s1 = md_exec.refresh_neighbors %s, %x, %cell
        cutoff(2.5) skin(0.3) prune_skin(0.1) cell_width(2.8) policy(check)
        : !nl, !vec
    %h1, %u = md_exec.pair_for %s1, %x, %cell outs(%h : !vec)
        reduce(%u0 : f64) cutoff(2.5) weights [0.5]
        exchange [antisymmetric, symmetric] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %d, %r2 : vector<3xf64>, f64
    } : !nl, !vec -> !vec, f64
    %w = md_exec.pair_for %s1, %x, %cell reduce(%u0 : f64) cutoff(2.5)
        weights [0.5] exchange [symmetric] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %p1 = arith.addf %p, %u : f64
    %q1 = arith.addf %q, %w : f64
    scf.yield %s1, %h1, %p1, %q1 : !nl, !vec, f64, f64
  }
  return %f, %a, %b : !vec, f64, f64
}
