// RUN: mdir-opt %s --md-exec-expose-validity | FileCheck %s
// RUN: mdir-opt %s --md-exec-expose-validity --md-exec-fuse-loops \
// RUN: | FileCheck %s --check-prefix=FUSED

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// The test of validity becomes a loop over particles. The limit is the
// square of half the skin. With the drift before it, the loop is fused: the
// thread that moves a particle tests it.
//
// CHECK-LABEL: func.func @steps(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[V:[a-z0-9]+]]: !md.field<@atoms, 3 x f64>, %[[CELL:[a-z0-9]+]]: !md.cell, %[[N:[a-z0-9]+]]: index)
// FUSED-LABEL: func.func @steps(
func.func @steps(%x: !vec, %v: !vec, %cell: !md.cell, %n: index) -> !vec {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48) : !nl
  // CHECK:      scf.for
  // CHECK-SAME:   iter_args(%[[XA:[a-z0-9]+]] = %[[X]], %[[NA:[a-z0-9]+]] = %{{[0-9]+}})
  %xe, %nle = scf.for %step = %c0 to %n step %c1
      iter_args(%xa = %x, %na = %nl0) -> (!vec, !nl) {
    // CHECK:      %[[X1:[0-9]+]] = md_exec.particle_for ins(%[[XA]], %[[V]] :
    // FUSED:      %[[LOOP:[0-9]+]]:2 = md_exec.particle_for
    // FUSED-SAME:   ins(%[[XA:[a-z0-9]+]], %{{[a-z0-9]+}}, %[[REF:[0-9]+]] :
    // FUSED-SAME:   outs(%{{[0-9]+}} : !md.field<@atoms, 3 x f64>) reduce(%{{[a-z0-9_]+}} : i1)
    // FUSED-NEXT: ^bb0(%[[XI:[a-z0-9]+]]: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %[[RI:[a-z0-9]+]]: vector<3xf64>):
    // FUSED-NEXT:   %[[XN:[0-9]+]] = arith.addf %[[XI]],
    // FUSED-NEXT:   %[[D:[0-9]+]] = arith.subf %[[XN]], %[[RI]]
    // FUSED:        md_exec.yield %[[XN]], %{{[0-9]+}} : vector<3xf64>, i1
    // FUSED-NOT:  md_exec.particle_for
    // FUSED:      md_exec.refresh_neighbors %{{[a-z0-9]+}}, %[[LOOP]]#0, %{{[a-z0-9]+}} moved(%[[LOOP]]#1)
    %e = md_exec.empty : !vec
    %x1 = md_exec.particle_for ins(%xa, %v : !vec, !vec) outs(%e : !vec) {
    ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>):
      %xn = arith.addf %x_i, %v_i : vector<3xf64>
      md_exec.yield %xn : vector<3xf64>
    } -> !vec

    // CHECK:      %[[REF:[0-9]+]] = md_exec.reference_positions %[[NA]] : !mdrt.neighbors<@atoms> -> !md.field<@atoms, 3 x f64>
    // CHECK:      %[[NO:[a-z0-9_]+]] = arith.constant false
    // CHECK:      %[[MOVED:[0-9]+]] = md_exec.particle_for ins(%[[X1]], %[[REF]] :
    // CHECK-SAME:   reduce(%[[NO]] : i1) {
    // CHECK-NEXT: ^bb0(%[[NOW:[a-z0-9]+]]: vector<3xf64>, %[[THEN:[a-z0-9]+]]: vector<3xf64>):
    // CHECK-NEXT:   %[[D:[0-9]+]] = arith.subf %[[NOW]], %[[THEN]]
    // CHECK-NEXT:   %[[SQ:[0-9]+]] = arith.mulf %[[D]], %[[D]]
    // CHECK-NEXT:   %[[D2:[0-9]+]] = vector.reduction <add>, %[[SQ]]
    // CHECK-NEXT:   %[[LIMIT:[a-z0-9_]+]] = arith.constant 1.562500e-02 : f64
    // CHECK-NEXT:   %[[FAR:[0-9]+]] = arith.cmpf ugt, %[[D2]], %[[LIMIT]]
    // CHECK-NEXT:   md_exec.yield %[[FAR]] : i1
    // CHECK:      %[[NB:[0-9]+]] = md_exec.refresh_neighbors %[[NA]], %[[X1]], %[[CELL]] moved(%[[MOVED]])
    // CHECK-SAME:   cutoff(1.500000e+00) skin(2.500000e-01) cell_width(1.750000e+00) policy(check)
    %nb = md_exec.refresh_neighbors %na, %x1, %cell
        cutoff(1.5) skin(0.25) cell_width(1.75) policy(check) : !nl, !vec
    scf.yield %x1, %nb : !vec, !nl
  }
  return %xe : !vec
}

// A refresh of an empty structure builds it whatever has moved.
//
// CHECK-LABEL: func.func @first(
func.func @first(%x: !vec, %cell: !md.cell) -> !nl {
  // CHECK-NOT:  md_exec.reference_positions
  // CHECK-NOT:  md_exec.particle_for
  // CHECK:      md_exec.refresh_neighbors
  // CHECK-SAME:   policy(always)
  %nl0 = md_exec.empty_neighbors kind(matrix) width(48) : !nl
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check) : !nl, !vec
  return %nl : !nl
}

// A refresh that has its test, one that builds always, and one in the
// storage form are left alone.
//
// CHECK-LABEL: func.func @left_alone(
func.func @left_alone(%x: !vec, %cell: !md.cell, %nl: !nl, %moved: i1,
                      %b: memref<?x3xf64>, %n: index) {
  // CHECK-NOT:  md_exec.reference_positions
  // CHECK:      md_exec.refresh_neighbors %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} moved(%{{[a-z0-9]+}})
  // CHECK:      md_exec.refresh_neighbors
  // CHECK-SAME:   policy(always)
  // CHECK:      md_exec.refresh_neighbors
  // CHECK-SAME:   policy(check) : !mdrt.neighbors<@atoms>, memref<?x3xf64>
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check) : !nl, !vec
  %nl2 = md_exec.refresh_neighbors %nl1, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always) : !nl, !vec
  %s = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64>)
      kind(matrix) width(48) : !nl
  %s1 = md_exec.refresh_neighbors %s, %b, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !nl, memref<?x3xf64>
  return
}
