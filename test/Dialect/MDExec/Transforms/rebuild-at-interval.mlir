// RUN: mdir-opt %s --md-exec-rebuild-at-interval=interval=10 | FileCheck %s
// RUN: not mdir-opt %s --md-exec-rebuild-at-interval 2>&1 \
// RUN: | FileCheck %s --check-prefix=NONE

!vec = !md.field<@atoms, 3 x f64>
!nl  = !mdrt.neighbors<@atoms>

md.particle_set @atoms

// NOT A DEFAULT (D88). A refresh that would test the displacements builds
// at a fixed interval instead; one whose test a loop has made, and one that
// always builds, keep their policies.
//
// CHECK-LABEL: func.func @refreshes(
// CHECK:         md_exec.refresh_neighbors {{.*}} interval(10) policy(interval)
// CHECK:         md_exec.refresh_neighbors {{.*}} moved(%{{[a-z0-9]+}}) {{.*}} policy(check)
// CHECK:         md_exec.refresh_neighbors {{.*}} policy(always)
// NONE: expected a positive interval of rebuilds, got 0
func.func @refreshes(%x: !vec, %cell: !md.cell, %nl: !nl, %moved: i1)
    -> (!nl, !nl, !nl) {
  %a = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check) : !nl, !vec
  %b = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(check) : !nl, !vec
  %c = md_exec.refresh_neighbors %nl, %x, %cell
      cutoff(2.5) skin(0.3) cell_width(2.8) policy(always) : !nl, !vec
  return %a, %b, %c : !nl, !nl, !nl
}
