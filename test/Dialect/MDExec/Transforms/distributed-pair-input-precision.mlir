// RUN: mdir-opt %s --md-exec-assign-precision="mode=mixed" | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-precision="mode=double" | FileCheck %s --check-prefix=DOUBLE

!vec = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>

module attributes {md_dist.driver} {
  md.particle_set @atoms
  // The first field is read only by the pair kernel. The second reaches
  // a native consumer as well and keeps its full precision.
  // CHECK-LABEL: func.func @gather
  // CHECK: md_exec.pair_for {{.*}} ins({{.*}} : !md.field<@atoms, f32>, !md.field<@atoms, f64>)
  // CHECK-NEXT: ^bb0({{.*}}: f32, {{.*}}: vector<3xf32>, {{.*}}: f32, {{.*}}: f32, {{.*}}: f64, {{.*}}: f64):
  // DOUBLE-LABEL: func.func @gather
  // DOUBLE: md_exec.pair_for {{.*}} ins({{.*}} : !md.field<@atoms, f64>, !md.field<@atoms, f64>)
  func.func @gather(%xbuf: memref<?x3xf64>, %qbuf: memref<?xf64>,
                    %mbuf: memref<?xf64>, %cell: !md.cell)
      -> (f64, !real) attributes {md_dist.domain_procedure} {
    %x = mdrt.from_buffer %xbuf : memref<?x3xf64> to !vec
    %q = mdrt.from_buffer %qbuf : memref<?xf64> to !real
    %m = mdrt.from_buffer %mbuf : memref<?xf64> to !real
    %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
    %list = md_exec.build_neighbors %cells, %x, %cell cutoff(0.8) skin(0.2)
        kind(matrix) width(96) : !mdrt.cells<@atoms>, !vec -> !nl
    %zero = arith.constant 0.0 : f64
    %sum = md_exec.pair_for %list, %x, %cell ins(%q, %m : !real, !real)
        reduce(%zero : f64) cutoff(0.8) weights [0.5] policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>, %qi: f64, %qj: f64, %mi: f64, %mj: f64):
      %qq = arith.mulf %qi, %qj : f64
      %v = arith.mulf %qq, %mi : f64
      md_exec.yield %v : f64
    } : !nl, !vec -> f64
    return %sum, %m : f64, !real
  }
}
