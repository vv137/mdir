// RUN: mdir-opt %s | mdir-opt | FileCheck %s

!vec    = !md.field<@atoms, 3 x f64>
!real   = !md.field<@atoms, f64>
!angles = !md.relation<@atoms, 3, reversal, @angles>
!inc    = !mdrt.incidence<@atoms, @angles, 3>
!of_angle = !md.field<@angles, f64>

md.particle_set @atoms
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)

// CHECK-LABEL: func.func @value_form(
func.func @value_form(%members: memref<?x3xi32>, %x: !vec, %cell: !md.cell,
                      %q: !real, %k: !of_angle) -> (!vec, f64) {
  // CHECK: %[[ANGLES:[0-9]+]] = mdrt.from_buffer %{{[a-z0-9]+}} : memref<?x3xi32> to !md.relation<@atoms, 3, reversal, @angles>
  %angles = mdrt.from_buffer %members : memref<?x3xi32> to !angles

  // CHECK: %[[INC:[0-9]+]] = md_exec.build_incidence %[[ANGLES]]
  // CHECK-SAME: : !md.relation<@atoms, 3, reversal, @angles> -> !mdrt.incidence<@atoms, @angles, 3>
  %inc = md_exec.build_incidence %angles : !angles -> !inc

  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  // CHECK: md_exec.tuple_for %[[INC]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(displacement(0, 1), displacement(2, 1))
  // CHECK-SAME: ins(%{{[a-z0-9]+}} : !md.field<@atoms, f64>)
  // CHECK-SAME: tuple(%{{[a-z0-9]+}} : !md.field<@angles, f64>)
  // CHECK-SAME: outs(%{{[a-z0-9]+}} : !md.field<@atoms, 3 x f64>) reduce(%{{[a-z0-9_]+}} : f64) arity(3) {
  // CHECK-NEXT: ^bb0(%{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: vector<3xf64>, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64, %{{[a-z0-9]+}}: f64):
  // CHECK: md_exec.yield %{{.*}} : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
  // CHECK-NEXT: } : !mdrt.incidence<@atoms, @angles, 3>, !md.field<@atoms, 3 x f64> -> !md.field<@atoms, 3 x f64>, f64
  %f, %u = md_exec.tuple_for %inc, %x, %cell
             coordinates(displacement(0, 1), displacement(2, 1))
             ins(%q : !real) tuple(%k : !of_angle)
             outs(%f0 : !vec) reduce(%u0 : f64) arity(3) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>, %q0: f64, %q1: f64, %q2: f64,
       %k_t: f64):
    %s = arith.addf %a, %b : vector<3xf64>
    %n = arith.negf %s : vector<3xf64>
    md_exec.yield %a, %n, %b, %k_t
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
  } : !inc, !vec -> !vec, f64
  return %f, %u : !vec, f64
}

// CHECK-LABEL: func.func @storage_form(
func.func @storage_form(%members: memref<?x3xi32>, %n: index,
                        %x: memref<?x3xf64>, %cell: !md.cell,
                        %k: memref<?xf64>, %f: memref<?x3xf64>) -> f64 {
  // CHECK: md_exec.build_incidence %{{[a-z0-9]+}} size(%{{[a-z0-9]+}}) : memref<?x3xi32> -> memref<?x?xi32>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x3xi32> -> memref<?x?xi32>
  %u0 = arith.constant 0.0 : f64
  // CHECK: md_exec.tuple_for %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %{{[a-z0-9]+}} coordinates(displacement(0, 1))
  // CHECK-SAME: tuple(%{{[a-z0-9]+}} : memref<?xf64>) outs(%{{[a-z0-9]+}} : memref<?x3xf64>)
  // CHECK-SAME: arity(3) overwrite [true]
  // CHECK: } : memref<?x?xi32>, memref<?x3xf64> -> f64
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
         tuple(%k : memref<?xf64>) outs(%f : memref<?x3xf64>)
         reduce(%u0 : f64) arity(3) overwrite [true] {
  ^bb0(%d: vector<3xf64>, %k_t: f64):
    md_exec.yield %d, %d, %d, %k_t
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, f64
  } : memref<?x?xi32>, memref<?x3xf64> -> f64
  return %u : f64
}
