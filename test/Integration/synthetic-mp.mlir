// A synthetic message-passing model in stages (#377, step 3;
// D[ml4-neural-op], docs/ml4-ir.md): two PaiNN-like layers whose bodies,
// and their vector-Jacobian products, are functions in the module, written
// by Inputs/synthetic_mp.py. md-differentiate carries the adjoints of the
// features back through the stages, and the forces and the virial come from
// the products of the edge stages, evaluated from both ends of each pair.
//
// The energy, the forces, and the virial against the script's reference:
// one reverse sweep over the whole energy, independently of the rules of
// md-differentiate (forces and virial agree with central differences of the
// energy, extrapolated in the step, to 4e-6 of 143 and 6e-10 of 41).
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %python %S/Inputs/synthetic_mp.py stages > %t/stages.mlir
// RUN: cat %t/stages.mlir %s > %t/model.mlir
// RUN: mdir-opt %t/model.mlir %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The derivatives against central differences of the energy in MDIR itself
// (D184): every coordinate and six strains of the cell.
//
// RUN: mdir-opt %t/model.mlir --md-check-derivative-coverage \
// RUN:     --md-check-derivatives %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The separate form of the derivative, the reference of the one gather of
// the adjoints, the forces, and the virial of each stage kernel over pairs
// (docs/ml4-ir.md, Section 4.3), agrees with the same reference.
//
// RUN: mdir-opt %t/model.mlir --md-check-exchange \
// RUN:     --md-differentiate="separate-stages=true" --md-expand-truncation \
// RUN:     --md-inline \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// In mixed precision, where the bodies compute in f32, within 3e-5 of the
// largest value (measured: 1.0e-6 in the energy, 9.7e-6 in the forces).
//
// RUN: sed -e 's/arith.constant 1.0e-12 : f64/arith.constant 3.0e-5 : f64/' \
// RUN:     %s > %t/mixed.mlir
// RUN: cat %t/stages.mlir %t/mixed.mlir \
// RUN: | mdir-opt %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// A product of the second edge stage that is not the transpose of its
// body's derivative (its cotangent of the distance scaled by 1.001) gives
// forces that the check refuses.
//
// RUN: %python %S/Inputs/synthetic_mp.py stages edge2 > %t/wrong.mlir
// RUN: cat %t/wrong.mlir %s > %t/wrong-model.mlir
// RUN: mdir-opt %t/wrong-model.mlir --md-check-derivatives %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm > %t/wrong.llvm
// RUN: not --crash mlir-runner %t/wrong.llvm -e main \
// RUN:     --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt 2>&1 \
// RUN: | FileCheck %s --check-prefix=WRONG

// WRONG: derivative check of 'model': force component

// The energy, the forces, and the virial agree, each printing 1, and the
// largest difference of each is printed after it.
// CHECK:      {{^}}1{{$}}
// CHECK-NEXT: {{^[0-9.e+-]+$}}
// CHECK-NEXT: {{^}}1{{$}}
// CHECK-NEXT: {{^[0-9.e+-]+$}}
// CHECK-NEXT: {{^}}1{{$}}
// CHECK-NEXT: {{^[0-9.e+-]+$}}

!vec   = !md.field<@atoms, 3 x f64>
!h     = !md.field<@atoms, irreps<4x0e + 4x1o>, f64>  // 16 components
!z     = !md.field<@atoms, i32>
!pairs = !md.relation<@atoms, 2, unordered>

md.particle_set @atoms

md.stage @embed forward(@embed.f) : (i32) -> vector<16xf64>
md.stage @edge1 forward(@edge1.f) vjp(@edge1.b)
    : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
      -> vector<16xf64>
md.stage @node1 forward(@node1.f) vjp(@node1.b)
    : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
md.stage @edge2 forward(@edge2.f) vjp(@edge2.b)
    : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
      -> vector<16xf64>
md.stage @node2 forward(@node2.f) vjp(@node2.b)
    : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
md.stage @readout forward(@readout.f) vjp(@readout.b)
    : (vector<16xf64>) -> f64

md.potential @model(%x: !vec, %cell: !md.cell, %z: !z) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %h0 = md.map_particles gather(%z : !z) {
  ^bb0(%zi: i32):
    %e = md.apply_stage @embed(%zi) : (i32) -> vector<16xf64>
    md.yield %e : vector<16xf64>
  } : !h
  %a1 = md.gather_relation %n, %x, %cell gather(%h0, %z : !h, !z)
      exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<16xf64>, %hj: vector<16xf64>,
       %zi: i32, %zj: i32):
    %m = md.apply_stage @edge1(%r, %d, %hi, %hj, %zi, %zj)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
          -> vector<16xf64>
    md.yield %m : vector<16xf64>
  } : !pairs, !vec -> !h
  %h1 = md.map_particles gather(%h0, %a1 : !h, !h) {
  ^bb0(%hi: vector<16xf64>, %ai: vector<16xf64>):
    %u = md.apply_stage @node1(%hi, %ai)
        : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
    md.yield %u : vector<16xf64>
  } : !h
  %a2 = md.gather_relation %n, %x, %cell gather(%h1, %z : !h, !z)
      exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %hi: vector<16xf64>, %hj: vector<16xf64>,
       %zi: i32, %zj: i32):
    %m = md.apply_stage @edge2(%r, %d, %hi, %hj, %zi, %zj)
        : (f64, vector<3xf64>, vector<16xf64>, vector<16xf64>, i32, i32)
          -> vector<16xf64>
    md.yield %m : vector<16xf64>
  } : !pairs, !vec -> !h
  %h2 = md.map_particles gather(%h1, %a2 : !h, !h) {
  ^bb0(%hi: vector<16xf64>, %ai: vector<16xf64>):
    %u = md.apply_stage @node2(%hi, %ai)
        : (vector<16xf64>, vector<16xf64>) -> vector<16xf64>
    md.yield %u : vector<16xf64>
  } : !h
  %u = md.sum_particles gather(%h2 : !h) {
  ^bb0(%hi: vector<16xf64>):
    %e = md.apply_stage @readout(%hi) : (vector<16xf64>) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

// 1 if every element of `values` is within `tolerance` of `reference`,
// relative to the largest element of the reference; then the largest
// difference, relative to it.
func.func @agrees(%values: memref<?xf64>, %reference: memref<?xf64>,
                  %tolerance: f64) -> (f64, f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %values, %c0 : memref<?xf64>
  %largest = scf.for %i = %c0 to %n step %c1 iter_args(%m = %zero) -> (f64) {
    %v = memref.load %reference[%i] : memref<?xf64>
    %a = math.absf %v : f64
    %b = arith.maximumf %m, %a : f64
    scf.yield %b : f64
  }
  %worst = scf.for %i = %c0 to %n step %c1 iter_args(%m = %zero) -> (f64) {
    %v = memref.load %values[%i] : memref<?xf64>
    %r = memref.load %reference[%i] : memref<?xf64>
    %d = arith.subf %v, %r : f64
    %a = math.absf %d : f64
    %b = arith.maximumf %m, %a : f64
    scf.yield %b : f64
  }
  %bound = arith.mulf %tolerance, %largest : f64
  %ok = arith.cmpf ole, %worst, %bound : f64
  %flag = arith.uitofp %ok : i1 to f64
  %relative = arith.divf %worst, %largest : f64
  return %flag, %relative : f64, f64
}

func.func @report(%flag: f64, %difference: f64) {
  func.call @printF64(%flag) : (f64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF64(%difference) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c9 = arith.constant 9 : index
  %c30 = arith.constant 30 : index
  %count = arith.constant 10 : index
  %xb = memref.alloc(%count) : memref<?x3xf64>
  %zb = memref.alloc(%count) : memref<?xi32>
  %xs = arith.constant dense<[0.563403, 0.986583, 0.494981, 0.389011, 0.359725, 1.192918, 0.623875, 1.19641, 0.49716, 0.557078, 0.250713, 0.37403, 0.451491, 0.605313, 0.302137, 1.157391, 0.157974, 0.367916, 0.308763, 0.327636, 0.399184, 1.048151, 0.009476, 1.075997, 0.364459, 0.612042, 0.615747, 0.703706, 1.105407, 0.178888]> : vector<30xf64>
  %zs = arith.constant dense<[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]> : vector<10xi32>
  scf.for %i = %c0 to %count step %c1 {
    %zi = vector.extract %zs[%i] : i32 from vector<10xi32>
    memref.store %zi, %zb[%i] : memref<?xi32>
    %base = arith.muli %i, %c3 : index
    scf.for %k = %c0 to %c3 step %c1 {
      %p = arith.addi %base, %k : index
      %v = vector.extract %xs[%p] : f64 from vector<30xf64>
      memref.store %v, %xb[%i, %k] : memref<?x3xf64>
    }
  }
  %x = mdrt.from_buffer %xb : memref<?x3xf64> to !vec
  %z = mdrt.from_buffer %zb : memref<?xi32> to !z
  %edge = arith.constant 1.2 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @model(%x, %cell, %z)
      request [energy, forces, virial]
      : (!vec, !md.cell, !z) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  %ub = memref.alloc(%c1) : memref<?xf64>
  %ur = memref.alloc(%c1) : memref<?xf64>
  memref.store %u, %ub[%c0] : memref<?xf64>
  %u_ref = arith.constant -10.4361111628344 : f64
  memref.store %u_ref, %ur[%c0] : memref<?xf64>
  %fv = memref.alloc(%c30) : memref<?xf64>
  %fr = memref.alloc(%c30) : memref<?xf64>
  %f_ref = arith.constant dense<[8.706353612532219, 23.650524803488018, 2.9656364145333054, -0.2828040145352597, -5.576210798608221, -20.398305705080674, -8.938635482389369, -21.16298191750237, -20.217760232368672, 142.5862572800457, -51.746916772426346, -5.75803085662225, -19.977555008927038, -10.446408266610625, 57.750756139350685, -26.82707933966845, -16.513967510106607, -2.9298024557293134, -106.46502939509752, 89.07157874884521, 24.51064982850263, -2.6207233910042453, -0.8179238998777925, 2.2427244189120557, 11.695860252082515, -6.903523709236779, -45.76550131972082, 2.1233554869614633, 0.4458293220355105, 7.599633768223053]> : vector<30xf64>
  scf.for %i = %c0 to %count step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      %p0 = arith.muli %i, %c3 : index
      %p = arith.addi %p0, %k : index
      %v = memref.load %fb[%i, %k] : memref<?x3xf64>
      memref.store %v, %fv[%p] : memref<?xf64>
      %r = vector.extract %f_ref[%p] : f64 from vector<30xf64>
      memref.store %r, %fr[%p] : memref<?xf64>
    }
  }
  %wv = memref.alloc(%c9) : memref<?xf64>
  %wr = memref.alloc(%c9) : memref<?xf64>
  %w_ref = arith.constant dense<[40.91114024218765, -10.444989094715021, 2.700392917073594, -10.444989094715021, -4.01382969258016, 4.124068058242066, 2.7003929170735934, 4.124068058242065, -11.536356642510048]> : vector<9xf64>
  scf.for %k = %c0 to %c9 step %c1 {
    %v = vector.extract %w[%k] : f64 from vector<9xf64>
    memref.store %v, %wv[%k] : memref<?xf64>
    %r = vector.extract %w_ref[%k] : f64 from vector<9xf64>
    memref.store %r, %wr[%k] : memref<?xf64>
  }
  %tight = arith.constant 1.0e-12 : f64
  %e_ok, %e_d = func.call @agrees(%ub, %ur, %tight)
      : (memref<?xf64>, memref<?xf64>, f64) -> (f64, f64)
  %f_ok, %f_d = func.call @agrees(%fv, %fr, %tight)
      : (memref<?xf64>, memref<?xf64>, f64) -> (f64, f64)
  %w_ok, %w_d = func.call @agrees(%wv, %wr, %tight)
      : (memref<?xf64>, memref<?xf64>, f64) -> (f64, f64)
  func.call @report(%e_ok, %e_d) : (f64, f64) -> ()
  func.call @report(%f_ok, %f_d) : (f64, f64) -> ()
  func.call @report(%w_ok, %w_d) : (f64, f64) -> ()
  return
}
