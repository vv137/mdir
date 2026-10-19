// An embedded-atom potential on one domain, compiled and run: the density
// is a gather over the pairs, symmetric, and its forces come through the
// embedding (D279 on domains; here without md-distribute), against
// Inputs/eam_reference.py. eam-groups-gpu.test runs the same program on a
// device, over groups, with each strategy of the scatter of the loops over
// groups (D[scatter-op]).
//
// RUN: mdir-opt %s %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// The system of dist-eam.mlir: 64 particles near a cubic lattice of spacing
// 1.2 in a periodic cube of edge 4.8, and the potential of the analytic
// form of Finnis and Sinclair [Finnis1984], in reduced units:
//
//   U = Σ_i F(ρ_i) + Σ_{i<j} φ(r_ij),  ρ_i = Σ_{j≠i} f(r_ij),
//   f(r) = (r − 2)²,  F(ρ) = −√ρ,  φ(r) = (r − 2)² (3 − 2r + 0.4r²),
//
// for r < 2. The values of Inputs/eam_reference.py: the energy
// 31.860504482882163; the force components [0][0] 0.44715410193808536,
// [17][1] -0.5103718852189691, [45][0] -0.9530026683464269, [63][1]
// 1.3779958407190194, each to the largest, 2.240583255195037; the trace of
// the virial 584.8240522027986 and its xy component -0.576229182332499, to
// the trace. Each check prints 1 if the value agrees to a relative
// `%tol`, 1e-10 here (the GPU test takes 1e-5 in mixed precision).
//
// CHECK: 31.8605
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NOT: {{.}}

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!buf   = memref<?x3xf64>

md.particle_set @atoms

md.potential @eam(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %e1 = md.sum_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    %m = arith.negf %s : f64
    md.yield %m : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %c0 = arith.constant 3.0 : f64
    %c1 = arith.constant -2.0 : f64
    %cq = arith.constant 0.4 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %c1, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %cq, %r2 : f64
    %p0 = arith.addf %c0, %a : f64
    %p = arith.addf %p0, %b : f64
    %k = arith.mulf %t2, %p : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %scale: f64, %tolerance: f64) {
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %scale : f64
  %bound = arith.mulf %tolerance, %magnitude : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// The positions of dist-eam.mlir: the sites of a 4 x 4 x 4 lattice, each
// moved by up to 0.1 along each axis by a linear congruential generator.
func.func @place(%x: !buf) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %spacing = arith.constant 1.2 : f64
  %amplitude = arith.constant 0.2 : f64
  %half = arith.constant 0.5 : f64
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %seed = arith.constant 42 : i64
  %n = memref.dim %x, %c0 : !buf
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %p0 = arith.cmpi eq, %k, %c0 : index
      %p1 = arith.cmpi eq, %k, %c1 : index
      %d1 = arith.divui %i, %c4 : index
      %d2 = arith.divui %d1, %c4 : index
      %q01 = arith.select %p1, %d1, %d2 : index
      %q = arith.select %p0, %i, %q01 : index
      %digit = arith.remui %q, %c4 : index
      %wide = arith.index_cast %digit : index to i64
      %site = arith.sitofp %wide : i64 to f64
      %base = arith.mulf %site, %spacing : f64
      %t0 = arith.muli %s, %a : i64
      %t1 = arith.addi %t0, %c : i64
      %t2 = arith.remui %t1, %m : i64
      %u = arith.sitofp %t2 : i64 to f64
      %f = arith.divf %u, %mf : f64
      %centered = arith.subf %f, %half : f64
      %shift = arith.mulf %centered, %amplitude : f64
      %value = arith.addf %base, %shift : f64
      memref.store %value, %x[%i, %k] : !buf
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c17 = arith.constant 17 : index
  %c45 = arith.constant 45 : index
  %c63 = arith.constant 63 : index
  %count = arith.constant 64 : index
  %xb = memref.alloc(%count) : !buf
  call @place(%xb) : (!buf) -> ()
  %edge = arith.constant 4.8 : f64
  %tol = arith.constant 1.0e-10 : f64
  %x = mdrt.from_buffer %xb : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @eam(%x, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  %u_ref = arith.constant 31.860504482882163 : f64
  call @check(%u, %u_ref, %u_ref, %tol) : (f64, f64, f64, f64) -> ()
  %scale = arith.constant 2.240583255195037 : f64
  %f00 = memref.load %fb[%c0, %c0] : !buf
  %f00_ref = arith.constant 0.44715410193808536 : f64
  call @check(%f00, %f00_ref, %scale, %tol) : (f64, f64, f64, f64) -> ()
  %f17 = memref.load %fb[%c17, %c1] : !buf
  %f17_ref = arith.constant -0.5103718852189691 : f64
  call @check(%f17, %f17_ref, %scale, %tol) : (f64, f64, f64, f64) -> ()
  %f45 = memref.load %fb[%c45, %c0] : !buf
  %f45_ref = arith.constant -0.9530026683464269 : f64
  call @check(%f45, %f45_ref, %scale, %tol) : (f64, f64, f64, f64) -> ()
  %f63 = memref.load %fb[%c63, %c1] : !buf
  %f63_ref = arith.constant 1.3779958407190194 : f64
  call @check(%f63, %f63_ref, %scale, %tol) : (f64, f64, f64, f64) -> ()
  %wxx = vector.extract %w[0] : f64 from vector<9xf64>
  %wyy = vector.extract %w[4] : f64 from vector<9xf64>
  %wzz = vector.extract %w[8] : f64 from vector<9xf64>
  %t0 = arith.addf %wxx, %wyy : f64
  %trace = arith.addf %t0, %wzz : f64
  %trace_ref = arith.constant 584.8240522027986 : f64
  call @check(%trace, %trace_ref, %trace_ref, %tol) : (f64, f64, f64, f64) -> ()
  %wxy = vector.extract %w[1] : f64 from vector<9xf64>
  %wxy_ref = arith.constant -0.576229182332499 : f64
  call @check(%wxy, %wxy_ref, %trace_ref, %tol) : (f64, f64, f64, f64) -> ()
  // The forces sum to zero, to the tolerance of the largest component.
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %sum = scf.for %i = %c0 to %count step %c1 iter_args(%s = %zero) -> (f64) {
    %fx = memref.load %fb[%i, %c0] : !buf
    %s1 = arith.addf %s, %fx : f64
    scf.yield %s1 : f64
  }
  %n64 = arith.constant 64.0 : f64
  %bound = arith.mulf %scale, %n64 : f64
  call @check(%sum, %zero, %bound, %tol) : (f64, f64, f64, f64) -> ()
  return
}
