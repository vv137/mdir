// A many-body energy on several domains through md-distribute
// (D276, docs/distributed-ml2.md, Section 14.2): a density
// gathered over pairs, a map of it, a sum over particles with a field
// argument, and a pair sum that reads the map at both particles, through
// its copy to the ghosts. On the unique traversal the density is summed at
// the owners by G^T before the map reads it.
//
// The energy is compared with the same potential on one domain through
// md_exec; each check prints 1 if they agree to a relative 1e-10.
//
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-distribute="grid=2,1,1 traversal=unique potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-distribute="grid=3,2,1 traversal=unique potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=directed potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: mdir-opt %s --md-distribute="grid=16,1,1 traversal=unique potentials=many_body" \
// RUN:     %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!buf   = memref<?x3xf64>

md.particle_set @atoms

// U = Σ_i q_i sqrt(ρ_i) + Σ_{i<j} ψ_i ψ_j / r_ij, ρ_i = Σ_j exp(−r_ij),
// ψ = sqrt(ρ).
md.potential @many_body(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %psi = md.map_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : !sca
  %e1 = md.sum_particles gather(%psi, %q : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

// The same, on one domain.
md.potential @reference(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %m = arith.negf %r : f64
    %e = math.exp %m : f64
    md.yield %e : f64
  } : !pairs, !vec -> !sca
  %psi = md.map_particles gather(%rho : !sca) {
  ^bb0(%a: f64):
    %s = math.sqrt %a : f64
    md.yield %s : f64
  } : !sca
  %e1 = md.sum_particles gather(%psi, %q : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  %e2 = md.sum_relation %n, %x, %cell gather(%psi : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %pi: f64, %pj: f64):
    %p = arith.mulf %pi, %pj : f64
    %k = arith.divf %p, %r : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  %u = arith.addf %e1, %e2 : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

// Places the particles, as lj-forces.mlir does.
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

// q_i = 1 + (i mod 3) / 2: a field that differs between particles.
func.func @charges(%n: index) -> memref<?xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %one = arith.constant 1.0 : f64
  %half = arith.constant 0.5 : f64
  %q = memref.alloc(%n) : memref<?xf64>
  scf.for %i = %c0 to %n step %c1 {
    %r = arith.remui %i, %c3 : index
    %ri = arith.index_cast %r : index to i64
    %rf = arith.sitofp %ri : i64 to f64
    %h = arith.mulf %rf, %half : f64
    %v = arith.addf %one, %h : f64
    memref.store %v, %q[%i] : memref<?xf64>
  }
  return %q : memref<?xf64>
}

func.func @distributed(%n: index, %edge: f64) -> f64 {
  %x = memref.alloc(%n) : !buf
  call @place(%x) : (!buf) -> ()
  %qb = call @charges(%n) : (index) -> memref<?xf64>
  %xs = mdrt.from_buffer %x : !buf to !vec
  %qs = mdrt.from_buffer %qb : memref<?xf64> to !sca
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u = md.evaluate @many_body(%xs, %cell, %qs) request [energy]
      : (!vec, !md.cell, !sca) -> f64
  return %u : f64
}

func.func @one_domain(%n: index, %edge: f64) -> f64 {
  %x = memref.alloc(%n) : !buf
  call @place(%x) : (!buf) -> ()
  %qb = call @charges(%n) : (index) -> memref<?xf64>
  %xs = mdrt.from_buffer %x : !buf to !vec
  %qs = mdrt.from_buffer %qb : memref<?xf64> to !sca
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u = md.evaluate @reference(%xs, %cell, %qs) request [energy]
      : (!vec, !md.cell, !sca) -> f64
  return %u : f64
}

// CHECK: 1
func.func @main() {
  %n = arith.constant 64 : index
  %edge = arith.constant 4.8 : f64
  %ud = call @distributed(%n, %edge) : (index, f64) -> f64
  %u = call @one_domain(%n, %edge) : (index, f64) -> f64
  %d = arith.subf %ud, %u : f64
  %e = math.absf %d : f64
  %s = math.absf %u : f64
  %tolerance = arith.constant 1.0e-10 : f64
  %bound = arith.mulf %tolerance, %s : f64
  %agrees = arith.cmpf ole, %e, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
