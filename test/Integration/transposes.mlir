// The adjoint identity of each declared transpose of the structure of the
// md ops (D[md-transposes], docs/primitives-map.md, Section 1.3), through
// md-differentiate, compiled and run on the host in double precision.
//
// md-differentiate takes the transpose of an op's structure from the op
// (AdjointOpInterface) and fills its kernel with the reverse derivative of
// the source kernel. For an op A linear in the field q, the identity is
//
//   <A q, 1> = <q, A^T 1>:  U(q) = <q, ∂U/∂q>,
//
// with ∂U/∂q from the derivative in the field (request derivative(2)).
// For the positions, in which the ops are not linear, it is the identity
// of the linearization J of the energy at x, with the forces F = −J^T 1:
//
//   <J v, 1> = <v, J^T 1>:  δU[v] = −<F, v>,
//
// with δU[v], the derivative along a direction v, written by hand in a
// second potential whose kernels read v (the forward rule, independent of
// md-differentiate). Each check prints 1 if the two sides agree to a
// relative 1e-12 of the larger.
//
// One entry for each pair, source op and the op of its transpose:
// RUN: mdir-opt %s %md_passes --convert-md-to-md-exec="skin=0.2 width=64" \
// RUN:     %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm > %t.ll.mlir
// sum_relation -> gather_relation (positions, forces; a field, a gather):
// RUN: mlir-runner %t.ll.mlir -e sum_relation --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=TWO
// gather_relation -> gather_relation from both ends (positions):
// RUN: mlir-runner %t.ll.mlir -e gather_relation --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=ONE
// map_particles -> map_particles (the adjoint of a field of the positions):
// RUN: mlir-runner %t.ll.mlir -e map_particles --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=ONE
// sum_tuples -> gather_tuples (positions; a field):
// RUN: mlir-runner %t.ll.mlir -e sum_tuples --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=TWO
// sum_particles -> map_particles (positions; a field):
// RUN: mlir-runner %t.ll.mlir -e sum_particles --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=TWO
//
// ONE: 1
// ONE-NOT: {{.}}
// TWO: 1
// TWO-NEXT: 1
// TWO-NOT: {{.}}
//
// The identity tells a transpose from what is not one: the forces of
// @density with the cotangent at the neighbor dropped, written by hand in
// @one_ended, miss it (by 0.68 relative here), since c differs at the two
// ends of a pair.
// RUN: mlir-runner %t.ll.mlir -e wrong --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt | FileCheck %s --check-prefix=WRONG
// WRONG: 0
// WRONG-NOT: {{.}}

// The system of eam.mlir: 64 particles near a cubic lattice of spacing 1.2
// in a periodic cube of edge 4.8, the pairs within 2.0.

!vec   = !md.field<@atoms, 3 x f64>
!sca   = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!bonds = !md.relation<@atoms, 2, unordered, @bonds>
!buf   = memref<?x3xf64>
!sbuf  = memref<?xf64>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)

//===----------------------------------------------------------------------===//
// sum_relation: S = Σ_{i<j} φ(r_ij) and S(q) = Σ_{i<j} f(r_ij) (q_i + q_j)
//===----------------------------------------------------------------------===//

// φ(r) = (r − 2)² (1 + r)
md.potential @pair(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c1 = arith.constant 1.0 : f64
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %s = arith.addf %c1, %r : f64
    %k = arith.mulf %t2, %s : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// δS[v] = Σ_{i<j} φ'(r) d_ij · (v_i − v_j) / r, d_ij = x_i − x_j, with
// φ'(r) = 2 (r − 2) (1 + r) + (r − 2)².
md.potential @pair_jvp(%x: !vec, %cell: !md.cell, %v: !vec) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell gather(%v : !vec)
         exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>, %vi: vector<3xf64>, %vj: vector<3xf64>):
    %c1 = arith.constant 1.0 : f64
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %s = arith.addf %c1, %r : f64
    %a0 = arith.mulf %c2, %t : f64
    %a = arith.mulf %a0, %s : f64
    %slope = arith.addf %a, %t2 : f64
    %dv = arith.subf %vi, %vj : vector<3xf64>
    %p = arith.mulf %d, %dv : vector<3xf64>
    %dot = vector.reduction <add>, %p : vector<3xf64> into f64
    %q = arith.divf %dot, %r : f64
    %k = arith.mulf %slope, %q : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

// f(r) = (r − 2)²: linear in q.
md.potential @pair_q(%x: !vec, %cell: !md.cell, %q: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell gather(%q : !sca) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %t2 = arith.mulf %t, %t : f64
    %qs = arith.addf %qi, %qj : f64
    %k = arith.mulf %t2, %qs : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// gather_relation: U = Σ_i c_i ρ_i, ρ_i = Σ_j f(r_ij), f(r) = (r − 2)²
//===----------------------------------------------------------------------===//

md.potential @density(%x: !vec, %cell: !md.cell, %c: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%rho, %c : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  md.return %u : f64
}

// (J v)_i = Σ_j f'(r_ij) d_ij · (v_i − v_j) / r_ij, and δU[v] = Σ_i c_i (J v)_i.
md.potential @density_jvp(%x: !vec, %cell: !md.cell, %c: !sca, %v: !vec)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %jv = md.gather_relation %n, %x, %cell gather(%v : !vec)
          exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>, %vi: vector<3xf64>, %vj: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %slope = arith.mulf %c2, %t : f64
    %dv = arith.subf %vi, %vj : vector<3xf64>
    %p = arith.mulf %d, %dv : vector<3xf64>
    %dot = vector.reduction <add>, %p : vector<3xf64> into f64
    %q = arith.divf %dot, %r : f64
    %k = arith.mulf %slope, %q : f64
    md.yield %k : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%jv, %c : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// map_particles: U = Σ_i m_i, m_i = c_i ρ_i²
//===----------------------------------------------------------------------===//

md.potential @embed(%x: !vec, %cell: !md.cell, %c: !sca) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %m = md.map_particles gather(%rho, %c : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %a2 = arith.mulf %a, %a : f64
    %p = arith.mulf %a2, %b : f64
    md.yield %p : f64
  } : !sca
  %u = md.sum_particles gather(%m : !sca) {
  ^bb0(%a: f64):
    md.yield %a : f64
  } : f64
  md.return %u : f64
}

// δU[v] = Σ_i 2 c_i ρ_i (J v)_i.
md.potential @embed_jvp(%x: !vec, %cell: !md.cell, %c: !sca, %v: !vec)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %f = arith.mulf %t, %t : f64
    md.yield %f : f64
  } : !pairs, !vec -> !sca
  %jv = md.gather_relation %n, %x, %cell gather(%v : !vec)
          exchange(symmetric, asserted) {
  ^bb0(%r: f64, %d: vector<3xf64>, %vi: vector<3xf64>, %vj: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %slope = arith.mulf %c2, %t : f64
    %dv = arith.subf %vi, %vj : vector<3xf64>
    %p = arith.mulf %d, %dv : vector<3xf64>
    %dot = vector.reduction <add>, %p : vector<3xf64> into f64
    %q = arith.divf %dot, %r : f64
    %k = arith.mulf %slope, %q : f64
    md.yield %k : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%rho, %jv, %c : !sca, !sca, !sca) {
  ^bb0(%a: f64, %g: f64, %b: f64):
    %c2 = arith.constant 2.0 : f64
    %p0 = arith.mulf %c2, %a : f64
    %p1 = arith.mulf %p0, %g : f64
    %p = arith.mulf %p1, %b : f64
    md.yield %p : f64
  } : f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// sum_tuples: S = Σ_b k (r_b − r0)², S(q) = Σ_b r_b (q_0 + 2 q_1)
//===----------------------------------------------------------------------===//

md.potential @bond(%x: !vec, %cell: !md.cell, %bonds: !bonds) -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1)) {
  ^bb0(%r: f64):
    %k = arith.constant 3.0 : f64
    %r0 = arith.constant 1.1 : f64
    %t = arith.subf %r, %r0 : f64
    %t2 = arith.mulf %t, %t : f64
    %e = arith.mulf %k, %t2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

// δS[v] = Σ_b 2 k (r − r0) d_01 · (v_0 − v_1) / r, d_01 = x_0 − x_1.
md.potential @bond_jvp(%x: !vec, %cell: !md.cell, %bonds: !bonds, %v: !vec)
    -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell
         coordinates(displacement(0, 1), distance(0, 1)) gather(%v : !vec) {
  ^bb0(%d: vector<3xf64>, %r: f64, %v0: vector<3xf64>, %v1: vector<3xf64>):
    %k2 = arith.constant 6.0 : f64
    %r0 = arith.constant 1.1 : f64
    %t = arith.subf %r, %r0 : f64
    %slope = arith.mulf %k2, %t : f64
    %dv = arith.subf %v0, %v1 : vector<3xf64>
    %p = arith.mulf %d, %dv : vector<3xf64>
    %dot = vector.reduction <add>, %p : vector<3xf64> into f64
    %q = arith.divf %dot, %r : f64
    %e = arith.mulf %slope, %q : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

md.potential @bond_q(%x: !vec, %cell: !md.cell, %q: !sca, %bonds: !bonds)
    -> f64 {
  %u = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
         gather(%q : !sca) {
  ^bb0(%r: f64, %q0: f64, %q1: f64):
    %c2 = arith.constant 2.0 : f64
    %t = arith.mulf %c2, %q1 : f64
    %s = arith.addf %q0, %t : f64
    %e = arith.mulf %r, %s : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// sum_particles: S = Σ_i (0.1 |x_i|² + 0.3 y_i), S(q) = Σ_i m_i q_i
//===----------------------------------------------------------------------===//

md.potential @well(%x: !vec, %cell: !md.cell) -> f64 {
  %u = md.sum_particles gather(%x : !vec) {
  ^bb0(%a: vector<3xf64>):
    %c = arith.constant 0.1 : f64
    %s = arith.constant 0.3 : f64
    %a2 = arith.mulf %a, %a : vector<3xf64>
    %n = vector.reduction <add>, %a2 : vector<3xf64> into f64
    %e0 = arith.mulf %c, %n : f64
    %y = vector.extract %a[1] : f64 from vector<3xf64>
    %e1 = arith.mulf %s, %y : f64
    %e = arith.addf %e0, %e1 : f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

// δS[v] = Σ_i (0.2 x_i + (0, 0.3, 0)) · v_i.
md.potential @well_jvp(%x: !vec, %cell: !md.cell, %v: !vec) -> f64 {
  %u = md.sum_particles gather(%x, %v : !vec, !vec) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %c = arith.constant dense<0.2> : vector<3xf64>
    %s = arith.constant dense<[0.0, 0.3, 0.0]> : vector<3xf64>
    %g0 = arith.mulf %c, %a : vector<3xf64>
    %g = arith.addf %g0, %s : vector<3xf64>
    %p = arith.mulf %g, %b : vector<3xf64>
    %e = vector.reduction <add>, %p : vector<3xf64> into f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}

md.potential @site_q(%x: !vec, %cell: !md.cell, %q: !sca, %m: !sca) -> f64 {
  %u = md.sum_particles gather(%q, %m : !sca, !sca) {
  ^bb0(%a: f64, %b: f64):
    %p = arith.mulf %a, %b : f64
    md.yield %p : f64
  } : f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// The wrong transpose: the forces of @density written with the cotangent
// at the center alone, −Σ_j c_i f'(r_ij) d_ij / r_ij, which is the
// transpose only where c is the same at both ends.
//===----------------------------------------------------------------------===//

md.potential @one_ended(%x: !vec, %cell: !md.cell, %c: !sca, %v: !vec)
    -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %g = md.gather_relation %n, %x, %cell gather(%c, %v : !sca, !vec)
         exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %ci: f64, %cj: f64, %vi: vector<3xf64>,
       %vj: vector<3xf64>):
    %c2 = arith.constant 2.0 : f64
    %t = arith.subf %r, %c2 : f64
    %slope = arith.mulf %c2, %t : f64
    %w = arith.mulf %ci, %slope : f64
    %p = arith.mulf %d, %vi : vector<3xf64>
    %dot = vector.reduction <add>, %p : vector<3xf64> into f64
    %q = arith.divf %dot, %r : f64
    %k = arith.mulf %w, %q : f64
    md.yield %k : f64
  } : !pairs, !vec -> !sca
  %u = md.sum_particles gather(%g : !sca) {
  ^bb0(%a: f64):
    md.yield %a : f64
  } : f64
  md.return %u : f64
}

//===----------------------------------------------------------------------===//
// The host side
//===----------------------------------------------------------------------===//

func.func private @printF64(f64)
func.func private @printNewline()

// Prints 1 if a and b agree to a relative 1e-12 of the larger, else 0.
func.func @check(%a: f64, %b: f64) {
  %tol = arith.constant 1.0e-12 : f64
  %d = arith.subf %a, %b : f64
  %e = math.absf %d : f64
  %ma = math.absf %a : f64
  %mb = math.absf %b : f64
  %m = arith.maximumf %ma, %mb : f64
  %bound = arith.mulf %tol, %m : f64
  %ok = arith.cmpf ole, %e, %bound : f64
  %nonzero = arith.cmpf ogt, %m, %bound : f64
  %both = arith.andi %ok, %nonzero : i1
  %flag = arith.uitofp %both : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

// The positions of eam.mlir.
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

// The next number of a linear congruential generator and its value in
// [−0.5, 0.5) plus `offset`.
func.func @draw(%s: i64, %offset: f64) -> (i64, f64) {
  %a = arith.constant 1103515245 : i64
  %c = arith.constant 12345 : i64
  %m = arith.constant 2147483648 : i64
  %mf = arith.constant 2147483648.0 : f64
  %half = arith.constant 0.5 : f64
  %t0 = arith.muli %s, %a : i64
  %t1 = arith.addi %t0, %c : i64
  %t2 = arith.remui %t1, %m : i64
  %u = arith.sitofp %t2 : i64 to f64
  %f = arith.divf %u, %mf : f64
  %centered = arith.subf %f, %half : f64
  %value = arith.addf %centered, %offset : f64
  return %t2, %value : i64, f64
}

func.func @fill_vectors(%v: !buf, %seed: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %v, %c0 : !buf
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %seed) -> (i64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (i64) {
      %next, %value = func.call @draw(%s, %zero) : (i64, f64) -> (i64, f64)
      memref.store %value, %v[%i, %k] : !buf
      scf.yield %next : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @fill_numbers(%v: !sbuf, %seed: i64, %offset: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %v, %c0 : !sbuf
  %last = scf.for %i = %c0 to %n step %c1 iter_args(%s = %seed) -> (i64) {
    %next, %value = func.call @draw(%s, %offset) : (i64, f64) -> (i64, f64)
    memref.store %value, %v[%i] : !sbuf
    scf.yield %next : i64
  }
  return
}

func.func @dot_vectors(%a: !buf, %b: !buf) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %a, %c0 : !buf
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%s0 = %zero) -> (f64) {
    %s3 = scf.for %k = %c0 to %c3 step %c1 iter_args(%s = %s0) -> (f64) {
      %x = memref.load %a[%i, %k] : !buf
      %y = memref.load %b[%i, %k] : !buf
      %p = arith.mulf %x, %y : f64
      %t = arith.addf %s, %p : f64
      scf.yield %t : f64
    }
    scf.yield %s3 : f64
  }
  return %sum : f64
}

func.func @dot_numbers(%a: !sbuf, %b: !sbuf) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %a, %c0 : !sbuf
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f64) {
    %x = memref.load %a[%i] : !sbuf
    %y = memref.load %b[%i] : !sbuf
    %p = arith.mulf %x, %y : f64
    %t = arith.addf %s, %p : f64
    scf.yield %t : f64
  }
  return %sum : f64
}

// The positions and a direction v, as buffers; the fields are made in each
// entry, since a field does not cross a call.
func.func @system() -> (!buf, !buf) {
  %count = arith.constant 64 : index
  %xb = memref.alloc(%count) : !buf
  call @place(%xb) : (!buf) -> ()
  %vb = memref.alloc(%count) : !buf
  %seed = arith.constant 7 : i64
  call @fill_vectors(%vb, %seed) : (!buf, i64) -> ()
  return %xb, %vb : !buf, !buf
}

// Copies. The program may write its results into the buffers of fields
// that it no longer reads (md-exec-assign-storage), so that a field takes
// a copy and the comparison reads the original.
func.func @copy_vectors(%a: !buf) -> !buf {
  %c0 = arith.constant 0 : index
  %n = memref.dim %a, %c0 : !buf
  %b = memref.alloc(%n) : !buf
  memref.copy %a, %b : !buf to !buf
  return %b : !buf
}

func.func @copy_numbers(%a: !sbuf) -> !sbuf {
  %c0 = arith.constant 0 : index
  %n = memref.dim %a, %c0 : !sbuf
  %b = memref.alloc(%n) : !sbuf
  memref.copy %a, %b : !sbuf to !sbuf
  return %b : !sbuf
}

// Numbers at the particles: about `offset`, from the seed.
func.func @numbers(%seed: i64, %offset: f64) -> !sbuf {
  %count = arith.constant 64 : index
  %b = memref.alloc(%count) : !sbuf
  call @fill_numbers(%b, %seed, %offset) : (!sbuf, i64, f64) -> ()
  return %b : !sbuf
}

// δU[v] against −<F, v>.
func.func @compare_forces(%jvp: f64, %fb: !buf, %vb: !buf) {
  %fv = call @dot_vectors(%fb, %vb) : (!buf, !buf) -> f64
  %minus = arith.negf %fv : f64
  call @check(%jvp, %minus) : (f64, f64) -> ()
  return
}

// U(q) against <q, ∂U/∂q>.
func.func @compare_field(%u: f64, %qb: !sbuf, %gb: !sbuf) {
  %qg = call @dot_numbers(%qb, %gb) : (!sbuf, !sbuf) -> f64
  call @check(%u, %qg) : (f64, f64) -> ()
  return
}

func.func @sum_relation() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f = md.evaluate @pair(%x, %cell) request [energy, forces]
      : (!vec, !md.cell) -> (f64, !vec)
  %jvp = md.evaluate @pair_jvp(%x, %cell, %v) request [energy]
      : (!vec, !md.cell, !vec) -> f64
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @compare_forces(%jvp, %fb, %vb) : (f64, !buf, !buf) -> ()
  %seed = arith.constant 11 : i64
  %offset = arith.constant 0.0 : f64
  %qb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %qf = call @copy_numbers(%qb) : (!sbuf) -> !sbuf
  %q = mdrt.from_buffer %qf : !sbuf to !sca
  %uq, %g = md.evaluate @pair_q(%x, %cell, %q) request [energy, derivative(2)]
      : (!vec, !md.cell, !sca) -> (f64, !sca)
  %gb = mdrt.to_buffer %g : !sca to !sbuf
  call @compare_field(%uq, %qb, %gb) : (f64, !sbuf, !sbuf) -> ()
  return
}

func.func @gather_relation() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %seed = arith.constant 13 : i64
  %offset = arith.constant 1.0 : f64
  %cb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %c = mdrt.from_buffer %cb : !sbuf to !sca
  %u, %f = md.evaluate @density(%x, %cell, %c) request [energy, forces]
      : (!vec, !md.cell, !sca) -> (f64, !vec)
  %jvp = md.evaluate @density_jvp(%x, %cell, %c, %v) request [energy]
      : (!vec, !md.cell, !sca, !vec) -> f64
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @compare_forces(%jvp, %fb, %vb) : (f64, !buf, !buf) -> ()
  return
}

func.func @map_particles() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %seed = arith.constant 17 : i64
  %offset = arith.constant 1.0 : f64
  %cb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %c = mdrt.from_buffer %cb : !sbuf to !sca
  %u, %f = md.evaluate @embed(%x, %cell, %c) request [energy, forces]
      : (!vec, !md.cell, !sca) -> (f64, !vec)
  %jvp = md.evaluate @embed_jvp(%x, %cell, %c, %v) request [energy]
      : (!vec, !md.cell, !sca, !vec) -> f64
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @compare_forces(%jvp, %fb, %vb) : (f64, !buf, !buf) -> ()
  return
}

func.func @sum_tuples() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  // 32 bonds, (2b, 2b + 1): neighbors along x on the lattice.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %mb = memref.alloc(%c32) : memref<?x2xi32>
  scf.for %b = %c0 to %c32 step %c1 {
    %i = arith.muli %b, %c2 : index
    %j = arith.addi %i, %c1 : index
    %ii = arith.index_cast %i : index to i32
    %jj = arith.index_cast %j : index to i32
    memref.store %ii, %mb[%b, %c0] : memref<?x2xi32>
    memref.store %jj, %mb[%b, %c1] : memref<?x2xi32>
  }
  %bonds = mdrt.from_buffer %mb : memref<?x2xi32> to !bonds
  %u, %f = md.evaluate @bond(%x, %cell, %bonds) request [energy, forces]
      : (!vec, !md.cell, !bonds) -> (f64, !vec)
  %jvp = md.evaluate @bond_jvp(%x, %cell, %bonds, %v) request [energy]
      : (!vec, !md.cell, !bonds, !vec) -> f64
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @compare_forces(%jvp, %fb, %vb) : (f64, !buf, !buf) -> ()
  %seed = arith.constant 19 : i64
  %offset = arith.constant 0.0 : f64
  %qb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %qf = call @copy_numbers(%qb) : (!sbuf) -> !sbuf
  %q = mdrt.from_buffer %qf : !sbuf to !sca
  %uq, %g = md.evaluate @bond_q(%x, %cell, %q, %bonds)
      request [energy, derivative(2)]
      : (!vec, !md.cell, !sca, !bonds) -> (f64, !sca)
  %gb = mdrt.to_buffer %g : !sca to !sbuf
  call @compare_field(%uq, %qb, %gb) : (f64, !sbuf, !sbuf) -> ()
  return
}

func.func @sum_particles() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f = md.evaluate @well(%x, %cell) request [energy, forces]
      : (!vec, !md.cell) -> (f64, !vec)
  %jvp = md.evaluate @well_jvp(%x, %cell, %v) request [energy]
      : (!vec, !md.cell, !vec) -> f64
  %fb = mdrt.to_buffer %f : !vec to !buf
  call @compare_forces(%jvp, %fb, %vb) : (f64, !buf, !buf) -> ()
  %seed = arith.constant 23 : i64
  %offset = arith.constant 0.0 : f64
  %qb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %qf = call @copy_numbers(%qb) : (!sbuf) -> !sbuf
  %q = mdrt.from_buffer %qf : !sbuf to !sca
  %seed_m = arith.constant 29 : i64
  %one = arith.constant 1.0 : f64
  %mb = call @numbers(%seed_m, %one) : (i64, f64) -> !sbuf
  %m = mdrt.from_buffer %mb : !sbuf to !sca
  %uq, %g = md.evaluate @site_q(%x, %cell, %q, %m)
      request [energy, derivative(2)]
      : (!vec, !md.cell, !sca, !sca) -> (f64, !sca)
  %gb = mdrt.to_buffer %g : !sca to !sbuf
  call @compare_field(%uq, %qb, %gb) : (f64, !sbuf, !sbuf) -> ()
  return
}

func.func @wrong() {
  %xb, %vb = call @system() : () -> (!buf, !buf)
  %x = mdrt.from_buffer %xb : !buf to !vec
  %vf = call @copy_vectors(%vb) : (!buf) -> !buf
  %v = mdrt.from_buffer %vf : !buf to !vec
  %edge = arith.constant 4.8 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %seed = arith.constant 13 : i64
  %offset = arith.constant 1.0 : f64
  %cb = call @numbers(%seed, %offset) : (i64, f64) -> !sbuf
  %c = mdrt.from_buffer %cb : !sbuf to !sca
  // −<F, v> of the one-ended forces against δU[v].
  %fv = md.evaluate @one_ended(%x, %cell, %c, %v) request [energy]
      : (!vec, !md.cell, !sca, !vec) -> f64
  %jvp = md.evaluate @density_jvp(%x, %cell, %c, %v) request [energy]
      : (!vec, !md.cell, !sca, !vec) -> f64
  call @check(%jvp, %fv) : (f64, f64) -> ()
  return
}
