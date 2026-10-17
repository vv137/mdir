// The derivative rules of G and G^T, through md-differentiate
// (D[ml2-distribute], docs/distributed-ml2.md, Section 14).
//
// The potential, written in md_dist,
//
//   U(x) = <c, G^T G_x x>_O = <G c, G_x x>_Γ,
//
// is affine in the positions x, with the gradient G^T G c: md-differentiate
// takes the adjoint of the ghost rows of G^T (md_dist.halo_sum) by its rule,
// a copy G (md_dist.halo_copy), and that of the ghost positions G_x by its
// rule, a sum G^T, and collects the owned rows. The forces F = −G^T G c
// then satisfy the adjoint identity
//
//   <c, F>_O = −<G c, G c>_Γ,
//
// whose right side is computed directly by a copy to the ghosts and an
// inner product over the ghost rows. Each grid of dist-lj.mlir; the check
// prints 1 if the two sides agree to a relative 1e-12.
//
// RUN: sed 's/GRID/1, 1, 1/' %s | mdir-opt %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: sed 's/GRID/2, 1, 1/' %s | mdir-opt %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: sed 's/GRID/3, 2, 1/' %s | mdir-opt %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
// RUN: sed 's/GRID/16, 1, 1/' %s | mdir-opt %md_passes --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The derivative takes both rules: a copy without images for the transpose
// of G^T, and a sum for that of the copy of the positions.
// RUN: sed 's/GRID/2, 1, 1/' %s | mdir-opt --md-differentiate \
// RUN: | FileCheck %s --check-prefix=RULES
// RULES-LABEL: md.function @quad.forces
// RULES: %[[H:.*]] = md_dist.halo
// RULES: %[[XG:.*]] = md_dist.halo_copy %{{.*}}, %[[H]] image(%arg1)
// RULES: %[[M:.*]] = md_dist.map_particles
// RULES: %[[B:.*]] = md_dist.halo_copy %[[M]], %[[H]] : !md_dist.field<@dd, owned, 3 x f64>
// RULES: %[[Z:.*]] = md_dist.zeros
// RULES: %[[S:.*]] = md_dist.halo_sum %[[Z]], %[[B]], %[[H]]
// RULES: %[[F:.*]] = md_dist.collect %[[S]]
// RULES: md.return %[[F]]

!vec  = !md.field<@atoms, 3 x f64>
!own  = !md_dist.field<@dd, owned, 3 x f64>
!gho  = !md_dist.field<@dd, ghost, 3 x f64>
!halo = !md_dist.halo<@dd, 2.0>
!buf  = memref<?x3xf64>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])

md.potential @quad(%x: !vec, %cell: !md.cell, %c: !vec) -> f64 {
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho
  %co = md_dist.distribute %c, %l : !vec -> !own
  %z = md_dist.zeros %l : !md_dist.layout<@dd> -> !own
  %v = md_dist.halo_sum %z, %xg, %h : !own, !gho, !halo -> !own
  %p = md_dist.sum_particles gather(%co, %v : !own, !own) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %ab = arith.mulf %a, %b : vector<3xf64>
    %s = vector.reduction <add>, %ab : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %u = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
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

// A field c that differs between particles: c_i = (1 + i mod 5, −x_i, 2).
func.func @field(%x: !buf) -> !buf {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  %n = memref.dim %x, %c0 : !buf
  %c = memref.alloc(%n) : !buf
  scf.for %i = %c0 to %n step %c1 {
    %r = arith.remui %i, %c5 : index
    %ri = arith.index_cast %r : index to i64
    %rf = arith.sitofp %ri : i64 to f64
    %a = arith.addf %one, %rf : f64
    %xi = memref.load %x[%i, %c0] : !buf
    %b = arith.negf %xi : f64
    memref.store %a, %c[%i, %c0] : !buf
    memref.store %b, %c[%i, %c1] : !buf
    memref.store %two, %c[%i, %c2] : !buf
  }
  return %c : !buf
}

func.func @copy(%x: !buf) -> !buf {
  %c0 = arith.constant 0 : index
  %n = memref.dim %x, %c0 : !buf
  %y = memref.alloc(%n) : !buf
  memref.copy %x, %y : !buf to !buf
  return %y : !buf
}

// <c, F> with F the forces of the potential.
func.func @left(%xb: !buf, %cb: !buf, %edge: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %y = call @copy(%xb) : (!buf) -> !buf
  %d = call @copy(%cb) : (!buf) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %c = mdrt.from_buffer %d : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %f = md.evaluate @quad(%x, %cell, %c) request [forces]
      : (!vec, !md.cell, !vec) -> !vec
  %fb = mdrt.to_buffer %f : !vec to !buf
  %n = memref.dim %fb, %c0 : !buf
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%a = %zero) -> (f64) {
    %t = scf.for %k = %c0 to %c3 step %c1 iter_args(%b = %a) -> (f64) {
      %fv = memref.load %fb[%i, %k] : !buf
      %cv = memref.load %cb[%i, %k] : !buf
      %p = arith.mulf %fv, %cv : f64
      %q = arith.addf %b, %p : f64
      scf.yield %q : f64
    }
    scf.yield %t : f64
  }
  return %s : f64
}

// <G c, G c> over the ghost rows.
func.func @right(%xb: !buf, %cb: !buf, %edge: f64) -> f64 {
  %y = call @copy(%xb) : (!buf) -> !buf
  %d = call @copy(%cb) : (!buf) -> !buf
  %x = mdrt.from_buffer %y : !buf to !vec
  %c = mdrt.from_buffer %d : !buf to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  %h = md_dist.halo %l, %xo, %cell : !own -> !halo
  %co = md_dist.distribute %c, %l : !vec -> !own
  %cg = md_dist.halo_copy %co, %h : !own, !halo -> !gho
  %p = md_dist.sum_particles gather(%cg : !gho) {
  ^bb0(%a: vector<3xf64>):
    %aa = arith.mulf %a, %a : vector<3xf64>
    %s = vector.reduction <add>, %aa : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %s = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  return %s : f64
}

// CHECK: 1
func.func @main() {
  %count = arith.constant 64 : index
  %x = memref.alloc(%count) : !buf
  call @place(%x) : (!buf) -> ()
  %c = call @field(%x) : (!buf) -> !buf
  %edge = arith.constant 4.8 : f64
  %lhs = call @left(%x, %c, %edge) : (!buf, !buf, f64) -> f64
  %rhs = call @right(%x, %c, %edge) : (!buf, !buf, f64) -> f64
  // <c, F> + <G c, G c> = 0, to 1e-12 of <G c, G c>.
  %sum = arith.addf %lhs, %rhs : f64
  %error = math.absf %sum : f64
  %tolerance = arith.constant 1.0e-12 : f64
  %bound = arith.mulf %tolerance, %rhs : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}
