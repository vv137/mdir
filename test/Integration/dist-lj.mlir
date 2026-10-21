// Lennard-Jones on several domains in one process against one domain
// (D275, docs/distributed-ml2.md, Section 12).
//
// The distributed program is written in md_dist and run by the reference
// executor (convert-md-dist-to-loops), whose pair search covers all pairs
// of each domain's owned and ghost rows; the reference is the same
// potential in md on one domain, differentiated by md-differentiate and run
// through md_exec's neighbor structures. Both traversals are run: directed
// (every pair at both owners, no reverse transfer) and unique (every pair
// once, the ghosts' forces returned by G^T). The grid comes from sed:
//
//   1 x 1 x 1   one domain, whose ghosts are images of its own particles
//   2 x 1 x 1   two domains
//   3 x 2 x 1   uneven domains (a 4.8 edge in thirds)
//   16 x 1 x 1  domains narrower than the cutoff, some of them empty, with
//               ghosts from domains that are not neighbors
//
// RUN: sed 's/GRID/2, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/1, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/3, 2, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// The trace of the transport shows the rows of each domain: domains 1 and
// 2 of the 16 own nothing.
//
// RUN: sed 's/GRID/16, 1, 1/' %s \
// RUN: | mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %md_passes \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | env MDRT_DIST_TRACE=1 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt 2> %t.trace \
// RUN: | FileCheck %s
// RUN: FileCheck %s --check-prefix=TRACE < %t.trace

// The system of lj-forces.mlir: 64 particles on a cubic lattice with the
// spacing 1.2, each displaced by up to 0.1 along every axis, in a periodic
// cube with the edge 4.8; Lennard-Jones with eps = 1 and sigma = 1,
// truncated at 2.0 without a switch.
//
// Each check prints 1 if the distributed value agrees with the one-domain
// value: the energy to a relative 1e-10, every force component to 1e-10 of
// the largest component, every virial component to 1e-10 of the trace, and
// the two sides of the adjoint identity to a relative 1e-12.

!vec   = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
!own   = !md_dist.field<@dd, owned, 3 x f64>
!gho   = !md_dist.field<@dd, ghost, 3 x f64>
!halo  = !md_dist.halo<@dd, 2.0>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])

md.potential @lj(%x: !vec, %cell: !md.cell, %eps: f64, %sigma: f64) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md.yield %k : f64
  } : !pairs, !vec -> f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

// Prints 1 if |value - reference| <= tolerance * |scale|, else 0.
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

// Prints 1 if every component of `a` is within tolerance * max |b| of `b`.
func.func @check_forces(%a: memref<?x3xf64>, %b: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %n = memref.dim %b, %c0 : memref<?x3xf64>
  %m:2 = scf.for %i = %c0 to %n step %c1 iter_args(%e = %zero, %s = %zero) -> (f64, f64) {
    %r:2 = scf.for %k = %c0 to %c3 step %c1 iter_args(%e1 = %e, %s1 = %s) -> (f64, f64) {
      %va = memref.load %a[%i, %k] : memref<?x3xf64>
      %vb = memref.load %b[%i, %k] : memref<?x3xf64>
      %d = arith.subf %va, %vb : f64
      %ad = math.absf %d : f64
      %ab = math.absf %vb : f64
      %e2 = arith.maximumf %e1, %ad : f64
      %s2 = arith.maximumf %s1, %ab : f64
      scf.yield %e2, %s2 : f64, f64
    }
    scf.yield %r#0, %r#1 : f64, f64
  }
  %tolerance = arith.constant 1.0e-10 : f64
  call @check(%m#0, %zero, %m#1, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

// Prints 1 if every component of `w` agrees with `ref` to 1e-10 of the
// trace of `ref`.
func.func @check_virial(%w: vector<9xf64>, %ref: vector<9xf64>) {
  %d = arith.subf %w, %ref : vector<9xf64>
  %ad = math.absf %d : vector<9xf64>
  %e = vector.reduction <maximumf>, %ad : vector<9xf64> into f64
  %xx = vector.extract %ref[0] : f64 from vector<9xf64>
  %yy = vector.extract %ref[4] : f64 from vector<9xf64>
  %zz = vector.extract %ref[8] : f64 from vector<9xf64>
  %t0 = arith.addf %xx, %yy : f64
  %trace = arith.addf %t0, %zz : f64
  %zero = arith.constant 0.0 : f64
  %tolerance = arith.constant 1.0e-10 : f64
  call @check(%e, %zero, %trace, %tolerance) : (f64, f64, f64, f64) -> ()
  return
}

// Places the particles, as lj-forces.mlir does.
func.func @place(%x: memref<?x3xf64>) {
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
  %n = memref.dim %x, %c0 : memref<?x3xf64>
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
      memref.store %value, %x[%i, %k] : memref<?x3xf64>
      scf.yield %t2 : i64
    }
    scf.yield %s3 : i64
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %count = arith.constant 64 : index
  %buffer = memref.alloc(%count) : memref<?x3xf64>
  call @place(%buffer) : (memref<?x3xf64>) -> ()
  %copy = memref.alloc(%count) : memref<?x3xf64>
  memref.copy %buffer, %copy : memref<?x3xf64> to memref<?x3xf64>

  %edge = arith.constant 4.8 : f64
  %eps = arith.constant 1.0 : f64
  %sigma = arith.constant 1.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  // The distributed evaluation.
  %xs = mdrt.from_buffer %copy : memref<?x3xf64> to !vec
  %lay = md_dist.partition %xs, %cell : !vec -> !md_dist.layout<@dd>
  %xo = md_dist.distribute %xs, %lay image(%cell) : !vec -> !own
  %h = md_dist.halo %lay, %xo, %cell : !own -> !halo
  %xg = md_dist.halo_copy %xo, %h image(%cell) : !own, !halo -> !gho

  // Directed: every pair at both of its owners.
  %nd = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
          : !own, !gho, !halo -> !md_dist.relation<@dd, directed>
  %id, %bd = md_dist.partition_centers %nd, %lay, %h
      : !md_dist.relation<@dd, directed>, !md_dist.layout<@dd>, !halo
      -> !md_dist.field<@dd, owned, i32>, !md_dist.field<@dd, owned, i32>
  %udp = md_dist.sum_pairs %nd, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %uip = md_dist.sum_pairs %nd, %xo, %xg center_set(%id : !md_dist.field<@dd, owned, i32>) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %ui = md_dist.all_reduce %uip : !md_dist.partial<f64> -> f64
  %ubp = md_dist.sum_pairs %nd, %xo, %xg center_set(%bd : !md_dist.field<@dd, owned, i32>) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<f64>
  %ub = md_dist.all_reduce %ubp : !md_dist.partial<f64> -> f64
  %uf = arith.addf %ui, %ub : f64
  %ud = md_dist.all_reduce %udp : !md_dist.partial<f64> -> f64
  // The force on the center: 4 eps (12 s^12 - 6 s^6) / r^2 times d.
  %fdo = md_dist.gather_pairs %nd, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  %fio = md_dist.gather_pairs %nd, %xo, %xg center_set(%id : !md_dist.field<@dd, owned, i32>) exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  %fbo = md_dist.gather_pairs %nd, %xo, %xg center_set(%bd : !md_dist.field<@dd, owned, i32>) exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !own
  %ffo = md_dist.map_particles gather(%fio, %fbo : !own, !own) {
  ^bb0(%a: vector<3xf64>, %b: vector<3xf64>):
    %sum = arith.addf %a, %b : vector<3xf64>
    md_dist.yield %sum : vector<3xf64>
  } : !own
  %ffd = md_dist.collect %ffo, %lay : !own -> !vec
  %fd = md_dist.collect %fdo, %lay : !own -> !vec
  // The virial: the force on the center times d, as an outer product.
  %wdp = md_dist.sum_pairs %nd, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %left = vector.shuffle %d, %d [0, 0, 0, 1, 1, 1, 2, 2, 2] : vector<3xf64>, vector<3xf64>
    %right = vector.shuffle %d, %d [0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xf64>, vector<3xf64>
    %dd  = arith.mulf %left, %right : vector<9xf64>
    %sv  = vector.broadcast %s : f64 to vector<9xf64>
    %w   = arith.mulf %sv, %dd : vector<9xf64>
    md_dist.yield %w : vector<9xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<vector<9xf64>>
  %wip = md_dist.sum_pairs %nd, %xo, %xg center_set(%id : !md_dist.field<@dd, owned, i32>) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %left = vector.shuffle %d, %d [0, 0, 0, 1, 1, 1, 2, 2, 2] : vector<3xf64>, vector<3xf64>
    %right = vector.shuffle %d, %d [0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xf64>, vector<3xf64>
    %dd  = arith.mulf %left, %right : vector<9xf64>
    %sv  = vector.broadcast %s : f64 to vector<9xf64>
    %w   = arith.mulf %sv, %dd : vector<9xf64>
    md_dist.yield %w : vector<9xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<vector<9xf64>>
  %wi = md_dist.all_reduce %wip : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
  %wbp = md_dist.sum_pairs %nd, %xo, %xg center_set(%bd : !md_dist.field<@dd, owned, i32>) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %left = vector.shuffle %d, %d [0, 0, 0, 1, 1, 1, 2, 2, 2] : vector<3xf64>, vector<3xf64>
    %right = vector.shuffle %d, %d [0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xf64>, vector<3xf64>
    %dd  = arith.mulf %left, %right : vector<9xf64>
    %sv  = vector.broadcast %s : f64 to vector<9xf64>
    %w   = arith.mulf %sv, %dd : vector<9xf64>
    md_dist.yield %w : vector<9xf64>
  } : !md_dist.relation<@dd, directed>, !own, !gho -> !md_dist.partial<vector<9xf64>>
  %wb = md_dist.all_reduce %wbp : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
  %wf = arith.addf %wi, %wb : vector<9xf64>
  %wd = md_dist.all_reduce %wdp : !md_dist.partial<vector<9xf64>> -> vector<9xf64>

  // Unique: every pair once, the forces of the ghosts returned by G^T.
  %nu = md_dist.neighborhood %xo, %xg, %h cutoff(2.0)
          : !own, !gho, !halo -> !md_dist.relation<@dd, unique>
  %iu, %bu = md_dist.partition_centers %nu, %lay, %h
      : !md_dist.relation<@dd, unique>, !md_dist.layout<@dd>, !halo
      -> !md_dist.field<@dd, owned, i32>, !md_dist.field<@dd, owned, i32>
  %uup = md_dist.sum_pairs %nu, %xo, %xg exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %t   = arith.subf %s12, %s6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %k   = arith.mulf %e4, %t : f64
    md_dist.yield %k : f64
  } : !md_dist.relation<@dd, unique>, !own, !gho -> !md_dist.partial<f64>
  %uu = md_dist.all_reduce %uup : !md_dist.partial<f64> -> f64
  %fuo, %fug = md_dist.gather_pairs %nu, %xo, %xg exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %c4  = arith.constant 4.0 : f64
    %c6  = arith.constant 6.0 : f64
    %c12 = arith.constant 12.0 : f64
    %i6  = arith.constant 6 : i32
    %sr  = arith.divf %sigma, %r : f64
    %s6  = math.fpowi %sr, %i6 : f64, i32
    %s12 = arith.mulf %s6, %s6 : f64
    %a12 = arith.mulf %c12, %s12 : f64
    %a6  = arith.mulf %c6, %s6 : f64
    %a   = arith.subf %a12, %a6 : f64
    %e4  = arith.mulf %c4, %eps : f64
    %num = arith.mulf %e4, %a : f64
    %r2  = arith.mulf %r, %r : f64
    %s   = arith.divf %num, %r2 : f64
    %sv  = vector.broadcast %s : f64 to vector<3xf64>
    %f   = arith.mulf %sv, %d : vector<3xf64>
    md_dist.yield %f : vector<3xf64>
  } : !md_dist.relation<@dd, unique>, !own, !gho
      -> !md_dist.field<@dd, owned, partial, 3 x f64>,
         !md_dist.field<@dd, ghost, partial, 3 x f64>
  %fuc = md_dist.halo_sum %fuo, %fug, %h
           : !md_dist.field<@dd, owned, partial, 3 x f64>,
             !md_dist.field<@dd, ghost, partial, 3 x f64>, !halo -> !own
  %fu = md_dist.collect %fuc, %lay : !own -> !vec

  // The adjoint identity <G a, b> = <a, G^T b>, with a the owned
  // positions and b the ghost positions with their images, which differ
  // between the images of one particle.
  %ga = md_dist.halo_copy %xo, %h : !own, !halo -> !gho
  %lhsp = md_dist.sum_particles gather(%ga, %xg : !gho, !gho) {
  ^bb0(%p: vector<3xf64>, %q: vector<3xf64>):
    %pq = arith.mulf %p, %q : vector<3xf64>
    %s = vector.reduction <add>, %pq : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %lhs = md_dist.all_reduce %lhsp : !md_dist.partial<f64> -> f64
  %z = md_dist.zeros %lay : !md_dist.layout<@dd> -> !own
  %gtb = md_dist.halo_sum %z, %xg, %h : !own, !gho, !halo -> !own
  %rhsp = md_dist.sum_particles gather(%xo, %gtb : !own, !own) {
  ^bb0(%p: vector<3xf64>, %q: vector<3xf64>):
    %pq = arith.mulf %p, %q : vector<3xf64>
    %s = vector.reduction <add>, %pq : vector<3xf64> into f64
    md_dist.yield %s : f64
  } : !md_dist.partial<f64>
  %rhs = md_dist.all_reduce %rhsp : !md_dist.partial<f64> -> f64

  // The reference, on one domain.
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec
  %u, %f, %w = md.evaluate @lj(%x, %cell, %eps, %sigma)
      request [energy, forces, virial]
      : (!vec, !md.cell, f64, f64) -> (f64, !vec, vector<9xf64>)

  %tol = arith.constant 1.0e-10 : f64
  %fine = arith.constant 1.0e-12 : f64
  %fr = mdrt.to_buffer %f : !vec to memref<?x3xf64>
  %fdb = mdrt.to_buffer %fd : !vec to memref<?x3xf64>
  %ffdb = mdrt.to_buffer %ffd : !vec to memref<?x3xf64>
  %fub = mdrt.to_buffer %fu : !vec to memref<?x3xf64>

  // The energy on one domain, and against Inputs/lj_reference.py with the
  // switch taken out (SWITCH_FROM = CUTOFF): -226.88590520072805.
  // CHECK: -226.886
  // CHECK-NEXT: 1
  call @printF64(%u) : (f64) -> ()
  call @printNewline() : () -> ()
  %u_ref = arith.constant -226.88590520072805 : f64
  call @check(%u, %u_ref, %u_ref, %tol) : (f64, f64, f64, f64) -> ()
  // Directed: energy, forces, virial.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%uf, %ud, %ud, %tol) : (f64, f64, f64, f64) -> ()
  call @check(%ud, %u, %u, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%ffdb, %fdb) : (memref<?x3xf64>, memref<?x3xf64>) -> ()
  call @check_forces(%fdb, %fr) : (memref<?x3xf64>, memref<?x3xf64>) -> ()
  call @check_virial(%wf, %wd) : (vector<9xf64>, vector<9xf64>) -> ()
  call @check_virial(%wd, %w) : (vector<9xf64>, vector<9xf64>) -> ()
  // Unique: energy and forces.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%uu, %u, %u, %tol) : (f64, f64, f64, f64) -> ()
  call @check_forces(%fub, %fr) : (memref<?x3xf64>, memref<?x3xf64>) -> ()
  // The adjoint identity.
  // CHECK-NEXT: 1
  call @check(%lhs, %rhs, %rhs, %fine) : (f64, f64, f64, f64) -> ()
  return
}

// TRACE: domain 0 owns 7 rows, holds 136 ghosts
// TRACE: domain 1 owns 0 rows, holds 165 ghosts
// TRACE: domain 2 owns 0 rows, holds 165 ghosts
// TRACE: domain 15 owns 9 rows, holds 133 ghosts
// TRACE: halo_copy of 3 components
// TRACE: halo_sum of 3 components
