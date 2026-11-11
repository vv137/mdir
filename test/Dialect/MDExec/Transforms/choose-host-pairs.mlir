// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="cpu=znver4 vector-bits=512 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes vector-bits=256" \
// RUN: | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="cpu=znver3 vector-bits=256" \
// RUN: | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes vector-bits=256" \
// RUN: | mdir-opt | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes vector-bits=512" \
// RUN: | FileCheck %s --check-prefix=WIDE
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --md-exec-choose-host-pairs="mode=lanes vector-bits=512" \
// RUN: | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --md-exec-choose-host-pairs="mode=lanes vector-bits=128" \
// RUN: | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --md-exec-choose-host-pairs="cpu=unmeasured vector-bits=128" \
// RUN: | FileCheck %s --check-prefix=AVX
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --md-exec-choose-host-pairs="mode=scalar" \
// RUN: | FileCheck %s --check-prefix=OFF
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes width=2" \
// RUN: | FileCheck %s --check-prefix=TWO
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes vector-bits=256" \
// RUN:   --md-exec-assign-storage | FileCheck %s --check-prefix=STORAGE
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=lanes vector-bits=256 explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHY
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-choose-host-pairs="mode=scalar explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR
// RUN: mdir-opt %s --md-exec-choose-host-pairs="mode=lanes explain=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOALGEBRA
// RUN: not mdir-opt %s --md-exec-choose-host-pairs=mode=fast 2>&1 \
// RUN: | FileCheck %s --check-prefix=MODE
// RUN: not mdir-opt %s --md-exec-choose-host-pairs="mode=lanes width=3" 2>&1 \
// RUN: | FileCheck %s --check-prefix=WIDTH
// RUN: not mdir-opt %s --md-exec-choose-host-pairs="mode=lanes width=128" 2>&1 \
// RUN: | FileCheck %s --check-prefix=WIDE-WIDTH
// RUN: not mdir-opt %s --md-exec-choose-host-pairs="mode=lanes width=-4" 2>&1 \
// RUN: | FileCheck %s --check-prefix=NEGATIVE-WIDTH
// RUN: not mdir-opt %s --md-exec-choose-host-pairs="vector-bits=0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=BITS

// The rows of the loops over pairs on the host take `host(lanes(W))`
// (D[host-pair-vector]) where the precondition of HostPairs.h holds and the
// mode selects it. W is the host's register over the bits of the kernel's
// type (8 in f32 and 4 in f64 for 256 bits, 16 and 8 for 512), or, in the
// deterministic mode, 8 and 4 whatever the host. `mode=auto` takes the
// lanes where the host's policy does (`znver3`, measured) and keeps the
// scalar row on a host that was not measured, except in the deterministic
// mode, which takes the lanes on every host; `mode=scalar` takes none.

!vec = !md.field<@atoms, 3 x f32>
!vec64 = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f32>
!type = !md.field<@atoms, i32>
!nl = !mdrt.neighbors<@atoms>
md.particle_set @atoms

// OFF-NOT: host(
// AUTO: remark: host pairs: scalar: the host 'znver4' has no measured policy of lanes, which keeps the scalar row (mode=auto)
// SCALAR: remark: host pairs: scalar: mode=scalar
// MODE: error: expected 'auto', 'lanes', or 'scalar' for 'mode', got 'fast'
// WIDTH: error: expected a width of 0 or of lanes: the lanes are a power of two from 1 to 64, not 3
// WIDE-WIDTH: error: expected a width of 0 or of lanes: the lanes are a power of two from 1 to 64, not 128
// NEGATIVE-WIDTH: error: expected a width of 0 or of lanes: the lanes are a power of two from 1 to 64, not -4
// BITS: error: expected vector-bits of at least 1
// NOALGEBRA: remark: host pairs: scalar: no lanes(4): states no algebra of the values it reduces; run 'md-exec-assign-algebra'

// A Lennard-Jones loop in f32 with the forces, the energy widened to f64,
// and a field of charges.
// AVX-LABEL: func.func @lj(
// AVX: policy(directed, owner_only) host(lanes(8)) {
// AVX: policy(directed, owner_only) host(lanes(4)) {
// WIDE-LABEL: func.func @lj(
// WIDE: policy(directed, owner_only) host(lanes(16)) {
// WIDE: policy(directed, owner_only) host(lanes(8)) {
// TWO-LABEL: func.func @lj(
// TWO: policy(directed, owner_only) host(lanes(2)) {
// TWO: policy(directed, owner_only) host(lanes(2)) {
// STORAGE-LABEL: func.func @lj(
// STORAGE: md_exec.pair_for {{.*}} policy(directed, owner_only) host(lanes(8)) {
// STORAGE: } {{.*}}memref<?x3xf32
// STORAGE: md_exec.pair_for {{.*}} policy(directed, owner_only) host(lanes(4)) {
// STORAGE: } {{.*}}memref<?x3xf64
// WHY: remark: host pairs: lanes(8)
// WHY: remark: host pairs: lanes(4)
func.func @lj(%x: !vec, %x64: !vec64, %q: !real, %cell: !md.cell)
    -> (!vec, f64, f64) {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f0 = md_exec.zeros : !vec
  %u0 = arith.constant 0.0 : f64
  %f, %u = md_exec.pair_for %nl, %x, %cell ins(%q : !real) outs(%f0 : !vec)
      reduce(%u0 : f64) cutoff(1.0) weights [0.5]
      exchange [antisymmetric, symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
    %one = arith.constant 1.0 : f32
    %c3 = arith.constant 3 : i32
    %inv = arith.divf %one, %r2 : f32
    %s6 = math.fpowi %inv, %c3 : f32, i32
    %qq = arith.mulf %qi, %qj : f32
    %e = arith.mulf %s6, %qq : f32
    %b = vector.broadcast %e : f32 to vector<3xf32>
    %fd = arith.mulf %b, %d : vector<3xf32>
    %wide = arith.extf %e : f32 to f64
    md_exec.yield %fd, %wide : vector<3xf32>, f64
  } : !nl, !vec -> !vec, f64

  %cells64 = md_exec.build_cells %x64, %cell width(1.0)
      : !vec64 -> !mdrt.cells<@atoms>
  %nl64 = md_exec.build_neighbors %cells64, %x64, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec64 -> !nl
  %v0 = arith.constant 0.0 : f64
  %v = md_exec.pair_for %nl64, %x64, %cell reduce(%v0 : f64) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %e = math.exp %r2 : f64
    md_exec.yield %e : f64
  } : !nl, !vec64 -> f64
  return %f, %u, %v : !vec, f64, f64
}

// The loops that the first slice refuses keep the scalar row.
// AVX-LABEL: func.func @refused(
// AVX-NOT: host(
// AVX: return
// WHY: remark: host pairs: scalar: no lanes(8): the policy is not (directed, owner_only): the lanes write at the centers alone
// WHY: remark: host pairs: scalar: no lanes(8): the exponent of 'math.fpowi' is not a scalar constant
// WHY: remark: host pairs: scalar: no lanes(8): the kernel holds 'scf.if', which has no lane mapping
// WHY: remark: host pairs: scalar: no lanes(8): the kernel holds 'vector.reduction', which has no lane mapping
func.func @refused(%x: !vec, %t: !type, %cell: !md.cell) -> (!vec, f32, f32, f32) {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %f0 = md_exec.zeros : !vec
  // Two-ended: it adds at the neighbor as well (D290).
  %f = md_exec.pair_for %nl, %x, %cell outs(%f0 : !vec) cutoff(1.0)
      exchange [antisymmetric] policy(directed, atomic) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %d : vector<3xf32>
  } : !nl, !vec -> !vec
  %z = arith.constant 0.0 : f32
  // A power by a field of the pair.
  %p = md_exec.pair_for %nl, %x, %cell ins(%t : !type) reduce(%z : f32)
      cutoff(1.0) exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>, %ti: i32, %tj: i32):
    %e = math.fpowi %r2, %tj : f32, i32
    md_exec.yield %e : f32
  } : !nl, !vec -> f32
  // A region in the kernel.
  %c = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %half = arith.constant 0.5 : f32
    %near = arith.cmpf olt, %r2, %half : f32
    %e = scf.if %near -> f32 {
      scf.yield %r2 : f32
    } else {
      scf.yield %half : f32
    }
    md_exec.yield %e : f32
  } : !nl, !vec -> f32
  // A reduction of a vector inside the kernel.
  %s = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = vector.reduction <add>, %d : vector<3xf32> into f32
    md_exec.yield %e : f32
  } : !nl, !vec -> f32
  return %f, %p, %c, %s : !vec, f32, f32, f32
}

// The loops over a domain, a power by a vector of exponents, and a value
// that is not a number or a vector of one dimension keep the scalar row. The
// lanes always take the minimum image, so a loop whose rows carry their
// images (images(explicit), which the distributed lowering builds) must not
// take them. The types of the kernel's arguments, f32 or f64 and numbers or
// vectors of one dimension, are the verifier's (host-pairs-invalid.mlir).
// AVX-LABEL: func.func @domain(
// AVX-NOT: host(
// AVX: return
// WHY: remark: host pairs: scalar: no lanes(8): the loop is over a domain (centers, center_set, ghost_free, or images(explicit)), which the first slice does not take
// WHY: remark: host pairs: scalar: no lanes(8): the loop is over a domain (centers, center_set, ghost_free, or images(explicit)), which the first slice does not take
// WHY: remark: host pairs: scalar: no lanes(8): the exponent of 'math.fpowi' is not a scalar constant
// WHY: remark: host pairs: scalar: no lanes(8): a value of 'arith.constant' is neither a number nor a vector of one dimension
func.func @domain(%x: !vec, %cell: !md.cell, %n: index) -> (f32, f32, f32, f32) {
  %cells = md_exec.build_cells %x, %cell width(1.0)
      : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %centered = md_exec.build_neighbors %cells, %x, %cell centers(%n)
      cutoff(1.0) skin(0.0) kind(matrix) width(48)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %imaged = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(48) images(explicit)
      : !mdrt.cells<@atoms>, !vec -> !nl
  %z = arith.constant 0.0 : f32
  %a = md_exec.pair_for %centered, %x, %cell centers(%n) reduce(%z : f32)
      cutoff(1.0) exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !nl, !vec -> f32
  %b = md_exec.pair_for %imaged, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !nl, !vec -> f32
  %c = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %k = arith.constant dense<3> : vector<3xi32>
    %p = math.fpowi %d, %k : vector<3xf32>, vector<3xi32>
    %e = vector.extract %p[0] : f32 from vector<3xf32>
    md_exec.yield %e : f32
  } : !nl, !vec -> f32
  %e2 = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %m = arith.constant dense<1.0> : vector<2x2xf32>
    md_exec.yield %r2 : f32
  } : !nl, !vec -> f32
  return %a, %b, %c, %e2 : f32, f32, f32, f32
}
