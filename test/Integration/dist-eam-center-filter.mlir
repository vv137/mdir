// D[distributed-pair-center-filter]: both EAM pair terms use a proved
// interior/boundary partition. Four particles form a periodic one-dimensional
// chain at spacings 0.8 and 1.6 in a 4.8 box. The grid has interior and
// boundary centers; the second run leaves empty domains.
//
// The independent analytic energy is
// 4[-sqrt((2-0.8)^2 + (2-1.6)^2)] + 2[phi(0.8) + phi(1.6)]
// = -0.026684256269407847, with phi(r) = (r-2)^2(3-2r+0.4r^2).
// Forces and virial are also compared to the one-domain EAM potential.
//
// RUN: sed 's/GRID/2, 1, 1/; s/TOLERANCE/1.0e-10/' %s \
// RUN: | mdir-opt %md_passes --md-exec-assign-algebra --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

// RUN: sed 's/GRID/8, 1, 1/; s/TOLERANCE/1.0e-5/' %s \
// RUN: | mdir-opt %md_passes --md-exec-assign-algebra \
// RUN:     --convert-md-dist-to-md-exec="functions=main" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN:     --md-exec-assign-precision="mode=mixed" \
// RUN:     --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/2, 1, 1/; s/TOLERANCE/1.0e-10/' %s \
// RUN: | mdir-opt %md_passes --md-exec-assign-algebra \
// RUN:     --convert-md-dist-to-md-exec="functions=main" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// RUN: sed 's/GRID/8, 1, 1/; s/TOLERANCE/1.0e-10/' %s \
// RUN: | mdir-opt %md_passes --md-exec-assign-algebra \
// RUN:     --convert-md-dist-to-md-exec="functions=main" \
// RUN:     --convert-md-dist-to-loops \
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s

!vec = !md.field<@atoms, 3 x f64>
!sca = !md.field<@atoms, f64>
!pairs = !md.relation<@atoms, 2, unordered>
!own = !md_dist.field<@dd, owned, 3 x f64>
!ghost = !md_dist.field<@dd, ghost, 3 x f64>
!rho = !md_dist.field<@dd, owned, f64>
!ro = !md_dist.field<@dd, owned, partial, f64>
!rg = !md_dist.field<@dd, ghost, partial, f64>
!mask = !md_dist.field<@dd, owned, i32>
!halo = !md_dist.halo<@dd, 2.0>
!relation = !md_dist.relation<@dd, unique>
!buffer = memref<?x3xf64>

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([GRID])

md.potential @filtered(%x: !vec, %cell: !md.cell) -> f64 {
  %layout = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %owned = md_dist.distribute %x, %layout image(%cell) : !vec -> !own
  %h = md_dist.halo %layout, %owned, %cell : !own -> !halo
  %ghost = md_dist.halo_copy %owned, %h image(%cell) : !own, !halo -> !ghost
  %n = md_dist.neighborhood %owned, %ghost, %h cutoff(2.0)
      : !own, !ghost, !halo -> !relation
  %interior, %boundary = md_dist.partition_centers %n, %layout, %h
      : !relation, !md_dist.layout<@dd>, !halo -> !mask, !mask
  %rio, %rig = md_dist.gather_pairs %n, %owned, %ghost
      center_set(%interior : !mask) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %t = arith.subf %r, %two : f64
    %v = arith.mulf %t, %t : f64
    md_dist.yield %v : f64
  } : !relation, !own, !ghost -> !ro, !rg
  %ri = md_dist.halo_sum %rio, %rig, %h : !ro, !rg, !halo -> !rho
  %rbo, %rbg = md_dist.gather_pairs %n, %owned, %ghost
      center_set(%boundary : !mask) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %t = arith.subf %r, %two : f64
    %v = arith.mulf %t, %t : f64
    md_dist.yield %v : f64
  } : !relation, !own, !ghost -> !ro, !rg
  %rb = md_dist.halo_sum %rbo, %rbg, %h : !ro, !rg, !halo -> !rho
  %density = md_dist.map_particles gather(%ri, %rb : !rho, !rho) {
  ^bb0(%a: f64, %b: f64):
    %s = arith.addf %a, %b : f64
    md_dist.yield %s : f64
  } : !rho
  %ep = md_dist.sum_particles gather(%density : !rho) {
  ^bb0(%v: f64):
    %s = math.sqrt %v : f64
    %e = arith.negf %s : f64
    md_dist.yield %e : f64
  } : !md_dist.partial<f64>
  %embed = md_dist.all_reduce %ep : !md_dist.partial<f64> -> f64
  %pip = md_dist.sum_pairs %n, %owned, %ghost
      center_set(%interior : !mask) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %minus2 = arith.constant -2.0 : f64
    %c = arith.constant 0.4 : f64
    %t = arith.subf %r, %two : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %minus2, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %c, %r2 : f64
    %s = arith.addf %three, %a : f64
    %q = arith.addf %s, %b : f64
    %v = arith.mulf %t2, %q : f64
    md_dist.yield %v : f64
  } : !relation, !own, !ghost -> !md_dist.partial<f64>
  %pi = md_dist.all_reduce %pip : !md_dist.partial<f64> -> f64
  %pbp = md_dist.sum_pairs %n, %owned, %ghost
      center_set(%boundary : !mask) exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %minus2 = arith.constant -2.0 : f64
    %c = arith.constant 0.4 : f64
    %t = arith.subf %r, %two : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %minus2, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %c, %r2 : f64
    %s = arith.addf %three, %a : f64
    %q = arith.addf %s, %b : f64
    %v = arith.mulf %t2, %q : f64
    md_dist.yield %v : f64
  } : !relation, !own, !ghost -> !md_dist.partial<f64>
  %pb = md_dist.all_reduce %pbp : !md_dist.partial<f64> -> f64
  %pair = arith.addf %pi, %pb : f64
  %energy = arith.addf %embed, %pair : f64
  md.return %energy : f64
}

md.potential @reference(%x: !vec, %cell: !md.cell) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(2.0) : !vec -> !pairs
  %rho = md.gather_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %t = arith.subf %r, %two : f64
    %v = arith.mulf %t, %t : f64
    md.yield %v : f64
  } : !pairs, !vec -> !sca
  %embed = md.sum_particles gather(%rho : !sca) {
  ^bb0(%v: f64):
    %s = math.sqrt %v : f64
    %e = arith.negf %s : f64
    md.yield %e : f64
  } : f64
  %pair = md.sum_relation %n, %x, %cell exchange(symmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %minus2 = arith.constant -2.0 : f64
    %c = arith.constant 0.4 : f64
    %t = arith.subf %r, %two : f64
    %t2 = arith.mulf %t, %t : f64
    %a = arith.mulf %minus2, %r : f64
    %r2 = arith.mulf %r, %r : f64
    %b = arith.mulf %c, %r2 : f64
    %s = arith.addf %three, %a : f64
    %q = arith.addf %s, %b : f64
    %v = arith.mulf %t2, %q : f64
    md.yield %v : f64
  } : !pairs, !vec -> f64
  %energy = arith.addf %embed, %pair : f64
  md.return %energy : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %scale: f64) {
  %diff = arith.subf %value, %reference : f64
  %abs = math.absf %diff : f64
  %tol = arith.constant TOLERANCE : f64
  %bound = arith.mulf %tol, %scale : f64
  %ok = arith.cmpf ole, %abs, %bound : f64
  %flag = arith.uitofp %ok : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %n = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f64
  %a = arith.constant 0.2 : f64
  %b = arith.constant 1.0 : f64
  %c = arith.constant 2.6 : f64
  %d = arith.constant 3.4 : f64
  %edge = arith.constant 4.8 : f64
  %one = arith.constant 1.0 : f64
  %analytic = arith.constant -0.026684256269407847 : f64
  %x = memref.alloc(%n) : !buffer
  memref.store %a, %x[%c0, %c0] : !buffer
  memref.store %b, %x[%c1, %c0] : !buffer
  memref.store %c, %x[%c2, %c0] : !buffer
  memref.store %d, %x[%c3, %c0] : !buffer
  scf.for %i = %c0 to %n step %c1 {
    memref.store %zero, %x[%i, %c1] : !buffer
    memref.store %zero, %x[%i, %c2] : !buffer
  }
  %positions = mdrt.from_buffer %x : !buffer to !vec
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %u, %f, %w = md.evaluate @filtered(%positions, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  %ur, %fr, %wr = md.evaluate @reference(%positions, %cell)
      request [energy, forces, virial]
      : (!vec, !md.cell) -> (f64, !vec, vector<9xf64>)
  // CHECK: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  call @check(%ur, %analytic, %one) : (f64, f64, f64) -> ()
  call @check(%u, %ur, %one) : (f64, f64, f64) -> ()
  %fb = mdrt.to_buffer %f : !vec to !buffer
  %frb = mdrt.to_buffer %fr : !vec to !buffer
  %max = scf.for %i = %c0 to %n step %c1 iter_args(%m = %zero) -> (f64) {
    %r = scf.for %j = %c0 to %c3 step %c1 iter_args(%v = %m) -> (f64) {
      %x0 = memref.load %fb[%i, %j] : !buffer
      %x1 = memref.load %frb[%i, %j] : !buffer
      %diff = arith.subf %x0, %x1 : f64
      %abs = math.absf %diff : f64
      %next = arith.maximumf %v, %abs : f64
      scf.yield %next : f64
    }
    scf.yield %r : f64
  }
  call @check(%max, %zero, %one) : (f64, f64, f64) -> ()
  %dv = arith.subf %w, %wr : vector<9xf64>
  %av = math.absf %dv : vector<9xf64>
  %vw = vector.reduction <maximumf>, %av : vector<9xf64> into f64
  call @check(%vw, %zero, %one) : (f64, f64, f64) -> ()
  return
}
