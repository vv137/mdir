// The refusals of the uncarrying of lists of units in
// convert-md-exec-to-gpu (#898): a loop carries one list, its argument
// refreshed where it is (md-exec-reuse-neighbors's form, D419); a loop that
// yields another list, and a list passed through the results of scf.if or
// scf.while, are refused naming the op, not rewired to the loop's init or
// left to fail later for want of storage.
//
// RUN: mdir-opt %s --split-input-file --convert-md-exec-to-gpu --verify-diagnostics

// The loop yields a list other than its argument.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @other(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %e1 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %e2 = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{yields a list of units as its value 0 that is not its argument through refreshes of it; a loop carries one list, refreshed where it is (md-exec-reuse-neighbors's form)}}
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%l = %e1) -> (!sp) {
    %m = md_exec.refresh_segmented_pairs %e2 at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
    scf.yield %m : !sp
  }
  return
}

// -----

// The list passes through the results of scf.if.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @branch(%c: i1, %t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell) {
  %e = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{'scf.if' op passes a list of units through its results or its block arguments; only scf.for carries one, refreshed where it is (md-exec-reuse-neighbors's form)}}
  %l = scf.if %c -> (!sp) {
    %m = md_exec.refresh_segmented_pairs %e at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
    scf.yield %m : !sp
  } else {
    scf.yield %e : !sp
  }
  return
}

// -----

// The list passes through the values of scf.while.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @while(%c: i1, %t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell) {
  %e = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  // expected-error @+1 {{'scf.while' op passes a list of units through its results or its block arguments; only scf.for carries one, refreshed where it is (md-exec-reuse-neighbors's form)}}
  %l = scf.while (%a = %e) : (!sp) -> (!sp) {
    scf.condition(%c) %a : !sp
  } do {
  ^bb0(%b: !sp):
    %m = md_exec.refresh_segmented_pairs %b at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
    scf.yield %m : !sp
  }
  return
}

// -----

// Accepted: the loop yields its argument through two refreshes of it.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @chain(%t: memref<?xi32, 1>, %k: memref<?x3xi32, 1>, %x: memref<?x3xf64, 1>, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %e = md_exec.empty_segmented_pairs %t kinds(%k : memref<?x3xi32, 1>) cutoff(0.9) skin(0.1) : memref<?xi32, 1> -> !sp
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%l = %e) -> (!sp) {
    %m = md_exec.refresh_segmented_pairs %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
    %p = md_exec.refresh_segmented_pairs %m at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.1) : !sp, memref<?x3xf64, 1>
    scf.yield %p : !sp
  }
  return
}
