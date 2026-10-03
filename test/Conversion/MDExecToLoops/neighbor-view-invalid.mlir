// RUN: mdir-opt %s --split-input-file --verify-diagnostics
md.particle_set @atoms
func.func @bad_rows(%c: memref<2xi32>, %e: memref<3x4xi32>, %n: index) {
  // expected-error @+1 {{counts and entries must have the same row count}}
  %v = md_exec.neighbor_view %c, %e local_size(%n) : memref<2xi32>, memref<3x4xi32> -> !mdrt.neighbors<@atoms>
  return
}
// -----
md.particle_set @atoms
func.func @bad_type(%c: memref<?xi64>, %e: memref<?x?xi32>, %n: index) {
  // expected-error @+1 {{requires rank-one i32 counts and rank-two i32 entries}}
  %v = md_exec.neighbor_view %c, %e local_size(%n) : memref<?xi64>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>
  return
}
