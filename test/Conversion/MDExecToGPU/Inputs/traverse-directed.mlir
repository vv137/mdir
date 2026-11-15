// A traversal written by hand that visits each pair in both orders, over a
// matrix, in the storage form that convert-md-exec-to-gpu takes
// (traverse-invalid.test): the loops over a matrix stay md_exec.pair_for.
!x = memref<?x3xf64, #md_exec.set_layout<@atoms, 2>, 1>
!s = memref<?xf64, #md_exec.set_layout<@atoms, 1>, 1>
!matrix = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  func.func @directed(%x: !x, %cell: !md.cell, %s0: !s, %s1: !s) -> f64 {
    %c0 = arith.constant 0 : index
    %n = memref.dim %x, %c0 : !x
    %u0 = arith.constant 0.0 : f64
    %nl = md_exec.empty_neighbors size(%n) positions(!x) kind(matrix) width(96) : !matrix
    %r = md_exec.refresh_neighbors %nl, %x, %cell cutoff(2.5) skin(0.3)
        cell_width(2.8) policy(always) : !matrix, !x
    %u = md_alg.traverse_pairs structure %r : !matrix
        positions(%x : !x) cell(%cell : !md.cell) reduce(%u0 : f64)
        scratch(%s0, %s1 : !s, !s)
        directed visit unordered combine atomic
        algebra ["any_order"] exchange ["symmetric"] cutoff(2.5) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_alg.yield %r2 : f64
    } -> f64
    return %u : f64
  }
}
