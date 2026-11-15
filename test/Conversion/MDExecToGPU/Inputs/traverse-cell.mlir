// The buffer form of md_alg.traverse_pairs with its cell as `!md.cell`,
// which the serial definition does not take (traverse-invalid.test).
!i = memref<?xi32>
func.func @cell(%e: !i, %np: index, %nu: index, %x: memref<?x3xf64>,
                %cell: !md.cell, %u0: f64) -> f64 {
  %u = md_alg.traverse_pairs
      lists(%e, %e, %e, %e, %e, %e, %e : !i, !i, !i, !i, !i, !i, !i)
      sizes(%np, %nu)
      contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
      reach(1.0) positions(%x : memref<?x3xf64>) cell(%cell : !md.cell)
      reduce(%u0 : f64) unique visit unordered combine atomic
      algebra ["any_order"] exchange ["symmetric"] cutoff(0.9) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_alg.yield %r2 : f64
  } -> f64
  return %u : f64
}
