// A type-pair spring tests a captured coefficient table and a per-atom input.
// The test checks both endpoints against an independent all-pairs loop.
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
func.func private @spring(%nl: !nl, %x: memref<?x3xf64>, %box: !md.cell,
                          %q: memref<?xf32>, %types: memref<?xi32>, %k: memref<2x2xf32>,
                          %f: memref<?x3xf32>) {
  %built = md_exec.refresh_neighbors %nl, %x, %box cutoff(1.0) skin(0.0) cell_width(1.0) policy(always) : !nl, memref<?x3xf64>
  md_exec.pair_for %built, %x, %box ins(%q, %types : memref<?xf32>, memref<?xi32>) outs(%f : memref<?x3xf32>)
      cutoff(1.0) exchange [antisymmetric] policy(directed, owner_only) {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qa: f32, %qb: f32, %ta: i32, %tb: i32):
    %a = arith.index_cast %ta : i32 to index
    %b = arith.index_cast %tb : i32 to index
    %spring = memref.load %k[%a, %b] : memref<2x2xf32>
    %charge = arith.mulf %qa, %qb : f32
    %kq = arith.mulf %spring, %charge : f32
    %v = vector.broadcast %kq : f32 to vector<3xf32>
    %df = arith.mulf %d, %v : vector<3xf32>
    md_exec.yield %df : vector<3xf32>
  } : !nl, memref<?x3xf64>
  return
}
