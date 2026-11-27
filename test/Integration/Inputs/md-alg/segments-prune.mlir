// The buffer form of a loop over segments that prunes in the loop, which
// its serial definition refuses (md-alg-segments-serial.test).
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
func.func @buffers(%b: memref<?xi32>, %n: index, %x: memref<?x3xf64>, %cell: vector<3xf64>) -> f64 {
  %u0 = arith.constant 0.0 : f64
  %u = md_alg.traverse_pairs lists(%b, %b, %b, %b, %b, %b, %b, %b : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%n, %n)
      contract(!sp) reach(1.0)
      positions(%x : memref<?x3xf64>) cell(%cell : vector<3xf64>) reduce(%u0 : f64)
      unique visit unordered combine atomic algebra ["any_order"] exchange ["symmetric"]
      cutoff(0.9) weights [0.5] prune(4096, 1.0)
    segment(0, 0) sites [(0, 0) [0]] {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_alg.yield %r2 : f64
    } -> f64
  return %u : f64
}
