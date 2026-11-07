// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
func.func @escaping(%field: !vec, %frames: index, %initial: memref<?x3xf64, 1>)
    -> memref<?x3xf64, 1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %frame = %c0 to %frames step %c1
      iter_args(%previous = %initial) -> (memref<?x3xf64, 1>) {
    // expected-error@+1 {{a device field view in a loop must be read in that region}}
    %view = mdrt.to_buffer %field : !vec to memref<?x3xf64, 1>
    scf.yield %view : memref<?x3xf64, 1>
  }
  return %result : memref<?x3xf64, 1>
}
