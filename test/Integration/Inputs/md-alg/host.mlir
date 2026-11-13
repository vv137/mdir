// The buffers of md-alg-contract.mlir in the memory of the host, for its
// serial run (convert-md-alg-to-scf, D[md-alg]).

!buf = memref<?xi32>
!wide = memref<?xi64>

func.func @alloc(%n: index) -> !buf {
  %b = memref.alloc (%n) : !buf
  return %b : !buf
}

func.func @alloc_wide(%n: index) -> !wide {
  %b = memref.alloc (%n) : !wide
  return %b : !wide
}

func.func @upload(%h: memref<?xi32>) -> !buf {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %h, %c0 : memref<?xi32>
  %b = memref.alloc (%n) : !buf
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %h[%i] : memref<?xi32>
    memref.store %v, %b[%i] : !buf
  }
  return %b : !buf
}

func.func @download(%b: !buf) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %b, %c0 : !buf
  %h = memref.alloc (%n) : memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %b[%i] : !buf
    memref.store %v, %h[%i] : memref<?xi32>
  }
  return %h : memref<?xi32>
}
