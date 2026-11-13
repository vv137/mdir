// The buffers of md-alg-contract.mlir in the memory of a device, for its
// run through the emitters (convert-md-alg-to-gpu, D406).

!buf = memref<?xi32, 1>
!wide = memref<?xi64, 1>

func.func @alloc(%n: index) -> !buf {
  %b = gpu.alloc (%n) : !buf
  return %b : !buf
}

func.func @alloc_wide(%n: index) -> !wide {
  %b = gpu.alloc (%n) : !wide
  return %b : !wide
}

func.func @upload(%h: memref<?xi32>) -> !buf {
  %c0 = arith.constant 0 : index
  %n = memref.dim %h, %c0 : memref<?xi32>
  %b = gpu.alloc (%n) : !buf
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %b, %h : !buf, memref<?xi32>
  gpu.wait [%t1]
  return %b : !buf
}

func.func @download(%b: !buf) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %n = memref.dim %b, %c0 : !buf
  %h = memref.alloc (%n) : memref<?xi32>
  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %h, %b : memref<?xi32>, !buf
  gpu.wait [%t1]
  return %h : memref<?xi32>
}
