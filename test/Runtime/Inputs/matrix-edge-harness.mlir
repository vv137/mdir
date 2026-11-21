func.func private @printI64(i64)
func.func private @printNewline()
func.func @model(%x: memref<?x3xf64>, %slots: memref<?x5xi32>,
                 %counts: memref<?xi32>, %offsets: memref<?xi32>,
                 %builds: i64, %forces: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i64
  %n = memref.dim %counts, %c0 : memref<?xi32>
  %host = memref.alloc(%n) : memref<?xi32>
  %t = gpu.wait async
  %copy = gpu.memcpy async [%t] %host, %counts : memref<?xi32>, memref<?xi32>
  gpu.wait [%copy]
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> i64 {
    %count = memref.load %host[%i] : memref<?xi32>
    %wide = arith.extui %count : i32 to i64
    %next = arith.addi %s, %wide : i64
    scf.yield %next : i64
  }
  func.call @printI64(%sum) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @main() {
  %n = arith.constant 45 : index
  %side = arith.constant 4.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %spacing = arith.constant 0.005 : f64
  %host = memref.alloc(%n) : memref<?x3xf64>
  scf.for %i = %c0 to %n step %c1 {
    %iw = arith.index_cast %i : index to i64
    %r = arith.sitofp %iw : i64 to f64
    %coord = arith.mulf %r, %spacing : f64
    scf.for %d = %c0 to %c3 step %c1 {
      memref.store %coord, %host[%i, %d] : memref<?x3xf64>
    }
  }
  %x = gpu.alloc (%n) : memref<?x3xf64, 1>
  %f = gpu.alloc (%n) : memref<?x3xf64, 1>
  %t = gpu.wait async
  %copy = gpu.memcpy async [%t] %x, %host : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%copy]
  %out = func.call @step(%x, %f, %side, %side, %side) : (memref<?x3xf64, 1>, memref<?x3xf64, 1>, f64, f64, f64) -> memref<?x3xf64, 1>
  return
}
