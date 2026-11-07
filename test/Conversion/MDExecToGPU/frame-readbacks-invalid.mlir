// RUN: mdir-opt %s --convert-md-exec-to-gpu --split-input-file --verify-diagnostics
// A frame batch refuses an unconverted host consumer of a result.
func.func @escape_sum(%sums: memref<1xf64, 1>, %host: memref<1xf64>) -> f64
    attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  gpu.memcpy %host, %sums : memref<1xf64>, memref<1xf64, 1> {mdir.readback = 1 : i64}
  // expected-error @+1 {{device frame result still requires a host readback}}
  %value = memref.load %host[%c0] {mdir.readback = 1 : i64} : memref<1xf64>
  return %value : f64
}

// -----
// A join after the new reader cannot repair a source already released at
// the old host copy's last use. Refuse that case rather than load it.
func.func private @mdrtSideBegin()
func.func private @mdrtSideEnd()
func.func @released_before_reader(%host: memref<1xf64>,
    %out: memref<1xf64, 1>) attributes {mdir.frame_batch} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sums = gpu.alloc () : memref<1xf64, 1>
  %alias = memref.memory_space_cast %sums : memref<1xf64, 1> to memref<1xf64>
  gpu.memcpy %host, %sums : memref<1xf64>, memref<1xf64, 1> {mdir.readback = 2 : i64}
  %value = memref.load %host[%c0] {mdir.readback = 2 : i64} : memref<1xf64>
  gpu.dealloc %alias : memref<1xf64>
  func.call @mdrtSideBegin() : () -> ()
  // expected-error @+1 {{a side-stream frame readback source may be released before its reader}}
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %value, %out[%c0] : memref<1xf64, 1>
    gpu.terminator
  }
  func.call @mdrtSideEnd() : () -> ()
  return
}
