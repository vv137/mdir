// RUN: mdir-opt %s --md-exec-check-transfers -o %t
// RUN: mdir-opt %s -o %t.original
// RUN: diff %t.original %t
// RUN: mdir-opt %S/assign-storage-device.mlir --md-exec-assign-storage="memory=device" --md-exec-check-transfers -o /dev/null
// RUN: mdir-opt %S/../../../Conversion/MDExecToGPU/kernels.mlir --md-exec-assign-algebra --convert-md-exec-to-gpu="contract=false" --md-exec-check-transfers -o /dev/null

// Checking changes no operation or attribute. The conversion fixture
// includes the actual deferred-readback rewrite and reduction copies.

func.func @immediate(%src: memref<4xf64>, %dst: memref<4xf64, 1>) {
  %begin = gpu.wait async
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<4xf64, 1>, memref<4xf64>
  gpu.wait [%copied]
  return
}

func.func @synchronous(%src: memref<4xf64>, %dst: memref<4xf64, 1>) {
  gpu.memcpy %dst, %src : memref<4xf64, 1>, memref<4xf64>
  return
}

func.func @independent_work() {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  %src = memref.alloc() : memref<4xf64>
  %dst = memref.alloc() : memref<4xf64>
  %other = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  %value = memref.load %src[%c0] : memref<4xf64>
  memref.store %zero, %other[%c0] : memref<4xf64>
  gpu.wait [%copied]
  memref.dealloc %src : memref<4xf64>
  memref.dealloc %dst : memref<4xf64>
  memref.dealloc %other : memref<4xf64>
  return
}

func.func @deferred_group() {
  %src = memref.alloc() : memref<4xf64>
  %first = memref.alloc() : memref<4xf64>
  %second = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  %one = gpu.memcpy async [%begin] %first, %src : memref<4xf64>, memref<4xf64>
  %two = gpu.memcpy async [%begin] %second, %src : memref<4xf64>, memref<4xf64>
  %joined = gpu.wait async [%one, %two]
  gpu.wait [%joined]
  memref.dealloc %src : memref<4xf64>
  memref.dealloc %first : memref<4xf64>
  memref.dealloc %second : memref<4xf64>
  return
}

// An explicit copy dependency orders the second copy's memory accesses.
// A blocking wait on the last token completes the whole chain.
func.func @copy_chain() {
  %src = memref.alloc() : memref<4xf64>
  %first = memref.alloc() : memref<4xf64>
  %second = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  %one = gpu.memcpy async [%begin] %first, %src : memref<4xf64>, memref<4xf64>
  %two = gpu.memcpy async [%one] %second, %first : memref<4xf64>, memref<4xf64>
  gpu.wait [%two]
  memref.dealloc %src : memref<4xf64>
  memref.dealloc %first : memref<4xf64>
  memref.dealloc %second : memref<4xf64>
  return
}

func.func @disjoint_slice() {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  %src = memref.alloc() : memref<4xf64>
  %base = memref.alloc() : memref<8xf64>
  %head = memref.subview %base[0] [4] [1] : memref<8xf64> to memref<4xf64, strided<[1]>>
  %tail = memref.subview %base[4] [4] [1] : memref<8xf64> to memref<4xf64, strided<[1], offset: 4>>
  %begin = gpu.wait async
  %copied = gpu.memcpy async [%begin] %head, %src : memref<4xf64, strided<[1]>>, memref<4xf64>
  memref.store %zero, %tail[%c0] : memref<4xf64, strided<[1], offset: 4>>
  gpu.wait [%copied]
  memref.dealloc %src : memref<4xf64>
  memref.dealloc %base : memref<8xf64>
  return
}

func.func private @stop() attributes {dyn.effects = {write = ["dyn.stop"]}}

func.func private @read_source(%src: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %value = memref.load %src[%c0] : memref<4xf64>
  return
}

func.func @checked_calls() {
  %src = memref.alloc() : memref<4xf64>
  %dst = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  func.call @read_source(%src) : (memref<4xf64>) -> ()
  func.call @stop() : () -> ()
  gpu.wait [%copied]
  memref.dealloc %src : memref<4xf64>
  memref.dealloc %dst : memref<4xf64>
  return
}

md.particle_set @atoms

// The storage-form loop uses D342's checked operand/row summary.
func.func @checked_loop() {
  %c4 = arith.constant 4 : index
  %src = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %dst = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %other = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %begin = gpu.wait async
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<?xf64, #md_exec.set_layout<@atoms, 1>>, memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  md_exec.particle_for ins(%src : memref<?xf64, #md_exec.set_layout<@atoms, 1>>)
      outs(%other : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  ^bb0(%value: f64):
    md_exec.yield %value : f64
  }
  gpu.wait [%copied]
  memref.dealloc %src : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  memref.dealloc %dst : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  memref.dealloc %other : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  return
}
