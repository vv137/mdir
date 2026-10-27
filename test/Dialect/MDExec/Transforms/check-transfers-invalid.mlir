// RUN: mdir-opt %s --split-input-file --md-exec-check-transfers -verify-diagnostics -o /dev/null

func.func @read_destination(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination read before copy completion}}
  %value = memref.load %dst[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @write_destination(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination written before copy completion}}
  memref.store %zero, %dst[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @write_source() {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  %src = memref.alloc() : memref<4xf64>
  %dst = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer source written before copy completion}}
  memref.store %zero, %src[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @free_source() {
  %src = memref.alloc() : memref<4xf64>
  %dst = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer source freed or reused before copy completion}}
  memref.dealloc %src : memref<4xf64>
  %reuse = memref.alloc() : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @free_destination(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination freed or reused before copy completion}}
  memref.dealloc %dst : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @read_cast(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %alias = memref.cast %dst : memref<4xf64> to memref<?xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination read before copy completion}}
  %value = memref.load %alias[%c0] : memref<?xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @write_subview(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  %alias = memref.subview %dst[0] [2] [1] : memref<4xf64> to memref<2xf64, strided<[1]>>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination written before copy completion}}
  memref.store %zero, %alias[%c0] : memref<2xf64, strided<[1]>>
  gpu.wait [%copy]
  return
}

// -----

func.func @free_cast(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %alias = memref.cast %dst : memref<4xf64> to memref<?xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{transfer destination freed or reused before copy completion}}
  memref.dealloc %alias : memref<?xf64>
  gpu.wait [%copy]
  return
}

// -----

// A wait that produces another async token does not make host data ready.
func.func @dominance_only(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  %joined = gpu.wait async [%copy]
  // expected-error @+1 {{transfer destination read before copy completion}}
  %value = memref.load %dst[%c0] : memref<4xf64>
  gpu.wait [%joined]
  return
}

// -----

func.func @no_completion(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %begin = gpu.wait async
  // expected-error @+1 {{no recognized same-block blocking completion; token dominance is not completion}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  %joined = gpu.wait async [%copy]
  return
}

// -----

func.func @unresolved_alias(%src: memref<4xf64>, %dst: memref<4xf64>, %other: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy is still in flight}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{unsupported transfer contract: cannot prove storage roots or slices disjoint}}
  %value = memref.load %other[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func @read_in_loop(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  scf.for %i = %c0 to %c1 step %c1 {
    // expected-error @+1 {{transfer destination read before copy completion}}
    %value = memref.load %dst[%c0] : memref<4xf64>
  }
  gpu.wait [%copy]
  return
}

// -----

func.func private @write_alias(%dst: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{transfer destination written before copy completion}}
  memref.store %zero, %dst[%c0] : memref<?xf64>
  return
}

func.func @write_in_function(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %alias = memref.cast %dst : memref<4xf64> to memref<?xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  func.call @write_alias(%alias) : (memref<?xf64>) -> ()
  gpu.wait [%copy]
  return
}

// -----

func.func private @opaque()
func.func @unknown_call(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %begin = gpu.wait async
  // expected-note @+1 {{copy is still in flight}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  // expected-error @+1 {{unsupported transfer contract: unknown declared function effects}}
  func.call @opaque() : () -> ()
  gpu.wait [%copy]
  return
}

// -----

func.func private @read_carried(%arg: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%carried = %arg) -> memref<4xf64> {
    // expected-error @+1 {{unsupported transfer contract: cannot prove storage roots or slices disjoint}}
    %value = memref.load %carried[%c0] : memref<4xf64>
    scf.yield %carried : memref<4xf64>
  }
  return
}

func.func @call_loop_alias() {
  %src = memref.alloc() : memref<4xf64>
  %dst = memref.alloc() : memref<4xf64>
  %begin = gpu.wait async
  // expected-note @+1 {{copy is still in flight}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  func.call @read_carried(%dst) : (memref<4xf64>) -> ()
  gpu.wait [%copy]
  return
}

// -----

md.particle_set @atoms
func.func @checked_loop_read() {
  %c4 = arith.constant 4 : index
  %src = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %dst = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %other = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<?xf64, #md_exec.set_layout<@atoms, 1>>, memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  // expected-error @+1 {{transfer destination read before copy completion}}
  md_exec.particle_for ins(%dst : memref<?xf64, #md_exec.set_layout<@atoms, 1>>)
      outs(%other : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  ^bb0(%value: f64):
    md_exec.yield %value : f64
  }
  gpu.wait [%copied]
  return
}

// -----

md.particle_set @atoms
func.func @incomplete_loop() {
  %c4 = arith.constant 4 : index
  %src = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %dst = memref.alloc(%c4) : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  %begin = gpu.wait async
  // expected-note @+1 {{copy is still in flight}}
  %copied = gpu.memcpy async [%begin] %dst, %src : memref<?xf64, #md_exec.set_layout<@atoms, 1>>, memref<?xf64, #md_exec.set_layout<@atoms, 1>>
  // expected-error @+1 {{unsupported transfer contract: incomplete loop effect summary}}
  md_exec.particle_for outs(%dst : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
    %c0 = arith.constant 0 : index
    %value = memref.load %dst[%c0] : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    md_exec.yield %value : f64
  }
  gpu.wait [%copied]
  return
}

// -----

func.func @free_disjoint_alias() {
  %src = memref.alloc() : memref<4xf64>
  %base = memref.alloc() : memref<8xf64>
  %head = memref.subview %base[0] [4] [1] : memref<8xf64> to memref<4xf64, strided<[1]>>
  %tail = memref.subview %base[4] [4] [1] : memref<8xf64> to memref<4xf64, strided<[1], offset: 4>>
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %head, %src : memref<4xf64, strided<[1]>>, memref<4xf64>
  // Free concerns the whole generation, even through a disjoint slice.
  // expected-error @+1 {{transfer destination freed or reused before copy completion}}
  memref.dealloc %tail : memref<4xf64, strided<[1], offset: 4>>
  gpu.wait [%copy]
  return
}

// -----

func.func @wrong_token(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %c0 = arith.constant 0 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  gpu.wait [%begin]
  // expected-error @+1 {{transfer destination read before copy completion}}
  %value = memref.load %dst[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}

// -----

func.func private @false_pure(%dst: memref<4xf64>) attributes {dyn.effects = {}} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f64
  // A declaration does not override the checked body of a definition.
  // expected-error @+1 {{transfer destination written before copy completion}}
  memref.store %zero, %dst[%c0] : memref<4xf64>
  return
}

func.func @check_definition(%src: memref<4xf64>, %dst: memref<4xf64>) {
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  func.call @false_pure(%dst) : (memref<4xf64>) -> ()
  gpu.wait [%copy]
  return
}

// -----

func.func @conditional_completion(%src: memref<4xf64>, %dst: memref<4xf64>, %take: i1) {
  %c0 = arith.constant 0 : index
  %begin = gpu.wait async
  // expected-note @+1 {{copy starts here}}
  %copy = gpu.memcpy async [%begin] %dst, %src : memref<4xf64>, memref<4xf64>
  scf.if %take {
    gpu.wait [%copy]
  }
  // expected-error @+1 {{transfer destination read before copy completion}}
  %value = memref.load %dst[%c0] : memref<4xf64>
  gpu.wait [%copy]
  return
}
