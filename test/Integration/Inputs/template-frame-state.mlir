// Reference epochs and retained packing cells have distinct fixed views.
// Fill all views before checking them to expose overlap; growth preserves
// their addresses and all 90 exact f64 values. No decimal serialization.
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
!state = memref<?x3xf64, 1>
memref.global "private" constant @water : memref<2x4xi32> = dense<[[4, 2, 7, 0], [6, 1, 8, 3]]>
func.func private @mdrtTemplatePairsFrameState(!list, i64) -> !state attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsGrow(!list, i64) -> i64
func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr
func.func private @mdrtDeviceActivationEnter(!llvm.ptr)
func.func private @mdrtDeviceActivationClose(!llvm.ptr)
func.func private @printI64(i64)
func.func private @printNewline()

func.func @values(%slot: i64, %row: index, %axis: index) -> f64 {
  %r = arith.index_cast %row : index to i64
  %a = arith.index_cast %axis : index to i64
  %c64 = arith.constant 64 : i64
  %c4 = arith.constant 4 : i64
  %s = arith.muli %slot, %c64 : i64
  %q = arith.muli %r, %c4 : i64
  %u = arith.addi %s, %q : i64
  %v = arith.addi %u, %a : i64
  %f = arith.sitofp %v : i64 to f64
  %quarter = arith.constant 0.25 : f64
  %value = arith.mulf %f, %quarter : f64
  return %value : f64
}
func.func @fill(%device: !state, %slot: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %device, %c0 : !state
  %host = memref.alloc(%n) : memref<?x3xf64>
  scf.for %row = %c0 to %n step %c1 {
    scf.for %axis = %c0 to %c3 step %c1 {
      %value = func.call @values(%slot, %row, %axis) : (i64, index, index) -> f64
      memref.store %value, %host[%row, %axis] : memref<?x3xf64>
    }
  }
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %device, %host : !state, memref<?x3xf64>
  gpu.wait [%done]
  memref.dealloc %host : memref<?x3xf64>
  return
}
func.func @check(%device: !state, %slot: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c2 = arith.constant 2 : index
  %s4 = arith.constant 4 : i64
  %s6 = arith.constant 6 : i64
  %units = arith.cmpi ult, %slot, %s4 : i64
  %reaches = arith.cmpi eq, %slot, %s6 : i64
  %fixed = arith.select %reaches, %c2, %c6 : index
  %expected = arith.select %units, %c4, %fixed : index
  %n = memref.dim %device, %c0 : !state
  %same_size = arith.cmpi eq, %n, %expected : index
  cf.assert %same_size, "template reference-state shape differs"
  %host = memref.alloc(%n) : memref<?x3xf64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?x3xf64>, !state
  gpu.wait [%done]
  scf.for %row = %c0 to %n step %c1 {
    scf.for %axis = %c0 to %c3 step %c1 {
      %reference = func.call @values(%slot, %row, %axis) : (i64, index, index) -> f64
      %actual = memref.load %host[%row, %axis] : memref<?x3xf64>
      %wanted_bits = arith.bitcast %reference : f64 to i64
      %actual_bits = arith.bitcast %actual : f64 to i64
      %same = arith.cmpi eq, %wanted_bits, %actual_bits : i64
      %bad = arith.cmpi ne, %wanted_bits, %actual_bits : i64
      scf.if %bad {
        func.call @printI64(%slot) : (i64) -> ()
        func.call @printNewline() : () -> ()
        func.call @printI64(%wanted_bits) : (i64) -> ()
        func.call @printNewline() : () -> ()
        func.call @printI64(%actual_bits) : (i64) -> ()
        func.call @printNewline() : () -> ()
      }
      cf.assert %same, "template reference-state bits differ"
    }
  }
  memref.dealloc %host : memref<?x3xf64>
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i7 = arith.constant 7 : index
  %c10 = arith.constant 10 : index
  %types = gpu.alloc(%c10) : memref<?xi32, 1>
  %water0 = memref.get_global @water : memref<2x4xi32>
  %water = memref.cast %water0 : memref<2x4xi32> to memref<?x4xi32>
  %list = md_exec.empty_segmented_pairs %types, %water
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  scf.for %i = %i0 to %i7 step %i1 {
    %slot = arith.index_cast %i : index to i64
    %view = func.call @mdrtTemplatePairsFrameState(%list, %slot) : (!list, i64) -> !state
    func.call @fill(%view, %slot) : (!state, i64) -> ()
  }
  %before = func.call @mdrtTemplatePairsFrameState(%list, %c0) : (!list, i64) -> !state
  %pointer = memref.extract_aligned_pointer_as_index %before : !state -> index
  scf.for %i = %i0 to %i7 step %i1 {
    %slot = arith.index_cast %i : index to i64
    %view = func.call @mdrtTemplatePairsFrameState(%list, %slot) : (!list, i64) -> !state
    func.call @check(%view, %slot) : (!state, i64) -> ()
  }
  %required = arith.constant 64 : i64
  %grew = func.call @mdrtTemplatePairsGrow(%list, %required) : (!list, i64) -> i64
  %did_grow = arith.cmpi eq, %grew, %c1 : i64
  cf.assert %did_grow, "template frame-state fixture did not grow"
  %after = func.call @mdrtTemplatePairsFrameState(%list, %c0) : (!list, i64) -> !state
  %new_pointer = memref.extract_aligned_pointer_as_index %after : !state -> index
  %stable = arith.cmpi eq, %pointer, %new_pointer : index
  cf.assert %stable, "template reference-state allocation moved on pair growth"
  scf.for %i = %i0 to %i7 step %i1 {
    %slot = arith.index_cast %i : index to i64
    %view = func.call @mdrtTemplatePairsFrameState(%list, %slot) : (!list, i64) -> !state
    func.call @check(%view, %slot) : (!state, i64) -> ()
  }
  func.call @printI64(%c1) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()
  return
}
