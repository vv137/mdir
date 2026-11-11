// Exercise the actual device-renumber registry after a nonidentity spatial sort.
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
!device = memref<?xi32, 1>
!host = memref<?xi32>
memref.global "private" constant @water : memref<2x4xi32> = dense<[[0,1,2,3],[4,5,6,7]]>
memref.global "private" constant @excluded : memref<1x2xi32> = dense<[[8,9]]>
memref.global "private" constant @ids : memref<10xi32> = dense<[0,1,2,3,4,5,6,7,8,9]>
memref.global "private" constant @points : memref<10x3xf64> = dense<[[5.0,0.0,0.0],[5.0,0.0,0.0],[5.0,0.0,0.0],[5.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[3.0,0.0,0.0],[4.0,0.0,0.0]]>
memref.global "private" constant @identity_members : memref<8xi32> = dense<[0,1,2,4,5,6,8,9]>
memref.global "private" constant @sorted_members : memref<8xi32> = dense<[6,7,8,0,1,2,4,5]>
memref.global "private" constant @identity_slots : memref<10xi32> = dense<[0,1,2,-1,3,4,5,-1,6,7]>
memref.global "private" constant @sorted_slots : memref<10xi32> = dense<[3,4,5,-1,6,7,0,1,2,-1]>
memref.global "private" constant @rows : memref<3xi32> = dense<[0,1,2]>
memref.global "private" constant @neighbors : memref<2xi32> = dense<[1,0]>
func.func private @mdrtTemplatePairsMetadata(!list, i64) -> !device attributes {llvm.emit_c_interface}
func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr
func.func private @mdrtDeviceActivationEnter(!llvm.ptr)
func.func private @mdrtDeviceActivationClose(!llvm.ptr)
func.func private @mdrtDeviceTuplesOpen() -> !llvm.ptr
func.func private @mdrtDeviceTuplesBegin(!llvm.ptr)
func.func private @mdrtDeviceTuplesEnter(!llvm.ptr)
func.func private @mdrtDeviceTuplesClose(!llvm.ptr)
func.func private @printI64(i64)
func.func private @printNewline()
func.func @expect(%device: !device, %expected: !host) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %expected, %c0 : !host
  %size = memref.dim %device, %c0 : !device
  %same = arith.cmpi eq, %n, %size : index
  cf.assert %same, "template metadata length differs"
  %host = memref.alloc(%n) : !host
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : !host, !device
  gpu.wait [%done]
  %true = arith.constant true
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%good = %true) -> (i1) {
    %got = memref.load %host[%i] : !host
    %ref = memref.load %expected[%i] : !host
    %okay = arith.cmpi eq, %got, %ref : i32
    %bad = arith.cmpi ne, %got, %ref : i32
    scf.if %bad {
      %index = arith.index_cast %i : index to i64
      %actual = arith.extsi %got : i32 to i64
      %reference = arith.extsi %ref : i32 to i64
      func.call @printI64(%index) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%actual) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%reference) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }
    %next = arith.andi %good, %okay : i1
    scf.yield %next : i1
  }
  memref.dealloc %host : !host
  return %all : i1
}


func.func @exercise(%sort: i1) {
  %n = arith.constant 10 : index
  %ids0 = memref.get_global @ids : memref<10xi32>
  %points0 = memref.get_global @points : memref<10x3xf64>
  %ids = gpu.alloc(%n) : memref<?xi32, 1>
  %points = gpu.alloc(%n) : memref<?x3xf64, 1>
  %ih = memref.cast %ids0 : memref<10xi32> to memref<?xi32>
  %ph = memref.cast %points0 : memref<10x3xf64> to memref<?x3xf64>
  %ready = gpu.wait async
  %copy1 = gpu.memcpy async [%ready] %ids, %ih : memref<?xi32, 1>, memref<?xi32>
  %copy2 = gpu.memcpy async [%copy1] %points, %ph : memref<?x3xf64, 1>, memref<?x3xf64>
  gpu.wait [%copy2]
  %order = gpu.alloc(%n) : memref<?xi32, 1>
  %sorted = gpu.alloc(%n) : memref<?xi32, 1>
  %six = arith.constant 6.0 : f64
  %cell = md.orthorhombic_cell %six, %six, %six
  %state_ids = scf.if %sort -> (!device) {
    md_exec.spatial_order %points, %cell, %ids outs(%order : !device) width(0.6) : memref<?x3xf64, 1>, !device
    md_exec.permute %ids, %order outs(%sorted : !device) : !device, !device
    scf.yield %sorted : !device
  } else {
    scf.yield %ids : !device
  }
  %w0 = memref.get_global @water : memref<2x4xi32>
  %e0 = memref.get_global @excluded : memref<1x2xi32>
  %w = memref.cast %w0 : memref<2x4xi32> to memref<?x4xi32>
  %e = memref.cast %e0 : memref<1x2xi32> to memref<?x2xi32>
  %rw = md_exec.renumber %w, %state_ids : memref<?x4xi32>, !device
  %re = md_exec.renumber %e, %state_ids : memref<?x2xi32>, !device
  %list = md_exec.empty_segmented_pairs %state_ids, %rw exclude(%re : memref<?x2xi32>) cutoff(1.0) skin(0.2) extent(0.1) alpha(0.2) : !device, memref<?x4xi32> -> !list
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %m = func.call @mdrtTemplatePairsMetadata(%list, %c0) : (!list, i64) -> !device
  %s = func.call @mdrtTemplatePairsMetadata(%list, %c1) : (!list, i64) -> !device
  %r = func.call @mdrtTemplatePairsMetadata(%list, %c2) : (!list, i64) -> !device
  %q = func.call @mdrtTemplatePairsMetadata(%list, %c3) : (!list, i64) -> !device
  %em, %es = scf.if %sort -> (!host, !host) {
    %a = memref.get_global @sorted_members : memref<8xi32>
    %b = memref.get_global @sorted_slots : memref<10xi32>
    %av = memref.cast %a : memref<8xi32> to !host
    %bv = memref.cast %b : memref<10xi32> to !host
    scf.yield %av, %bv : !host, !host
  } else {
    %a = memref.get_global @identity_members : memref<8xi32>
    %b = memref.get_global @identity_slots : memref<10xi32>
    %av = memref.cast %a : memref<8xi32> to !host
    %bv = memref.cast %b : memref<10xi32> to !host
    scf.yield %av, %bv : !host, !host
  }
  %rr = memref.get_global @rows : memref<3xi32>
  %qq = memref.get_global @neighbors : memref<2xi32>
  %rv = memref.cast %rr : memref<3xi32> to !host
  %qv = memref.cast %qq : memref<2xi32> to !host
  %a = func.call @expect(%m, %em) : (!device, !host) -> i1
  %b = func.call @expect(%s, %es) : (!device, !host) -> i1
  %c = func.call @expect(%r, %rv) : (!device, !host) -> i1
  %d = func.call @expect(%q, %qv) : (!device, !host) -> i1
  %ab = arith.andi %a, %b : i1
  %cd = arith.andi %c, %d : i1
  %all = arith.andi %ab, %cd : i1
  cf.assert %all, "template sorted metadata differs"
  %one = arith.extui %all : i1 to i64
  func.call @printI64(%one) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %store = func.call @mdrtDeviceTuplesOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceTuplesBegin(%store) : (!llvm.ptr) -> ()
  func.call @mdrtDeviceTuplesEnter(%store) : (!llvm.ptr) -> ()
  %false = arith.constant false
  %true = arith.constant true
  func.call @exercise(%false) : (i1) -> ()
  func.call @exercise(%true) : (i1) -> ()
  // Each exercise uploads fresh metadata and reads it back at once on the
  // shared stream; repeats expose an upload that has not landed yet.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %repeats = arith.constant 16 : index
  scf.for %i = %c0 to %repeats step %c1 {
    func.call @exercise(%true) : (i1) -> ()
  }
  func.call @mdrtDeviceTuplesClose(%store) : (!llvm.ptr) -> ()
  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()
  return
}
