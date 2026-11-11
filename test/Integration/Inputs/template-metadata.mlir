// Runtime metadata oracle: renumbered OHHM units, packed slots and the
// exact SS exclusion graph. The two creation paths use host and already
// device-resident members. Growth must preserve every static metadata view.
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
!device = memref<?xi32, 1>
!host = memref<?xi32>
memref.global "private" constant @water : memref<2x4xi32> = dense<[[4, 2, 7, 0], [6, 1, 8, 3]]>
memref.global "private" constant @excluded : memref<5x2xi32> = dense<[[4, 2], [5, 9], [9, 5], [6, 8], [5, 5]]>
memref.global "private" constant @members : memref<8xi32> = dense<[4, 2, 7, 6, 1, 8, 5, 9]>
memref.global "private" constant @slots : memref<10xi32> = dense<[-1, 4, 1, -1, 0, 6, 3, 2, 5, 7]>
memref.global "private" constant @rows : memref<3xi32> = dense<[0, 1, 2]>
memref.global "private" constant @neighbors : memref<2xi32> = dense<[1, 0]>
func.func private @mdrtTemplatePairsMetadata(!list, i64) -> !device attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsGrow(!list, i64) -> i64
func.func private @mdrtTemplatePairsCapacity(!list) -> i64
func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr
func.func private @mdrtDeviceActivationEnter(!llvm.ptr)
func.func private @mdrtDeviceActivationClose(!llvm.ptr)
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

func.func @check_views(%list: !list) -> i1 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %m = func.call @mdrtTemplatePairsMetadata(%list, %c0) : (!list, i64) -> !device
  %s = func.call @mdrtTemplatePairsMetadata(%list, %c1) : (!list, i64) -> !device
  %r = func.call @mdrtTemplatePairsMetadata(%list, %c2) : (!list, i64) -> !device
  %n = func.call @mdrtTemplatePairsMetadata(%list, %c3) : (!list, i64) -> !device
  %em0 = memref.get_global @members : memref<8xi32>
  %es0 = memref.get_global @slots : memref<10xi32>
  %er0 = memref.get_global @rows : memref<3xi32>
  %en0 = memref.get_global @neighbors : memref<2xi32>
  %em = memref.cast %em0 : memref<8xi32> to !host
  %es = memref.cast %es0 : memref<10xi32> to !host
  %er = memref.cast %er0 : memref<3xi32> to !host
  %en = memref.cast %en0 : memref<2xi32> to !host
  %a = func.call @expect(%m, %em) : (!device, !host) -> i1
  %b = func.call @expect(%s, %es) : (!device, !host) -> i1
  %c = func.call @expect(%r, %er) : (!device, !host) -> i1
  %d = func.call @expect(%n, %en) : (!device, !host) -> i1
  %ab = arith.andi %a, %b : i1
  %cd = arith.andi %c, %d : i1
  %all = arith.andi %ab, %cd : i1
  return %all : i1
}

func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %c10 = arith.constant 10 : index
  %types = gpu.alloc(%c10) : memref<?xi32, 1>
  %water = memref.get_global @water : memref<2x4xi32>
  %excluded = memref.get_global @excluded : memref<5x2xi32>
  %water_view = memref.cast %water : memref<2x4xi32> to memref<?x4xi32>
  %excluded_view = memref.cast %excluded : memref<5x2xi32> to memref<?x2xi32>
  %host_list = md_exec.empty_segmented_pairs %types, %water_view
      exclude(%excluded_view : memref<?x2xi32>)
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  %host_okay = func.call @check_views(%host_list) : (!list) -> i1
  %required = arith.constant 2000 : i64
  %grew = func.call @mdrtTemplatePairsGrow(%host_list, %required) : (!list, i64) -> i64
  %capacity = func.call @mdrtTemplatePairsCapacity(%host_list) : (!list) -> i64
  %fits = arith.cmpi uge, %capacity, %required : i64
  %grown_okay = func.call @check_views(%host_list) : (!list) -> i1
  %device_water = gpu.alloc() : memref<2x4xi32, 1>
  %device_excluded = gpu.alloc() : memref<5x2xi32, 1>
  %ready = gpu.wait async
  %w = gpu.memcpy async [%ready] %device_water, %water : memref<2x4xi32, 1>, memref<2x4xi32>
  %e = gpu.memcpy async [%w] %device_excluded, %excluded : memref<5x2xi32, 1>, memref<5x2xi32>
  gpu.wait [%e]
  %device_water_view = memref.cast %device_water : memref<2x4xi32, 1> to memref<?x4xi32, 1>
  %device_excluded_view = memref.cast %device_excluded : memref<5x2xi32, 1> to memref<?x2xi32, 1>
  %device_list = md_exec.empty_segmented_pairs %types, %device_water_view
      exclude(%device_excluded_view : memref<?x2xi32, 1>)
      cutoff(1.0) skin(0.2) extent(0.1) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32, 1> -> !list
  %device_okay = func.call @check_views(%device_list) : (!list) -> i1
  %first = arith.andi %host_okay, %grown_okay : i1
  %second = arith.andi %fits, %device_okay : i1
  %all = arith.andi %first, %second : i1
  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()
  %flag = arith.extui %all : i1 to i64
  func.call @printI64(%flag) : (i64) -> ()
  func.call @printNewline() : () -> ()
  cf.assert %all, "template runtime metadata differs"
  return
}
