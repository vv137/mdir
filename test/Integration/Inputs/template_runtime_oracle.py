"""Seed one exact pair to exercise production arena dispatch and reduction.

The source fixture contributes |O0-O1|^2, with O0=0 and O1=(1/2,0,0).
The independent expected energy is exactly 1/4. This tests runtime views
and the production loop, not search, validity, or a physical water model.
"""
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
anchor = '  %zero = arith.constant 0.0 : f64'
assert source.count(anchor) == 1
source = source.replace(anchor, '  func.call @seed_template(%list) : (!list) -> ()\n' + anchor)
runtime = r'''
memref.global "private" constant @runtime_water : memref<2x4xi32> = dense<[[0, 1, 2, 6], [3, 4, 5, 7]]>
memref.global "private" constant @runtime_order : memref<2xi32> = dense<[0, 1]>
memref.global "private" constant @runtime_frames : memref<6x3xf32> = dense<[[0.0, 0.0, 0.0], [0.125, 0.0, 0.0], [0.0, 0.125, 0.0], [0.5, 0.0, 0.0], [0.625, 0.0, 0.0], [0.5, 0.125, 0.0]]>
func.func private @mdrtTemplatePairsBuffer(!list) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsOrder(!list) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsGroupView(!list, i64) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsFrames(!list) -> memref<?x3xf32, 1> attributes {llvm.emit_c_interface}
func.func private @mdrtTemplatePairsLayoutValue(!list, i64) -> i64
func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr
func.func private @mdrtDeviceActivationEnter(!llvm.ptr)
func.func private @mdrtDeviceActivationClose(!llvm.ptr)
func.func private @printI64(i64)
func.func private @printNewline()
func.func @seed_template(%list: !list) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c39 = arith.constant 39 : index
  %s0 = arith.constant 0 : i64
  %s3 = arith.constant 3 : i64
  %s8 = arith.constant 8 : i64
  %s10 = arith.constant 10 : i64
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %image = arith.constant 537395712 : i32
  %pairs = func.call @mdrtTemplatePairsBuffer(%list) : (!list) -> memref<?xi32, 1>
  %order = func.call @mdrtTemplatePairsOrder(%list) : (!list) -> memref<?xi32, 1>
  %frames = func.call @mdrtTemplatePairsFrames(%list) : (!list) -> memref<?x3xf32, 1>
  %wo = func.call @mdrtTemplatePairsLayoutValue(%list, %s8) : (!list, i64) -> i64
  %eo = func.call @mdrtTemplatePairsLayoutValue(%list, %s10) : (!list, i64) -> i64
  %work = arith.index_cast %wo : i64 to index
  %entry = arith.index_cast %eo : i64 to index
  %n = memref.dim %pairs, %c0 : memref<?xi32, 1>
  %host = memref.alloc(%n) : memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %zero, %host[%i] : memref<?xi32>
  }
  %w1 = arith.addi %work, %c1 : index
  %w2 = arith.addi %work, %c2 : index
  %w3 = arith.addi %work, %c3 : index
  memref.store %zero, %host[%work] : memref<?xi32>
  memref.store %zero, %host[%w1] : memref<?xi32>
  memref.store %zero, %host[%w2] : memref<?xi32>
  memref.store %one, %host[%w3] : memref<?xi32>
  %e1 = arith.addi %entry, %c1 : index
  %e2 = arith.addi %entry, %c2 : index
  memref.store %one, %host[%entry] : memref<?xi32>
  memref.store %image, %host[%e1] : memref<?xi32>
  memref.store %one, %host[%e2] : memref<?xi32>
  memref.store %one, %host[%c39] : memref<?xi32>
  %group_starts = func.call @mdrtTemplatePairsGroupView(%list, %s0) : (!list, i64) -> memref<?xi32, 1>
  %group_lengths = func.call @mdrtTemplatePairsGroupView(%list, %s3) : (!list, i64) -> memref<?xi32, 1>
  %group_host = memref.alloc(%c2) : memref<?xi32>
  memref.store %zero, %group_host[%c0] : memref<?xi32>
  memref.store %zero, %group_host[%c1] : memref<?xi32>
  %gr = gpu.wait async
  %gs = gpu.memcpy async [%gr] %group_starts, %group_host : memref<?xi32, 1>, memref<?xi32>
  gpu.wait [%gs]
  %two = arith.constant 2 : i32
  memref.store %two, %group_host[%c0] : memref<?xi32>
  %gl = gpu.memcpy async [%gs] %group_lengths, %group_host : memref<?xi32, 1>, memref<?xi32>
  gpu.wait [%gl]
  memref.dealloc %group_host : memref<?xi32>
  %oh = memref.get_global @runtime_order : memref<2xi32>
  %fh = memref.get_global @runtime_frames : memref<6x3xf32>
  %ov = memref.cast %oh : memref<2xi32> to memref<?xi32>
  %fv = memref.cast %fh : memref<6x3xf32> to memref<?x3xf32>
  %ready = gpu.wait async
  %p = gpu.memcpy async [%ready] %pairs, %host : memref<?xi32, 1>, memref<?xi32>
  %o = gpu.memcpy async [%p] %order, %ov : memref<?xi32, 1>, memref<?xi32>
  %f = gpu.memcpy async [%o] %frames, %fv : memref<?x3xf32, 1>, memref<?x3xf32>
  gpu.wait [%f]
  memref.dealloc %host : memref<?xi32>
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %types = gpu.alloc(%c8) : memref<?xi32, 1>
  %positions = gpu.alloc(%c8) : memref<?x3xf64, 1>
  %sums = gpu.alloc(%c8) : memref<?xf64, 1>
  %partials = gpu.alloc(%c1024) : memref<?xf64, 1>
  %water0 = memref.get_global @runtime_water : memref<2x4xi32>
  %water = memref.cast %water0 : memref<2x4xi32> to memref<?x4xi32>
  %edge = arith.constant 2.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge
  %energy = func.call @execute(%types, %water, %positions, %cell, %sums, %partials)
      : (memref<?xi32, 1>, memref<?x4xi32>, memref<?x3xf64, 1>, !md.cell, memref<?xf64, 1>, memref<?xf64, 1>) -> f64
  %ref = arith.constant 0.25 : f64
  %same = arith.cmpf oeq, %energy, %ref : f64
  %bits = arith.bitcast %energy : f64 to i64
  func.call @printI64(%bits) : (i64) -> ()
  func.call @printNewline() : () -> ()
  cf.assert %same, "production template energy differs from exact 1/4 oracle"
  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()
  return
}
'''
if len(sys.argv) == 4:
    assert sys.argv[3] == '--staged'
    runtime = runtime.replace('arith.constant 8 : i64', 'arith.constant 11 : i64')
    runtime = runtime.replace('arith.constant 10 : i64', 'arith.constant 12 : i64')
    runtime = runtime.replace('%c39 = arith.constant 39', '%c39 = arith.constant 40')
    runtime = runtime.replace('memref.store %zero, %host[%w1]', 'memref.store %one, %host[%w1]')
    runtime = runtime.replace('memref.store %one, %host[%w3]', 'memref.store %zero, %host[%w3]')
    runtime = runtime.replace('  %n = memref.dim %pairs', '''  %s13 = arith.constant 13 : i64
  %co = func.call @mdrtTemplatePairsLayoutValue(%list, %s13) : (!list, i64) -> i64
  %count_offset = arith.index_cast %co : i64 to index
  %n = memref.dim %pairs''')
    runtime = runtime.replace('  %oh = memref.get_global', '''  memref.store %one, %host[%count_offset] : memref<?xi32>
  %oh = memref.get_global''')
    runtime = runtime.replace('  memref.dealloc %host : memref<?xi32>', '''  memref.dealloc %host : memref<?xi32>
  func.call @compact_seed(%list) : (!list) -> ()''')
    runtime += '\nfunc.func private @compact_seed(!list)\n'
pathlib.Path(sys.argv[2]).write_text(source + runtime)
