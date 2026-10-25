// RUN: mdir-opt %s --test-md-exec-loop-effects -o /dev/null | FileCheck %s

!real = !md.field<@atoms, f64>
md.particle_set @atoms
func.func private @opaque(f64) -> f64

func.func @unknown_body(%input: !real) -> !real {
  %empty = md_exec.empty : !real
  %out = md_exec.particle_for ins(%input : !real)
      outs(%empty : !real) {
  ^bb0(%x: f64):
    %value = func.call @opaque(%x) : (f64) -> f64
    md_exec.yield %value : f64
  } -> !real
  return %out : !real
}

// CHECK: md_exec.particle_for complete=0 accesses=2 reductions=0
// CHECK: operand=0 effect=read rows=own set=@atoms
// CHECK: operand=1 effect=write rows=own set=@atoms

func.func @undeclared_output_read(
    %dest: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  md_exec.particle_for outs(%dest : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
    %zero = arith.constant 0 : index
    %value = memref.load %dest[%zero] : memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    md_exec.yield %value : f64
  }
  return
}

// A declared write does not cover a read through a captured output buffer.
// CHECK: md_exec.particle_for complete=0 accesses=1 reductions=0
// CHECK: operand=0 effect=write rows=own set=@atoms

func.func @local_allocation(%input: !real) -> !real {
  %empty = md_exec.empty : !real
  %out = md_exec.particle_for ins(%input : !real)
      outs(%empty : !real) {
  ^bb0(%x: f64):
    %scratch = memref.alloc() : memref<1xf64>
    memref.dealloc %scratch : memref<1xf64>
    md_exec.yield %x : f64
  } -> !real
  return %out : !real
}

// Allocation and release are outside the interface's read/write vocabulary.
// CHECK: md_exec.particle_for complete=0 accesses=2 reductions=0

func.func @recursive_op_own_effect(%input: !real, %buffer: memref<1xf64>) -> !real {
  %empty = md_exec.empty : !real
  %out = md_exec.particle_for ins(%input : !real)
      outs(%empty : !real) {
  ^bb0(%x: f64):
    linalg.fill ins(%x : f64) outs(%buffer : memref<1xf64>)
    md_exec.yield %x : f64
  } -> !real
  return %out : !real
}

// Recursive effects include the enclosing op's own effects as well as its body.
// CHECK: md_exec.particle_for complete=0 accesses=2 reductions=0
