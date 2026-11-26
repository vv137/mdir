// The procedure of a domain, called at every step, frees the buffers that it
// allocates (D[distribute-pair-lowering]): each allocation not already
// released gets a release at the end of its block. A buffer that may
// outlive the call is refused, naming the op.
//
// RUN: mdir-opt --md-dist-release-domain-buffers %s --split-input-file \
// RUN:   -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func private @step(
// CHECK-NOT: md_dist.domain_procedure
// CHECK: %[[X:.*]] = memref.alloc(%{{.*}}) : memref<?x3xf64>
// CHECK: %[[Q:.*]] = memref.alloc(%{{.*}}) : memref<?xf64>
// CHECK: %[[F:.*]] = memref.alloc(%{{.*}}) : memref<?x3xf64>
// CHECK: memref.dealloc %[[F]]
// CHECK: call @mdrtDistMatrixRebuild(%{{.*}}, %[[X]])
// CHECK: call @read(%[[CAST:.*]])
// CHECK: scf.if
// CHECK:   %[[T:.*]] = memref.alloc
// CHECK:   memref.dealloc %[[T]]
// CHECK-NEXT: }
// CHECK-DAG: memref.dealloc %[[X]]
// CHECK-DAG: memref.dealloc %[[Q]]
// CHECK-NOT: memref.dealloc
// CHECK: return
func.func private @mdrtDistMatrixRebuild(i64, memref<?x3xf64>) -> i64
func.func private @read(%a: memref<?xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0] : memref<?xf64>
  return %v : f64
}
func.func private @step(%n: index, %h: i64, %b: i1) attributes {md_dist.domain_procedure} {
  %c0 = arith.constant 0 : index
  %x = memref.alloc(%n) : memref<?x3xf64>
  %q = memref.alloc(%n) : memref<?xf64>
  %f = memref.alloc(%n) : memref<?x3xf64>
  // Released already: left as it is.
  memref.dealloc %f : memref<?x3xf64>
  %r = call @mdrtDistMatrixRebuild(%h, %x) : (i64, memref<?x3xf64>) -> i64
  %v = memref.cast %q : memref<?xf64> to memref<?xf64>
  %s = call @read(%v) : (memref<?xf64>) -> f64
  scf.if %b {
    %t = memref.alloc(%n) : memref<?xf64>
    memref.store %s, %t[%c0] : memref<?xf64>
  }
  return
}

// -----

// A function without the mark keeps its buffers.
// CHECK-LABEL: func.func @run(
// CHECK-NOT: memref.dealloc
func.func @run(%n: index) {
  %x = memref.alloc(%n) : memref<?xf64>
  return
}

// -----

func.func private @keeps(memref<?xf64>)
func.func private @step(%n: index) attributes {md_dist.domain_procedure} {
  %x = memref.alloc(%n) : memref<?xf64>
  // expected-error @+1 {{is given a buffer that a domain procedure allocates, and may keep it}}
  call @keeps(%x) : (memref<?xf64>) -> ()
  return
}

// -----

func.func private @step(%n: index) -> memref<?xf64> attributes {md_dist.domain_procedure} {
  %x = memref.alloc(%n) : memref<?xf64>
  // expected-error @+1 {{carries a buffer that a domain procedure allocates out of the block}}
  return %x : memref<?xf64>
}

// -----

// Through a function of the module that returns its argument.
func.func private @pass(%a: memref<?xf64>) -> memref<?xf64> {
  // expected-error @+1 {{carries a buffer that a domain procedure allocates out of the block}}
  return %a : memref<?xf64>
}
func.func private @step(%n: index) attributes {md_dist.domain_procedure} {
  %x = memref.alloc(%n) : memref<?xf64>
  %y = call @pass(%x) : (memref<?xf64>) -> memref<?xf64>
  return
}

// -----

func.func private @step(%n: index, %b: i1, %y: memref<?xf64>) attributes {md_dist.domain_procedure} {
  %r = scf.if %b -> memref<?xf64> {
    %x = memref.alloc(%n) : memref<?xf64>
    // expected-error @+1 {{carries a buffer that a domain procedure allocates out of the block}}
    scf.yield %x : memref<?xf64>
  } else {
    scf.yield %y : memref<?xf64>
  }
  return
}

// -----

func.func private @step(%n: index) -> index attributes {md_dist.domain_procedure} {
  %x = memref.alloc(%n) : memref<?xf64>
  // expected-error @+1 {{takes an alias of a buffer that a domain procedure allocates}}
  %p = memref.extract_aligned_pointer_as_index %x : memref<?xf64> -> index
  return %p : index
}
