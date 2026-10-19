// md_exec.scatter_add compiled and run (D291): 1000 values of 3
// components added into 37 rows at the indices 7 k mod 37, once into a
// field that holds 100 in every component and once into zeros, against
// the same sum written out as a loop. The values are whole numbers, so
// every order of the additions gives the same bits, and each check
// compares exactly: it prints 1 where every component agrees.
//
// On one thread, with plain additions:
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-assign-storage \
// RUN:     --convert-md-exec-to-loops=atomic-grid=false \
// RUN: | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s
//
// Under OpenMP, with atomic additions:
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-assign-storage --convert-md-exec-to-loops \
// RUN: | mlir-opt %lower_loops_to_openmp \
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \
// RUN: | FileCheck %s
//
// scatter-add-gpu.test runs it on a device with each strategy.
//
// The storage form: the first continues in the buffer of the field it adds
// to, the second starts from zeros in a buffer of its own.
// RUN: mdir-opt %s --md-exec-assign-storage | FileCheck %s --check-prefix=STORE
// STORE: md_exec.scatter_add %{{.*}} at %{{.*}} into %[[A:[a-z0-9_]+]] overwrite(false) : memref<?x3xf64>, memref<?xi32>, memref<?x3xf64>
// STORE: md_exec.scatter_add %{{.*}} at %{{.*}} into %{{.*}} overwrite(true) : memref<?x3xf64>, memref<?xi32>, memref<?x3xf64>
//
// CHECK: 1
// CHECK-NEXT: 1
// CHECK-NOT: {{.}}

!val = !md.field<@items, 3 x f64>
!idx = !md.field<@items, i32>
!dst = !md.field<@atoms, 3 x f64>
!buf = memref<?x3xf64>

md.particle_set @atoms
md.particle_set @items

func.func private @printF64(f64)
func.func private @printNewline()

// Prints 1 if `a` and `b` agree in every component.
func.func @same(%a: !buf, %b: !buf) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %a, %c0 : !buf
  %true = arith.constant true
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%s = %true) -> (i1) {
    %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%t = %s) -> (i1) {
      %x = memref.load %a[%i, %k] : !buf
      %y = memref.load %b[%i, %k] : !buf
      %e = arith.cmpf oeq, %x, %y : f64
      %u = arith.andi %t, %e : i1
      scf.yield %u : i1
    }
    scf.yield %r : i1
  }
  %flag = arith.uitofp %all : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %count = arith.constant 1000 : index
  %rows = arith.constant 37 : index
  %one = arith.constant 1.0 : f64
  %hundred = arith.constant 100.0 : f64
  %zero = arith.constant 0.0 : f64
  %vb = memref.alloc(%count) : !buf
  %ib = memref.alloc(%count) : memref<?xi32>
  %ab = memref.alloc(%rows) : !buf
  %expect = memref.alloc(%rows) : !buf
  %expect0 = memref.alloc(%rows) : !buf
  scf.for %r = %c0 to %rows step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      memref.store %hundred, %ab[%r, %k] : !buf
      memref.store %hundred, %expect[%r, %k] : !buf
      memref.store %zero, %expect0[%r, %k] : !buf
    }
  }
  // v_k = (k, 1, k mod 3) at the row 7 k mod 37, and the sum written out.
  scf.for %k = %c0 to %count step %c1 {
    %w = arith.index_cast %k : index to i64
    %x = arith.sitofp %w : i64 to f64
    %m = arith.remui %k, %c3 : index
    %mw = arith.index_cast %m : index to i64
    %z = arith.sitofp %mw : i64 to f64
    memref.store %x, %vb[%k, %c0] : !buf
    memref.store %one, %vb[%k, %c1] : !buf
    memref.store %z, %vb[%k, %c2] : !buf
    %p = arith.muli %k, %c7 : index
    %row = arith.remui %p, %rows : index
    %row32 = arith.index_cast %row : index to i32
    memref.store %row32, %ib[%k] : memref<?xi32>
    scf.for %c = %c0 to %c3 step %c1 {
      %v = memref.load %vb[%k, %c] : !buf
      %e = memref.load %expect[%row, %c] : !buf
      %e1 = arith.addf %e, %v : f64
      memref.store %e1, %expect[%row, %c] : !buf
      %f = memref.load %expect0[%row, %c] : !buf
      %f1 = arith.addf %f, %v : f64
      memref.store %f1, %expect0[%row, %c] : !buf
    }
  }
  %v = mdrt.from_buffer %vb : !buf to !val
  %i = mdrt.from_buffer %ib : memref<?xi32> to !idx
  %a = mdrt.from_buffer %ab : !buf to !dst
  %a1 = md_exec.scatter_add %v at %i into %a : !val, !idx -> !dst
  %z0 = md_exec.zeros : !dst
  %a2 = md_exec.scatter_add %v at %i into %z0 : !val, !idx -> !dst
  %b1 = mdrt.to_buffer %a1 : !dst to !buf
  %b2 = mdrt.to_buffer %a2 : !dst to !buf
  call @same(%b1, %expect) : (!buf, !buf) -> ()
  call @same(%b2, %expect0) : (!buf, !buf) -> ()
  return
}
