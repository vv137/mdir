// RUN: mdir-opt %s --md-exec-assign-streams=reciprocal=true -split-input-file | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-streams -split-input-file | FileCheck %s --check-prefix=OFF

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

// The sum moves up past a permutation into another buffer, runs beside
// one, and the first stream waits for it before the permutation that reads
// its forces.
//
// CHECK-LABEL: func.func @window(
// CHECK:         md_exec.reciprocal {{.*}} {md_exec.side}
// CHECK-NEXT:    md_exec.permute %[[X:[a-z0-9_]+]], %{{[a-z0-9_]+}} outs(%[[Y1:[a-z0-9_]+]] :
// CHECK-NEXT:    md_exec.permute
// CHECK-NEXT:    md_exec.join
// CHECK-NEXT:    md_exec.permute %[[F:[a-z0-9_]+]],
// OFF-LABEL: func.func @window(
// OFF-NOT:     md_exec.side
// OFF-NOT:     md_exec.join
func.func @window(%n: index, %k: index, %cell: !md.cell) {
  %x = gpu.alloc (%n) : !pos
  %y1 = gpu.alloc (%n) : !pos
  %y2 = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %g = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  md_exec.permute %x, %o outs(%y1 : !pos) : !pos, !ord
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  md_exec.permute %x, %o outs(%y2 : !pos) : !pos, !ord
  md_exec.permute %f, %o outs(%g : !frc) : !frc, !ord
  return
}

// -----

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

func.func private @unknown()

// The sum stays below the op that writes the positions it reads. A call,
// whose effects are not declared, and a use of its energy end the window.
//
// CHECK-LABEL: func.func @ends(
// CHECK:         md_exec.permute %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} outs(%[[X:[a-z0-9_]+]] :
// CHECK-NEXT:    md_exec.reciprocal %[[X]], {{.*}} {md_exec.side}
// CHECK-NEXT:    md_exec.join
// CHECK-NEXT:    call @unknown()
// CHECK:         md_exec.reciprocal {{.*}} {md_exec.side}
// CHECK-NEXT:    md_exec.join
// CHECK-NEXT:    arith.addf
func.func @ends(%n: index, %k: index, %cell: !md.cell) -> f64 {
  %x = gpu.alloc (%n) : !pos
  %x0 = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  md_exec.permute %x0, %o outs(%x : !pos) : !pos, !ord
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  call @unknown() : () -> ()
  %e2, %w2 = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  %t = arith.addf %e, %e2 : f64
  return %t : f64
}

// -----

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

// Buffers that a loop carries and permutes are distinct at every
// iteration: the sum writes one while another is written beside it, and
// the first stream waits before the end of the iteration. Where the loop
// does not permute them, they may be the same, and the window is empty.
//
// CHECK-LABEL: func.func @carried(
// CHECK:         scf.for
// CHECK:           md_exec.reciprocal {{.*}} {md_exec.side}
// CHECK-NEXT:      md_exec.permute
// CHECK-NEXT:      md_exec.join
// CHECK-NEXT:      scf.yield
// CHECK:         scf.for
// CHECK:           md_exec.reciprocal {{.*}} {md_exec.side}
// CHECK-NEXT:      md_exec.join
// CHECK-NEXT:      md_exec.permute
func.func @carried(%n: index, %k: index, %steps: index, %cell: !md.cell) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %g = gpu.alloc (%n) : !frc
  %h = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %r:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%a = %f, %b = %g)
      -> (!frc, !frc) {
    %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%a : !frc)
        scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
                memref<?xf32, 1>, memref<?xf64, 1>)
        grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
        : !pos, !chg, !mod -> f64, vector<9xf64>
    md_exec.permute %h, %o outs(%b : !frc) : !frc, !ord
    scf.yield %b, %a : !frc, !frc
  }
  %t:2 = scf.for %i = %c0 to %steps step %c1 iter_args(%a = %f, %b = %g)
      -> (!frc, !frc) {
    %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%a : !frc)
        scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
                memref<?xf32, 1>, memref<?xf64, 1>)
        grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
        : !pos, !chg, !mod -> f64, vector<9xf64>
    md_exec.permute %h, %o outs(%b : !frc) : !frc, !ord
    scf.yield %a, %a : !frc, !frc
  }
  return
}

// -----

!pos = memref<?x3xf64, 1>
!frc = memref<?x3xf32, 1>
!chg = memref<?xf32, 1>
!ord = memref<?xi32, 1>
!mod = memref<?x?xf64, 1>

// Two arguments of the function may be the same memory.
//
// CHECK-LABEL: func.func @arguments(
// CHECK:         md_exec.reciprocal {{.*}} {md_exec.side}
// CHECK-NEXT:    md_exec.join
// CHECK-NEXT:    md_exec.permute
func.func @arguments(%x: !pos, %y: !pos, %n: index, %k: index,
                     %cell: !md.cell) {
  %z = gpu.alloc (%n) : !pos
  %q = gpu.alloc (%n) : !chg
  %o = gpu.alloc (%n) : !ord
  %m = gpu.alloc (%k, %k) : !mod
  %f = gpu.alloc (%n) : !frc
  %s0 = gpu.alloc (%k) : memref<?xi64, 1>
  %s1 = gpu.alloc (%k) : memref<?xf32, 1>
  %s2 = gpu.alloc (%k) : memref<?xf32, 1>
  %s3 = gpu.alloc (%k) : memref<?xf64, 1>
  %e, %w = md_exec.reciprocal %x, %q, %cell, %m outs(%f : !frc)
      scratch(%s0, %s1, %s2, %s3 : memref<?xi64, 1>, memref<?xf32, 1>,
              memref<?xf32, 1>, memref<?xf64, 1>)
      grid([8, 8, 8]) order(4) beta(3.0) coulomb(138.935457644)
      : !pos, !chg, !mod -> f64, vector<9xf64>
  md_exec.permute %z, %o outs(%y : !pos) : !pos, !ord
  return
}
