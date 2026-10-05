// By default (tuples-once), a loop over tuples that may share particles, has no
// global sums, and adds to its destination evaluates each tuple once, a
// thread for each tuple, and adds to the members with atomics: in f32 by
// PTX's own reduction.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu="tuples-once=true" | FileCheck %s
// RUN: mdir-opt %s --convert-md-exec-to-gpu="tuples-once=false" | FileCheck %s --check-prefix=ROWS
// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s --check-prefix=CENTERS
// RUN: mdir-opt %s --convert-md-exec-to-gpu="fuse-centers=false" | FileCheck %s --check-prefix=APART

md.particle_set @atoms

// CHECK-LABEL: func.func @bonds(
// CHECK:         gpu.memcpy
// CHECK:         gpu.launch
// CHECK-COUNT-6:   llvm.inline_asm has_side_effects "red.relaxed.gpu.global.add.f32 [$0], $1;"
// ROWS-LABEL:  func.func @bonds(
// ROWS-NOT:      llvm.inline_asm
func.func @bonds(%members: memref<?x2xi32>, %n: index,
                 %x: memref<?x3xf32, 1>, %cell: !md.cell,
                 %f: memref<?x3xf32, 1>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  return
}

// With global sums, a thread for each of the tuples or of the particles,
// whichever are fewer, takes every so manyth tuple and writes the sums of
// its tuples to its row, and the reduction is over those rows: a set of few
// tuples, as the pairs of the centers of groups, launches and reduces no
// more rows than it has tuples. The rows are reduced in one part when there
// are few; the block of that part then evaluates the tuples of each row
// itself, and the kernel that writes the rows to the buffer of
// contributions is launched only for several parts (#20).
//
// CHECK-LABEL: func.func @energy(
// CHECK:         %[[TUPLES:[a-z0-9_]+]] = memref.dim %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?x2xi32, 1>
// CHECK:         %[[COUNT:[0-9]+]] = arith.minui %{{[a-z0-9_]+}}, %[[TUPLES]] : index
// CHECK:         %[[PARTS:[0-9]+]] = arith.minsi
// CHECK:         %[[ONE:[0-9]+]] = arith.cmpi eq, %[[PARTS]]
// CHECK:         %[[SEVERAL:[0-9]+]] = arith.cmpi ne, %[[PARTS]]
// CHECK:         scf.if %[[SEVERAL]] {
// CHECK:           gpu.launch
// CHECK:             scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[TUPLES]] step %[[COUNT]]
// CHECK:             memref.store %{{[0-9]+}}, %[[A:[a-z0-9]+]][
// CHECK:         gpu.launch
// CHECK:           scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[COUNT]] step
// CHECK:             scf.if %[[ONE]] -> (f64) {
// CHECK:               scf.for %{{[a-z0-9]+}} = %{{[a-z0-9_]+}} to %[[TUPLES]] step %[[COUNT]]
// CHECK:             } else {
// CHECK:               memref.load %[[A]][
func.func @energy(%members: memref<?x2xi32>, %n: index,
                  %x: memref<?x3xf32, 1>, %cell: !md.cell,
                  %f: memref<?x3xf32, 1>, %a: memref<?xf64, 1>,
                  %b: memref<?xf64, 1>) -> f64 {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) reduce(%u0 : f64)
      scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    %s = vector.reduction <add>, %d : vector<3xf32> into f32
    %e = arith.extf %s : f32 to f64
    md_exec.yield %d, %n0, %e : vector<3xf32>, vector<3xf32>, f64
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1> -> f64
  return %u : f64
}

// Adjacent loops that evaluate each tuple once and have no global sums are
// one kernel over the tuples of both: a thread takes the first loop below
// the count of its tuples, and the second after it (#20).
//
// CHECK-LABEL: func.func @two(
// CHECK:         %[[N1:[a-z0-9_]+]] = memref.dim %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?x2xi32, 1>
// CHECK:         %[[N2:[a-z0-9_]+]] = memref.dim %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<?x2xi32, 1>
// CHECK:         %[[END:[0-9]+]] = arith.addi %[[N1]], %[[N2]] : index
// CHECK:         gpu.launch
// CHECK:           arith.cmpi ult, %{{[0-9]+}}, %[[END]]
// CHECK:           arith.cmpi ult, %{{[0-9]+}}, %[[N1]]
// CHECK:           arith.subi %{{[0-9]+}}, %[[N1]]
// CHECK-NOT:     gpu.launch
// CHECK:         return
// ROWS-LABEL:  func.func @two(
// ROWS-NOT:      llvm.inline_asm
func.func @two(%m1: memref<?x2xi32>, %m2: memref<?x2xi32>, %n: index,
               %x: memref<?x3xf32, 1>, %cell: !md.cell,
               %f: memref<?x3xf32, 1>) {
  %i1 = md_exec.build_incidence %m1 size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  %i2 = md_exec.build_incidence %m2 size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  md_exec.tuple_for %i1, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  md_exec.tuple_for %i2, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %n0, %d : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  return
}

// Sums over tuples evaluated once that reach only a later loop over tuples
// evaluated once, through arithmetic: the centers of a term over the centers
// of groups and its forces (D139). With one part, a block of the kernel of
// the forces computes the sums, as the kernel of the reduction would, then
// the arithmetic and the forces; the kernels of several parts are launched
// only for several, and leave the totals in a buffer of their own. With one part, one kernel runs instead of the
// reduction and the forces, and the host reads no sum (#66); without the
// fusion the results also take a buffer that a kernel clears.
//
// CENTERS-LABEL: func.func @centers(
// CENTERS:         scf.if %{{[0-9]+}} {
// CENTERS-COUNT-3:   gpu.launch
// CENTERS:         gpu.launch {{.*}}workgroup(
// CENTERS:           scf.if %{{[0-9]+}} -> (f64) {
// CENTERS:             gpu.barrier
// CENTERS:           } else {
// CENTERS:             memref.load %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : memref<1xf64, 1>
// CENTERS:           arith.mulf %{{[0-9]+}}, %{{[a-z0-9_]+}} {{.*}}: f64
// CENTERS-NOT:     gpu.launch
// CENTERS:         return
// APART-LABEL:  func.func @centers(
// APART-COUNT-5:  gpu.launch
// APART:          return
func.func @centers(%members: memref<?x2xi32>, %n: index,
                   %x: memref<?x3xf32, 1>, %cell: !md.cell,
                   %f: memref<?x3xf32, 1>, %w: memref<?xf64, 1>,
                   %a: memref<?xf64, 1>, %b: memref<?xf64, 1>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  %zero = arith.constant 0.0 : f64
  %s = md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      tuple(%w : memref<?xf64, 1>) reduce(%zero : f64)
      scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>, %wt: f64):
    %dx = vector.extract %d[0] : f32 from vector<3xf32>
    %e = arith.extf %dx : f32 to f64
    %c = arith.mulf %wt, %e : f64
    md_exec.yield %c : f64
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1> -> f64
  %k = arith.constant 2.0 : f64
  %g = arith.mulf %s, %k : f64
  %g32 = arith.truncf %g : f64 to f32
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      tuple(%w : memref<?xf64, 1>) outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>, %wt: f64):
    %v = vector.broadcast %g32 : f32 to vector<3xf32>
    %m = arith.negf %v : vector<3xf32>
    md_exec.yield %v, %m : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  return
}
