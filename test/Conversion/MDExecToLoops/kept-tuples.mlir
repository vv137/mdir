// With `kept-tuples`, the renumbered members of a tuple set and its
// incidence structure are taken from the runtime, which may keep them from
// one activation of an embedding program to the next
// (D253), and given back to it where they
// would be freed. Without the option they are built by loops in the
// module, as before.
//
// RUN: mdir-opt %s --convert-md-exec-to-loops="kept-tuples=true" | FileCheck %s
// RUN: mdir-opt %s --convert-md-exec-to-loops | FileCheck %s --check-prefix=LOOPS

md.particle_set @atoms

// CHECK-LABEL: func.func @renumbered(
// CHECK-SAME:      %[[FILE:[a-z0-9_]+]]: memref<?x2xi32>, %[[IDS:[a-z0-9_]+]]: memref<?xi32>, %[[N:[a-z0-9_]+]]: index
// CHECK:         %[[WIDE:[a-z0-9_]+]] = memref.cast %[[FILE]] : memref<?x2xi32> to memref<?x?xi32>
// CHECK:         %[[TAKEN:[a-z0-9_]+]] = call @mdrtTuplesRenumber(%[[WIDE]], %[[IDS]]) : (memref<?x?xi32>, memref<?xi32>) -> memref<?x?xi32>
// CHECK:         %[[MEMBERS:[a-z0-9_]+]] = memref.cast %[[TAKEN]] : memref<?x?xi32> to memref<?x2xi32>
// CHECK:         %[[AGAIN:[a-z0-9_]+]] = memref.cast %[[MEMBERS]] : memref<?x2xi32> to memref<?x?xi32>
// CHECK:         %[[INC:[a-z0-9_]+]] = call @mdrtTuplesIncidence(%[[AGAIN]], %[[N]]) : (memref<?x?xi32>, index) -> memref<?x?xi32>
// CHECK-NOT:     memref.alloc
// CHECK:         scf.parallel
// CHECK-NOT:     dealloc
// CHECK:         call @mdrtTuplesRelease(%[[TAKEN]]) : (memref<?x?xi32>) -> ()
// CHECK:         call @mdrtTuplesRelease(%[[INC]]) : (memref<?x?xi32>) -> ()
// CHECK:         return
// LOOPS-LABEL: func.func @renumbered(
// LOOPS-NOT:     call @mdrtTuples
// LOOPS:         memref.alloc
// LOOPS:         scf.for
// LOOPS:         memref.dealloc
// LOOPS:         return
func.func @renumbered(%file: memref<?x2xi32>, %ids: memref<?xi32>, %n: index,
                      %x: memref<?x3xf64>, %cell: !md.cell,
                      %f: memref<?x3xf64>) {
  %members = md_exec.renumber %file, %ids : memref<?x2xi32>, memref<?xi32>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf64>) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n0 = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n0 : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32>, memref<?x3xf64>
  return
}

// The members that the function is given, in the order of the files.
//
// CHECK-LABEL: func.func @given(
// CHECK-NOT:     call @mdrtTuplesRenumber(
// CHECK:         %[[INC:[a-z0-9_]+]] = call @mdrtTuplesIncidence(
// CHECK:         call @mdrtTuplesRelease(%[[INC]])
// CHECK:         return
func.func @given(%members: memref<?x2xi32>, %n: index, %x: memref<?x3xf64>,
                 %cell: !md.cell, %f: memref<?x3xf64>) {
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf64>) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n0 = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n0 : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32>, memref<?x3xf64>
  return
}

// Tuples that the program makes itself, as the triplets of a three-body
// term are at every build, are not the runtime's to keep: their structure
// is built here and freed here.
//
// CHECK-LABEL: func.func @found(
// CHECK-NOT:     call @mdrtTuples
// CHECK:         return
func.func @found(%count: index, %n: index, %x: memref<?x3xf64>,
                 %cell: !md.cell, %f: memref<?x3xf64>) {
  %members = memref.alloc(%count) : memref<?x2xi32>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf64>) arity(2) {
  ^bb0(%d: vector<3xf64>):
    %n0 = arith.negf %d : vector<3xf64>
    md_exec.yield %d, %n0 : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32>, memref<?x3xf64>
  memref.dealloc %members : memref<?x2xi32>
  return
}
