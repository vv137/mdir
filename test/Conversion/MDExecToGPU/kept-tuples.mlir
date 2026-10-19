// With `kept-tuples`, the renumbered members of a tuple set, its incidence
// structure, and the copy of the members on the device for the loops that
// evaluate each tuple once are taken from the runtime, which may keep them
// from one activation of an embedding program to the next
// (D253), and given back to it where they
// would be freed. Without the option they are built by loops in the module
// and uploaded, as before.
//
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu="kept-tuples=true" | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --convert-md-exec-to-gpu="kept-tuples=true deterministic=true" | FileCheck %s --check-prefix=ROWS
// The algebra, not the mode, decides (D[reduction-algebra]): ordered sums
// take the rows in a lowering that is not in the deterministic mode.
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true --convert-md-exec-to-gpu="kept-tuples=true" | FileCheck %s --check-prefix=ROWS
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu | FileCheck %s --check-prefix=LOOPS
//
// With `device-tuples` as well (D257), the identities are
// compared with the store's order by a kernel, the entry is taken from the
// store, and where it is to be built the inverse of the order, the
// renumbered members, and the rows are kernels of the module; the copy of
// the members for the loops over each tuple once is the renumbered members
// on the device. Nothing is copied between the host and the device.
// RUN: mdir-opt %s --md-exec-assign-algebra --convert-md-exec-to-gpu="kept-tuples=true device-tuples=true" | FileCheck %s --check-prefix=DEVICE
// DEVICE-LABEL: func.func @renumbered(
// DEVICE:         call @mdrtDeviceTuplesOrderKept(
// DEVICE:         call @mdrtDeviceTuplesOrderFlag()
// DEVICE:         gpu.launch
// DEVICE:         call @mdrtDeviceTuplesOrderEnd(
// DEVICE:         %[[H:[a-z0-9_]+]] = call @mdrtDeviceTuplesTake(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %c1_i32)
// DEVICE:         call @mdrtDeviceTuplesBuild(%[[H]])
// DEVICE:         %[[REN:[a-z0-9_]+]] = call @mdrtDeviceTuplesRenumbered(%[[H]])
// DEVICE:         call @mdrtDeviceTuplesPlace(%[[H]])
// DEVICE:         scf.if
// DEVICE:         gpu.launch
// DEVICE:         gpu.launch
// DEVICE:         call @mdrtDeviceTuplesStructure(%[[H]])
// DEVICE:         scf.if
// DEVICE:         gpu.launch
// DEVICE:         call @mdrtDeviceTuplesBuilt(%[[H]])
// DEVICE-NOT:     gpu.memcpy
// DEVICE-NOT:     mdrtDeviceTuplesMembers
// DEVICE:         gpu.launch
// DEVICE:         call @mdrtDeviceTuplesDone(%[[H]])
// DEVICE-NEXT:    return
// DEVICE-LABEL: func.func @given(
// DEVICE-NOT:     mdrtDeviceTuplesOrder
// DEVICE:         %[[G:[a-z0-9_]+]] = call @mdrtDeviceTuplesTake(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %c0_i32)
// DEVICE:         call @mdrtDeviceTuplesBuilt(%[[G]])
// DEVICE:         call @mdrtDeviceTuplesDone(%[[G]])
// DEVICE-LABEL: func.func @found(
// DEVICE-NOT:     call @mdrtDeviceTuples
// DEVICE:         return

md.particle_set @atoms

// The members renumbered by the identities, which are on the device, and
// the structure of the renumbered members.
//
// CHECK-LABEL: func.func @renumbered(
// CHECK-SAME:      %[[FILE:[a-z0-9_]+]]: memref<?x2xi32>, %[[IDS:[a-z0-9_]+]]: memref<?xi32, 1>, %[[N:[a-z0-9_]+]]: index
// CHECK:         %[[WIDE:[a-z0-9_]+]] = memref.cast %[[FILE]] : memref<?x2xi32> to memref<?x?xi32>
// CHECK:         %[[TAKEN:[a-z0-9_]+]] = call @mdrtDeviceTuplesRenumber(%[[WIDE]], %[[IDS]]) : (memref<?x?xi32>, memref<?xi32, 1>) -> memref<?x?xi32>
// CHECK:         %[[MEMBERS:[a-z0-9_]+]] = memref.cast %[[TAKEN]] : memref<?x?xi32> to memref<?x2xi32>
// CHECK:         %[[AGAIN:[a-z0-9_]+]] = memref.cast %[[MEMBERS]] : memref<?x2xi32> to memref<?x?xi32>
// CHECK:         %[[INC:[a-z0-9_]+]] = call @mdrtDeviceTuplesIncidence(%[[AGAIN]], %[[N]]) : (memref<?x?xi32>, index) -> memref<?x?xi32, 1>
// CHECK:         %[[COPY:[a-z0-9_]+]] = call @mdrtDeviceTuplesMembers(%{{[a-z0-9_]+}}) : (memref<?x?xi32>) -> memref<?x?xi32, 1>
// CHECK-NOT:     gpu.memcpy
// CHECK-NOT:     gpu.alloc
// CHECK:         gpu.launch
// CHECK-NOT:     dealloc
// CHECK:         call @mdrtDeviceTuplesRelease(%[[TAKEN]]) : (memref<?x?xi32>) -> ()
// CHECK:         %[[PLAIN:[a-z0-9_]+]] = memref.memory_space_cast %[[INC]] : memref<?x?xi32, 1> to memref<?x?xi32>
// CHECK:         call @mdrtDeviceTuplesRelease(%[[PLAIN]]) : (memref<?x?xi32>) -> ()
// CHECK:         %[[PLAIN2:[a-z0-9_]+]] = memref.memory_space_cast %[[COPY]] : memref<?x?xi32, 1> to memref<?x?xi32>
// CHECK:         call @mdrtDeviceTuplesRelease(%[[PLAIN2]]) : (memref<?x?xi32>) -> ()
// CHECK:         return
// In the deterministic mode the rows of the particles take the tuples: no
// copy of the members.
// ROWS-LABEL:  func.func @renumbered(
// ROWS:          call @mdrtDeviceTuplesRenumber(
// ROWS:          call @mdrtDeviceTuplesIncidence(
// ROWS-NOT:      call @mdrtDeviceTuplesMembers(
// ROWS:          return
// LOOPS-LABEL: func.func @renumbered(
// LOOPS-NOT:     call @mdrtDeviceTuples
// LOOPS:         gpu.memcpy
// LOOPS:         scf.for
// LOOPS:         gpu.memcpy
// LOOPS:         return
func.func @renumbered(%file: memref<?x2xi32>, %ids: memref<?xi32, 1>,
                      %n: index, %x: memref<?x3xf32, 1>, %cell: !md.cell,
                      %f: memref<?x3xf32, 1>) {
  %members = md_exec.renumber %file, %ids : memref<?x2xi32>, memref<?xi32, 1>
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

// The members that the function is given, in the order of the files.
//
// CHECK-LABEL: func.func @given(
// CHECK-NOT:     call @mdrtDeviceTuplesRenumber(
// CHECK:         call @mdrtDeviceTuplesIncidence(
// CHECK:         call @mdrtDeviceTuplesMembers(
// CHECK:         return
func.func @given(%members: memref<?x2xi32>, %n: index,
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

// Tuples that the program makes itself, as the triplets of a three-body
// term are at every build, are not the runtime's to keep: their structure
// is built here and freed here.
//
// CHECK-LABEL: func.func @found(
// CHECK-NOT:     call @mdrtDeviceTuples
// CHECK:         gpu.memcpy
// CHECK:         return
func.func @found(%count: index, %n: index, %x: memref<?x3xf32, 1>,
                 %cell: !md.cell, %f: memref<?x3xf32, 1>) {
  %members = memref.alloc(%count) : memref<?x2xi32>
  %inc = md_exec.build_incidence %members size(%n)
      : memref<?x2xi32> -> memref<?x?xi32, 1>
  md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
      outs(%f : memref<?x3xf32, 1>) arity(2) {
  ^bb0(%d: vector<3xf32>):
    %n0 = arith.negf %d : vector<3xf32>
    md_exec.yield %d, %n0 : vector<3xf32>, vector<3xf32>
  } : memref<?x?xi32, 1>, memref<?x3xf32, 1>
  memref.dealloc %members : memref<?x2xi32>
  return
}
