// Runs of loops over disjoint tuples become one kernel (D83), unless a loop
// reads what an earlier loop of the run writes, or writes what one reads.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s

md.particle_set @atoms

// Two sets of groups that read the same positions and write fields of their
// own: one kernel, in which each loop runs for the particles of its own
// field.
//
// CHECK-LABEL: func.func @independent(
// CHECK:         gpu.launch {{.*}} function(@independent_tuple_for_l{{[0-9]+}}_run2_{{[0-9]+}})
// CHECK-NOT:     gpu.launch
// CHECK:         return
func.func @independent(%pairs: memref<?x?xi32, 1>, %trios: memref<?x?xi32, 1>,
                       %x: memref<?x3xf64, 1>, %cell: !md.cell,
                       %a: memref<?x3xf64, 1>, %b: memref<?x3xf64, 1>) {
  md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      outs(%a : memref<?x3xf64, 1>) arity(2) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>):
    md_exec.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  md_exec.tuple_for %trios, %x, %cell coordinates(displacement(0, 1))
      outs(%b : memref<?x3xf64, 1>) arity(3) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>):
    md_exec.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  return
}

// The second loop writes the field that the first reads: two kernels.
//
// CHECK-LABEL: func.func @overwrites_input(
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         return
func.func @overwrites_input(%pairs: memref<?x?xi32, 1>,
                            %trios: memref<?x?xi32, 1>,
                            %x: memref<?x3xf64, 1>, %cell: !md.cell,
                            %y: memref<?x3xf64, 1>, %a: memref<?x3xf64, 1>) {
  md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      ins(%y : memref<?x3xf64, 1>)
      outs(%a : memref<?x3xf64, 1>) arity(2) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>, %y0: vector<3xf64>, %y1: vector<3xf64>):
    md_exec.yield %y0, %y1 : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  md_exec.tuple_for %trios, %x, %cell coordinates(displacement(0, 1))
      outs(%y : memref<?x3xf64, 1>) arity(3) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>):
    md_exec.yield %d, %d, %d : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  return
}

// The second loop reads the field that the first writes: two kernels.
//
// CHECK-LABEL: func.func @reads_output(
// CHECK:         gpu.launch
// CHECK:         gpu.launch
// CHECK:         return
func.func @reads_output(%pairs: memref<?x?xi32, 1>, %trios: memref<?x?xi32, 1>,
                        %x: memref<?x3xf64, 1>, %cell: !md.cell,
                        %a: memref<?x3xf64, 1>, %b: memref<?x3xf64, 1>) {
  md_exec.tuple_for %pairs, %x, %cell coordinates(displacement(0, 1))
      outs(%a : memref<?x3xf64, 1>) arity(2) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>):
    md_exec.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  md_exec.tuple_for %trios, %x, %cell coordinates(displacement(0, 1))
      ins(%a : memref<?x3xf64, 1>)
      outs(%b : memref<?x3xf64, 1>) arity(3) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>, %a0: vector<3xf64>, %a1: vector<3xf64>,
       %a2: vector<3xf64>):
    md_exec.yield %a0, %a1, %a2 : vector<3xf64>, vector<3xf64>, vector<3xf64>
  } : memref<?x?xi32, 1>, memref<?x3xf64, 1>
  return
}
