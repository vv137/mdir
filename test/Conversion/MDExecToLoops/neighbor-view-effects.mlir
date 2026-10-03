// A checked import reads the row buffers. Reinitializing them after their
// previous consumer finishes requires another check, even with the same SSA
// buffer handles. CSE must not discard that second checked import.
// RUN: mdir-opt %s --cse | FileCheck %s
md.particle_set @atoms
func.func private @consume(!mdrt.neighbors<@atoms>)
func.func @check_again(%counts: memref<?xi32>, %entries: memref<?x?xi32>,
                      %n: index, %new_count: i32) {
  %zero = arith.constant 0 : index
  %a = md_exec.neighbor_view %counts, %entries local_size(%n)
      : memref<?xi32>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>
  func.call @consume(%a) : (!mdrt.neighbors<@atoms>) -> ()
  memref.store %new_count, %counts[%zero] : memref<?xi32>
  %b = md_exec.neighbor_view %counts, %entries local_size(%n)
      : memref<?xi32>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>
  func.call @consume(%b) : (!mdrt.neighbors<@atoms>) -> ()
  return
}
// CHECK-COUNT-2: md_exec.neighbor_view
