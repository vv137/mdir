// RUN: mdir-opt %s -split-input-file -verify-diagnostics

func.func @wrong_relation(%nl: !mdrt.neighbors<@atoms>,
                          %x: !md.field<@atoms, 3 x f64>, %cell: !md.cell) {
  // expected-error@+1 {{expected the result to have type '!md.relation<@atoms, 3, reversal>', got '!md.relation<@atoms, 3, reversal, @angles>'}}
  %t = md_exec.build_triplets %nl, %x, %cell cutoff(1.2)
         : !mdrt.neighbors<@atoms>, !md.field<@atoms, 3 x f64>
           -> !md.relation<@atoms, 3, reversal, @angles>
  return
}

// -----

func.func @wrong_members(%nl: !mdrt.neighbors<@atoms>, %x: memref<?x3xf64>,
                         %cell: !md.cell) {
  // expected-error@+1 {{expected the members of triplets, memref<?x3xi32> on the host, got 'memref<?x2xi32>'}}
  %m = md_exec.build_triplets %nl, %x, %cell cutoff(1.2)
         : !mdrt.neighbors<@atoms>, memref<?x3xf64> -> memref<?x2xi32>
  return
}
