// An op of no registered dialect beside md_dist ops (#383): the pass took
// the dialect of every op it walked and of every user of its values, which
// is null for an unregistered op, and asserted. It now treats such an op
// as one of another dialect: one that does not use a value of md_dist is
// left as it is, and one that does is refused with the pass's diagnostic.
//
// RUN: mdir-opt --allow-unregistered-dialect --md-exec-assign-algebra --convert-md-dist-to-loops %s \
// RUN:   --split-input-file -verify-diagnostics | FileCheck %s

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

// CHECK-LABEL: func.func @f
// CHECK-NOT: md_dist.partition
// CHECK: "test.unknown"() : () -> ()
func.func @f(%x: !md.field<@atoms, 3 x f64>, %l: f64) {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  "test.unknown"() : () -> ()
  return
}

// -----

md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])

func.func @g(%x: !md.field<@atoms, 3 x f64>, %l: f64) {
  %cell = md.orthorhombic_cell %l, %l, %l
  %lay = md_dist.partition %x, %cell : !md.field<@atoms, 3 x f64> -> !md_dist.layout<@dd>
  // expected-error @below {{uses a value of md_dist, which the reference executor cannot hand to an op of another dialect}}
  "test.unknown"(%lay) : (!md_dist.layout<@dd>) -> ()
  return
}
