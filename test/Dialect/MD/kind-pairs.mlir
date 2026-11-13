// RUN: mdir-opt %s --verify-each | FileCheck %s
// RUN: mdir-opt %s --verify-each | mdir-opt --verify-each | FileCheck %s
// The segments of rule R3 (D[kind-pairs]) are structural: no positions,
// cell or hidden real payload. K kinds and the rest give (K+1)(K+2)/2
// segments, s(f, g) = f(f+1)/2 + g.
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @three on(@atoms) arity(3) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @anchors on(@atoms) arity(2) orientation(ordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
md.tuple_set @t10 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @t20 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @t21 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @t22 on(@atoms) arity(2) orientation(unordered)

// One kind of four places with a site, as #795's three classes.
// CHECK-LABEL: md.function @one
// CHECK: md.kind_pairs %{{.*}}, kinds(%{{.*}}) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.087343313000000005>] exclude(%{{.*}}) anchor(%{{.*}}) radius(1.100000e-01)
// CHECK-SAME: -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
md.function @one(%types: !md.field<@atoms, i32>, %kind: !md.relation<@atoms, 4, ordered, @four>, %excluded: !md.relation<@atoms, 2, unordered, @excluded>, %anchors: !md.relation<@atoms, 2, ordered, @anchors>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>) {
  %s00, %s10, %s11 = md.kind_pairs %types, kinds(%kind) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.087343313000000005>] exclude(%excluded) anchor(%anchors) radius(0.11) : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>, !md.relation<@atoms, 2, unordered, @excluded>, !md.relation<@atoms, 2, ordered, @anchors>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return %s00, %s10, %s11 : !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>
}

// Two kinds: the frames have three atoms each, so (1, 0) has arity 6 and
// (2, 0), (2, 1) arity 4; only the rest with itself is unordered.
// CHECK-LABEL: md.function @two
// CHECK: md.kind_pairs %{{.*}}, kinds(%{{.*}}, %{{.*}}) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 1.000000e-01>, #md.rigid_kind<frame = [0, 1, 2], sites = [3]
// CHECK-SAME: -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 6, ordered, @t10>, !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @t20>, !md.relation<@atoms, 4, ordered, @t21>, !md.relation<@atoms, 2, unordered, @t22>)
md.function @two(%types: !md.field<@atoms, i32>, %a: !md.relation<@atoms, 3, ordered, @three>, %b: !md.relation<@atoms, 4, ordered, @four>) {
  %r:6 = md.kind_pairs %types, kinds(%a, %b) [#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 0.1>, #md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 3, ordered, @three>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 6, ordered, @t10>, !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @t20>, !md.relation<@atoms, 4, ordered, @t21>, !md.relation<@atoms, 2, unordered, @t22>)
  md.return
}

// No kind: the rest alone, one segment of unordered pairs.
// CHECK-LABEL: md.function @none
// CHECK: md.kind_pairs %{{.*}}, kinds() [] : (!md.field<@atoms, i32>) -> !md.relation<@atoms, 2, unordered, @s11>
md.function @none(%types: !md.field<@atoms, i32>) {
  %s = md.kind_pairs %types, kinds() [] : (!md.field<@atoms, i32>) -> !md.relation<@atoms, 2, unordered, @s11>
  md.return
}
