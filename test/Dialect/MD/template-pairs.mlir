// RUN: mdir-opt %s --verify-each | FileCheck %s
// These domains are structural; no positions, cell or hidden real payload.
md.particle_set @atoms
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
// CHECK-LABEL: md.function @domains
// CHECK: md.template_pairs
// CHECK-SAME: -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
md.function @domains(%types: !md.field<@atoms, i32>, %water: !md.relation<@atoms, 4, ordered, @water>, %excluded: !md.relation<@atoms, 2, unordered, @excluded>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>) {
  %ww, %sw, %ss = md.template_pairs %types, %water exclude(%excluded) : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @water>, !md.relation<@atoms, 2, unordered, @excluded>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return %ww, %sw, %ss : !md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>
}

md.tuple_set @three on(@atoms) arity(3) orientation(ordered) disjoint
// CHECK-LABEL: md.function @three_site
// CHECK: md.template_pairs
md.function @three_site(%types: !md.field<@atoms, i32>, %water: !md.relation<@atoms, 3, ordered, @three>) {
  %ww, %sw, %ss = md.template_pairs %types, %water : (!md.field<@atoms, i32>, !md.relation<@atoms, 3, ordered, @three>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return
}
