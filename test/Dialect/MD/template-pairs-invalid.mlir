// RUN: mdir-opt %s -split-input-file -verify-diagnostics
md.particle_set @atoms
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
md.tuple_set @water on(@atoms) arity(4) orientation(ordered)
md.function @overlap(%types: !md.field<@atoms, i32>, %water: !md.relation<@atoms, 4, ordered, @water>) {
  // expected-error@+1 {{expected a declared disjoint molecule tuple set}}
  %ww, %sw, %ss = md.template_pairs %types, %water : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @water>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return
}
// -----
md.particle_set @atoms
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
md.function @real_payload(%types: !md.field<@atoms, f64>, %water: !md.relation<@atoms, 4, ordered, @water>) {
  // expected-error@+1 {{expected a scalar i32 universe field; only its cardinality is used}}
  %ww, %sw, %ss = md.template_pairs %types, %water : (!md.field<@atoms, f64>, !md.relation<@atoms, 4, ordered, @water>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return
}
// -----
md.particle_set @atoms
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
md.tuple_set @water on(@atoms) arity(4) orientation(ordered) disjoint
md.function @wrong_domain(%types: !md.field<@atoms, i32>, %water: !md.relation<@atoms, 4, ordered, @water>) {
  // expected-error@+1 {{expected named WW(6, ordered), SW(4, ordered), SS(2, unordered) domains of the universe}}
  %ww, %sw, %ss = md.template_pairs %types, %water : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @water>) -> (!md.relation<@atoms, 8, ordered>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return
}
// -----
md.particle_set @atoms
md.tuple_set @ww on(@atoms) arity(6) orientation(ordered)
md.tuple_set @sw on(@atoms) arity(4) orientation(ordered)
md.tuple_set @ss on(@atoms) arity(2) orientation(unordered)
md.particle_set @other
md.tuple_set @water on(@other) arity(4) orientation(ordered) disjoint
md.function @wrong_set(%types: !md.field<@atoms, i32>, %water: !md.relation<@other, 4, ordered, @water>) {
  // expected-error@+1 {{expected named ordered three- or four-site molecules of the universe}}
  %ww, %sw, %ss = md.template_pairs %types, %water : (!md.field<@atoms, i32>, !md.relation<@other, 4, ordered, @water>) -> (!md.relation<@atoms, 6, ordered, @ww>, !md.relation<@atoms, 4, ordered, @sw>, !md.relation<@atoms, 2, unordered, @ss>)
  md.return
}
