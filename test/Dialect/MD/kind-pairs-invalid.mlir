// RUN: mdir-opt %s -split-input-file -verify-diagnostics
// The refusals of md.kind_pairs and #md.rigid_kind (D[kind-pairs]), one
// case for each.
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// The universe supplies the particle set only.
md.function @universe(%types: !md.field<@atoms, f64>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected a scalar i32 universe field; only its particle set is used}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, f64>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// One descriptor for each kind.
md.function @descriptors(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected one #md.rigid_kind for each kind, 1, not 2}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>, #md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
md.tuple_set @three on(@atoms) arity(3) orientation(ordered) disjoint
// The wrong site count: a kind of three particles described with four places.
md.function @sites(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 3, ordered, @three>) {
  // expected-error@+1 {{expected kind 0 to be ordered tuples of its 4 places, as its #md.rigid_kind holds}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 3, ordered, @three>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
md.tuple_set @t20 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @t21 on(@atoms) arity(4) orientation(ordered)
// Two kinds are two tuple sets.
md.function @shared(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected the kinds to be distinct tuple sets}}
  %r:6 = md.kind_pairs %types, kinds(%k, %k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>, #md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @t20>, !md.relation<@atoms, 4, ordered, @t21>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// K = 1 has three segments.
md.function @count(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected (K + 1)(K + 2)/2 = 3 segments for K = 1 kinds, not 2}}
  %r:2 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// s(1, 0) pairs the rest (one member) with the kind (three frame atoms).
md.function @arity(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected segment 1 = s(1, 0) of arity 4, ordered}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// The rest with itself is unordered.
md.function @orientation(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected segment 2 = s(1, 1) of arity 2, unordered}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, ordered, @s00>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// A segment is not the excluded pairs.
md.function @alias(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>, %ex: !md.relation<@atoms, 2, unordered, @excluded>) {
  // expected-error@+1 {{a segment cannot alias a kind or the excluded pairs}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] exclude(%ex) : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>, !md.relation<@atoms, 2, unordered, @excluded>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @excluded>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
md.tuple_set @anchors on(@atoms) arity(2) orientation(ordered)
// The anchors' radius is positive.
md.function @radius(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>, %an: !md.relation<@atoms, 2, ordered, @anchors>) {
  // expected-error@+1 {{expected a positive finite anchor radius}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] anchor(%an) radius(0.0) : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>, !md.relation<@atoms, 2, ordered, @anchors>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// A kind's copies share no particle: its tuple set is declared disjoint.
md.function @overlap(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected each kind to be a declared disjoint tuple set}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.14772060947071208], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// A kind has a frame.
md.function @empty(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected a nonempty frame}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [], sites = [0, 1, 2, 3], weights = [], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// Each place is a frame atom or a site, once.
md.function @twice(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected the frame and the sites to hold each place 0 to 3 once}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [2], weights = [0.5, 0.25, 0.25], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// The frame is ascending; its first place is the anchor.
md.function @order(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected the frame's places in ascending order}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [1, 0, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// One weight for each site and frame atom.
md.function @weights(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected 3 weights, one for each site and frame atom}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.5], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// R2: the weights of a site sum to 1, here within 6 eps for three weights;
// one hydrogen weight of 0.15 instead of 0.1477 is refused.
md.function @sum(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected the weights of each site to sum to 1}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.70455878105857583, 0.14772060947071208, 0.15], extent = 0.1>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}

// -----
md.particle_set @atoms
md.tuple_set @four on(@atoms) arity(4) orientation(ordered) disjoint
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @s00 on(@atoms) arity(6) orientation(ordered)
md.tuple_set @s10 on(@atoms) arity(4) orientation(ordered)
md.tuple_set @s11 on(@atoms) arity(2) orientation(unordered)
// The covering radius is positive.
md.function @extent(%types: !md.field<@atoms, i32>, %k: !md.relation<@atoms, 4, ordered, @four>) {
  // expected-error@+1 {{expected a positive finite f64 extent}}
  %r:3 = md.kind_pairs %types, kinds(%k) [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.0>] : (!md.field<@atoms, i32>, !md.relation<@atoms, 4, ordered, @four>) -> (!md.relation<@atoms, 6, ordered, @s00>, !md.relation<@atoms, 4, ordered, @s10>, !md.relation<@atoms, 2, unordered, @s11>)
  md.return
}
