// The layout contract of !mdrt.neighbors (D413): the four
// words, always spelled and printed, and the normalization of a sort by a
// total key.
//
// RUN: mdir-opt %s | FileCheck %s
// RUN: mdir-opt %s | mdir-opt | FileCheck %s

md.particle_set @atoms

// CHECK-LABEL: func.func private @matrix(!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>)
func.func private @matrix(
    !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>)

// CHECK-LABEL: func.func private @groups(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
func.func private @groups(
    !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)

// CHECK-LABEL: func.func private @rows(!mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>)
func.func private @rows(
    !mdrt.neighbors<@atoms, directed, explicit, masked, stable(owner, search)>)

// The weaker organizations.
// CHECK-LABEL: func.func private @weaker(!mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>, !mdrt.neighbors<@atoms, directed, minimum, masked, none>, !mdrt.neighbors<@atoms, directed, minimum, removed, stable(owner, item)>)
func.func private @weaker(
    !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>,
    !mdrt.neighbors<@atoms, directed, minimum, masked, none>,
    !mdrt.neighbors<@atoms, directed, minimum, removed, stable(owner, item)>)

// A sort by a total key leaves no ties: sorted_within is read as stable
// (D409).
// CHECK-LABEL: func.func private @normalized(!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, item)>, !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>)
func.func private @normalized(
    !mdrt.neighbors<@atoms, directed, minimum, masked, sorted_within(owner, item)>,
    !mdrt.neighbors<@atoms, directed, minimum, masked, sorted_within(owner, search)>)
