// The refusals of the writer of a list of units and of the refresh that
// takes its test (D[md-alg-segments-writer]): the refresh takes `moved` and
// `stale` from one writer of its own list, positions, cell, and reaches; the
// writer forms the frames from positions in f64.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// A refresh whose test is not a writer's.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp, %m: i1) {
  // expected-error @+1 {{takes 'moved' and 'stale' from the writer of its list (md_exec.write_unit_frames) alone}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// `stale` without `moved`.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp) {
  %m, %s = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  // expected-error @+1 {{takes 'stale' without 'moved'}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// `stale` of one writer and `moved` of another.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp) {
  %m, %s = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  %m2, %s2 = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  // expected-error @+1 {{takes 'moved' and 'stale' from the writer of its list (md_exec.write_unit_frames) alone}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) stale(%s2) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// The test of a writer at other positions.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %y: !x, %cell: !md.cell, %l: !sp) {
  %m, %s = md_exec.write_unit_frames %l at(%y, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  // expected-error @+1 {{takes the test of a writer of another list, positions, or cell}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// The test of a writer of another list.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp, %k: !sp) {
  %m, %s = md_exec.write_unit_frames %k at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  // expected-error @+1 {{takes the test of a writer of another list, positions, or cell}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// The test of a writer in another cell.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %other: !md.cell, %l: !sp) {
  %m, %s = md_exec.write_unit_frames %l at(%x, %other) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  // expected-error @+1 {{takes the test of a writer of another list, positions, or cell}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// The test of a writer of another pruned reach.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp) {
  %m, %s = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.06) : !sp, !x
  // expected-error @+1 {{takes the test of a writer of other reaches}}
  %r = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%m) stale(%s) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x
  return
}

// -----

// A writer that forms the frames from positions in f32.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x32, %cell: !md.cell, %l: !sp) {
  // expected-error @+1 {{expected positions of 3 components in f64, from which the frames are formed}}
  %m, %s = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !x32
  return
}

// -----

// A writer with a pruned reach beyond the skin.
!sp = !mdrt.segmented_pairs<@atoms, kinds [#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [0.5, 0.25, 0.25], extent = 0.1>], segments [(0, 0) unique, minimum, removed, grouped_by(group), (1, 0) unique, minimum, removed, grouped_by(group), (1, 1) unique, minimum, removed, grouped_by(group)]>
md.particle_set @atoms
!x = !md.field<@atoms, 3 x f64>
!x32 = !md.field<@atoms, 3 x f32>
func.func @f(%x: !x, %cell: !md.cell, %l: !sp) {
  // expected-error @+1 {{expected finite skins with 0 <= prune_skin <= skin}}
  %m, %s = md_exec.write_unit_frames %l at(%x, %cell) cutoff(0.9) skin(0.1) prune_skin(0.2) : !sp, !x
  return
}
