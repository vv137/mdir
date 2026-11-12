// What the verifier refuses of host(lanes(W)) (D402): a
// width that is not a power of two from 1 to 64, and the lanes on a loop
// whose buffers are on a device; and the kernel types that the lanes would
// refuse, which the verifier refuses first.
//
// RUN: mdir-opt %s -split-input-file -verify-diagnostics
// RUN: not mdir-opt %s -split-input-file --md-exec-assign-algebra \
// RUN:   --md-exec-assign-storage=memory=device 2>&1 | FileCheck %s --check-prefix=DEVICE

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @zero(%x: !vec, %cell: !md.cell) -> f32 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(48) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %z = arith.constant 0.0 : f32
  // expected-error @+1 {{cannot take host(lanes(0)): the lanes are a power of two from 1 to 64, not 0}}
  %e = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) host(lanes(0)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !mdrt.neighbors<@atoms>, !vec -> f32
  return %e : f32
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @wide(%x: !vec, %cell: !md.cell) -> f32 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(48) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %z = arith.constant 0.0 : f32
  // expected-error @+1 {{cannot take host(lanes(128)): the lanes are a power of two from 1 to 64, not 128}}
  %e = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) host(lanes(128)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !mdrt.neighbors<@atoms>, !vec -> f32
  return %e : f32
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

// The squared distance of the kernel is of the type of the positions; the
// lanes' refusal of a type other than f32 and f64 is not reached.
func.func @integer(%x: !vec, %cell: !md.cell) -> f32 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(48) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %z = arith.constant 0.0 : f32
  // expected-error @+1 {{expected kernel argument 0 to have type 'f64', got 'i32'}}
  %e = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) host(lanes(8)) {
  ^bb0(%r2: i32, %d: vector<3xf32>):
    md_exec.yield %z : f32
  } : !mdrt.neighbors<@atoms>, !vec -> f32
  return %e : f32
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

// On a device: the storage of md-exec-assign-storage=memory=device carries
// the attribute to buffers with a memory space, which the verifier refuses
// (DEVICE above; the default RUN line takes the value form, which is fine).
func.func @device(%x: !vec, %cell: !md.cell) -> f32 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !mdrt.cells<@atoms>
  %nl = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(48) : !mdrt.cells<@atoms>, !vec -> !mdrt.neighbors<@atoms>
  %z = arith.constant 0.0 : f32
  %e = md_exec.pair_for %nl, %x, %cell reduce(%z : f32) cutoff(1.0)
      exchange [symmetric] policy(directed, owner_only) host(lanes(8)) {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_exec.yield %r2 : f32
  } : !mdrt.neighbors<@atoms>, !vec -> f32
  return %e : f32
}

// DEVICE: error: 'md_exec.pair_for' op cannot take host(lanes(8)): its buffers are on a device, and the lanes are a strategy of the host
