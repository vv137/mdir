// RUN: mdir-opt %s --split-input-file -verify-diagnostics -o /dev/null

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// The initial path and every yielded path must establish the obligation.
func.func @initial_cutoff(%x: !vec, %cell: !md.cell, %count: index, %owned: index, %other: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell
      cutoff(1.0) skin(0.0) kind(matrix) width(64) : !cells, !vec -> !nl
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%nl = %initial) -> (!nl) {
    // expected-error @+1 {{exceeds the cutoff that the neighbor structure was built with}}
    %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = md_exec.refresh_neighbors %nl, %x, %cell
        cutoff(2.0) skin(0.0) cell_width(1.0) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// The initial path and every yielded path must establish the obligation.
func.func @yielded_cutoff(%x: !vec, %cell: !md.cell, %count: index, %owned: index, %other: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.0) skin(0.0) kind(matrix) width(64) : !cells, !vec -> !nl
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%nl = %initial) -> (!nl) {
    // expected-error @+1 {{exceeds the cutoff that the neighbor structure was built with}}
    %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = md_exec.refresh_neighbors %nl, %x, %cell
        cutoff(1.0) skin(0.0) cell_width(1.0) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// The initial path and every yielded path must establish the obligation.
func.func @centers_mismatch(%x: !vec, %cell: !md.cell, %count: index, %owned: index, %other: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell centers(%owned)
      cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit) : !cells, !vec -> !nl
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%nl = %initial) -> (!nl) {
    // expected-error @+1 {{the centers of the loop are not those of the neighbor structure}}
    %value = md_exec.pair_for %nl, %x, %cell centers(%owned) reduce(%zero : f64) cutoff(1.5) images(explicit) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = md_exec.refresh_neighbors %nl, %x, %cell centers(%other)
        cutoff(2.0) skin(0.0) cell_width(1.0) images(explicit) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// The initial path and every yielded path must establish the obligation.
func.func @images_mismatch(%x: !vec, %cell: !md.cell, %count: index, %owned: index, %other: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell
      cutoff(2.0) skin(0.0) kind(matrix) width(64) : !cells, !vec -> !nl
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%nl = %initial) -> (!nl) {
    // expected-error @+1 {{the loop takes images(minimum), but the neighbor structure holds images(explicit)}}
    %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = md_exec.refresh_neighbors %nl, %x, %cell
        cutoff(2.0) skin(0.0) cell_width(1.0) images(explicit) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// A zero-trip result is its initial value, not the body rebuild.
func.func @zero_trip(%x: !vec, %cell: !md.cell) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %result = scf.for %i = %c0 to %c0 step %c1 iter_args(%nl = %initial) -> (!nl) {
    %next = md_exec.refresh_neighbors %nl, %x, %cell cutoff(2.0) skin(0.0)
        cell_width(1.0) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  // expected-error @+1 {{exceeds the cutoff that the neighbor structure was built with}}
  %value = md_exec.pair_for %result, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// A good call cannot excuse another caller's insufficient cutoff.
func.func private @consume(%nl: !nl, %x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{exceeds the cutoff that the neighbor structure was built with}}
  %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}
func.func @callers(%x: !vec, %cell: !md.cell) {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %good = md_exec.build_neighbors %cells, %x, %cell cutoff(2.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %small = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %first = func.call @consume(%good, %x, %cell) : (!nl, !vec, !md.cell) -> f64
  %second = func.call @consume(%small, %x, %cell) : (!nl, !vec, !md.cell) -> f64
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// expected-note @+1 {{unresolved coverage boundary}}
func.func @public_argument(%nl: !nl, %x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot prove neighbor coverage across public function boundary}}
  %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// expected-note @+1 {{unresolved coverage boundary}}
func.func private @no_calls(%nl: !nl, %x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot prove neighbor coverage across private function without known call sites boundary}}
  %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// expected-note @+1 {{unresolved coverage boundary}}
func.func private @recursive(%nl: !nl, %x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot prove neighbor coverage across recursive private function boundary}}
  %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  %again = func.call @recursive(%nl, %x, %cell) : (!nl, !vec, !md.cell) -> f64
  return %value : f64
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// expected-note @+1 {{unresolved coverage boundary}}
func.func private @address_taken(%nl: !nl, %x: !vec, %cell: !md.cell) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error @+1 {{cannot prove neighbor coverage across indirect or address-taken private function boundary}}
  %value = md_exec.pair_for %nl, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}
func.func @take_address() {
  %f = func.constant @address_taken : (!nl, !vec, !md.cell) -> f64
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// A carried permutation must establish the fact for the whole reachable orbit.
func.func @permuted_initial(%x: !vec, %cell: !md.cell, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %good = md_exec.build_neighbors %cells, %x, %cell cutoff(2.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %small = md_exec.build_neighbors %cells, %x, %cell cutoff(1.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %result:2 = scf.for %i = %c0 to %count step %c1
      iter_args(%a = %good, %b = %small) -> (!nl, !nl) {
    // expected-error @+1 {{exceeds the cutoff that the neighbor structure was built with}}
    %value = md_exec.pair_for %a, %x, %cell reduce(%zero : f64) cutoff(1.5)
        policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    scf.yield %b, %a : !nl, !nl
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

func.func @unrecognized_if(%x: !vec, %cell: !md.cell, %take: i1) {
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %good = md_exec.build_neighbors %cells, %x, %cell cutoff(2.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  // expected-note @+1 {{unresolved coverage boundary}}
  %chosen = scf.if %take -> (!nl) {
    scf.yield %good : !nl
  } else {
    scf.yield %good : !nl
  }
  // expected-error @+1 {{cannot prove neighbor coverage across scf.if boundary}}
  %value = md_exec.pair_for %chosen, %x, %cell reduce(%zero : f64) cutoff(1.5)
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return
}
