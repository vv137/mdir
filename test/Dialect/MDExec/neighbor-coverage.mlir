// RUN: mdir-opt %s | mdir-opt -o /dev/null

!vec = !md.field<@atoms, 3 x f64>
!nl = !mdrt.neighbors<@atoms>
!cells = !mdrt.cells<@atoms>
md.particle_set @atoms

// Both the initial list and every yielded list cover the requested cutoff.
// They need not have identical built cutoffs. The result includes zero trips.
func.func @loop_paths(%x: !vec, %cell: !md.cell, %count: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell cutoff(2.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %final = scf.for %i = %c0 to %count step %c1 iter_args(%neighbors = %initial) -> !nl {
    %value = md_exec.pair_for %neighbors, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    %next = md_exec.refresh_neighbors %neighbors, %x, %cell cutoff(2.5) skin(0.0)
        cell_width(1.0) policy(always) : !nl, !vec
    scf.yield %next : !nl
  }
  %value = md_exec.pair_for %final, %x, %cell reduce(%zero : f64) cutoff(1.5) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}

// Identity-carried neighbor and center values have anchored initial facts.
func.func @identity_centers(%x: !vec, %cell: !md.cell, %count: index, %owned: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %initial = md_exec.build_neighbors %cells, %x, %cell centers(%owned)
      cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
      : !cells, !vec -> !nl
  %final:2 = scf.for %i = %c0 to %count step %c1
      iter_args(%neighbors = %initial, %centers = %owned) -> (!nl, index) {
    %value = md_exec.pair_for %neighbors, %x, %cell centers(%centers)
        reduce(%zero : f64) cutoff(1.5) images(explicit) policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    scf.yield %neighbors, %centers : !nl, index
  }
  %value = md_exec.pair_for %final#0, %x, %cell centers(%final#1)
      reduce(%zero : f64) cutoff(1.5) images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}

// Every call site must establish the fact, with exact center substitution.
func.func private @consume(%neighbors: !nl, %x: !vec, %cell: !md.cell, %centers: index) -> f64 {
  %zero = arith.constant 0.0 : f64
  %value = md_exec.pair_for %neighbors, %x, %cell centers(%centers)
      reduce(%zero : f64) cutoff(1.5) images(explicit) policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %r2 : f64
  } : !nl, !vec -> f64
  return %value : f64
}

func.func private @forward(%neighbors: !nl, %x: !vec, %cell: !md.cell, %centers: index) -> f64 {
  %value = func.call @consume(%neighbors, %x, %cell, %centers) : (!nl, !vec, !md.cell, index) -> f64
  return %value : f64
}

func.func @first_call(%x: !vec, %cell: !md.cell, %owned: index) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%owned)
      cutoff(2.0) skin(0.0) kind(matrix) width(64) images(explicit)
      : !cells, !vec -> !nl
  %value = func.call @forward(%nl, %x, %cell, %owned) : (!nl, !vec, !md.cell, index) -> f64
  return %value : f64
}

func.func @second_call(%x: !vec, %cell: !md.cell, %owned: index) -> f64 {
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %nl = md_exec.build_neighbors %cells, %x, %cell centers(%owned)
      cutoff(2.5) skin(0.0) kind(matrix) width(64) images(explicit)
      : !cells, !vec -> !nl
  %value = func.call @forward(%nl, %x, %cell, %owned) : (!nl, !vec, !md.cell, index) -> f64
  return %value : f64
}

// A carried permutation is safe when every initial list in its orbit covers.
func.func @permuted_paths(%x: !vec, %cell: !md.cell, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %cells = md_exec.build_cells %x, %cell width(1.0) : !vec -> !cells
  %first = md_exec.build_neighbors %cells, %x, %cell cutoff(2.0) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %second = md_exec.build_neighbors %cells, %x, %cell cutoff(2.5) skin(0.0)
      kind(matrix) width(64) : !cells, !vec -> !nl
  %result:2 = scf.for %i = %c0 to %count step %c1
      iter_args(%a = %first, %b = %second) -> (!nl, !nl) {
    %value = md_exec.pair_for %a, %x, %cell reduce(%zero : f64) cutoff(1.5)
        policy(directed, owner_only) {
    ^bb0(%r2: f64, %d: vector<3xf64>):
      md_exec.yield %r2 : f64
    } : !nl, !vec -> f64
    scf.yield %b, %a : !nl, !nl
  }
  return
}
