// RUN: mdir-opt %s -split-input-file -verify-diagnostics

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.particle_set @other

  func.func @direct_cross_set(
      %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
    // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
    %b = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64, #md_exec.set_layout<@other, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.particle_set @other

  func.func @unknown_after_select(
      %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>,
      %choose: i1) {
    %plain = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64>
    %selected = arith.select %choose, %plain, %plain : memref<?xf64>
    // expected-error@+1 {{cannot infer the source set of a cast to set @other}}
    %b = memref.cast %selected : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@other, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms

  func.func @unknown_block_argument(%plain: memref<?xf64>) {
    // expected-error@+1 {{cannot infer the source set of a cast to set @atoms}}
    %a = memref.cast %plain : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms

  func.func @declared_input_boundary(%plain: memref<?xf64>) {
    %a = memref.cast %plain {md_exec.input_set_boundary} : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.particle_set @other

  func.func @boundary_cannot_be_retagged(%input: memref<?xf64>) {
    %atoms = memref.cast %input {md_exec.input_set_boundary} : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    %plain = memref.cast %atoms : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64>
    // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
    %other = memref.cast %plain : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@other, 1>>
    return
  }
}
