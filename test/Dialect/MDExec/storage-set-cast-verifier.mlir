// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// The marker remains a unit attribute of the module itself.
// expected-error@+1 {{expects 'md_exec.typed_storage' on a module}}
module attributes {md_exec.typed_storage = true} {
}

// -----

module {
  // expected-error@+1 {{expects 'md_exec.typed_storage' on a module}}
  func.func @misplaced_typed_storage() attributes {md_exec.typed_storage} {
    return
  }
}

// -----

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

// The boundary marker is checked even without the typed-storage module flag.
module {
  md.particle_set @atoms

  func.func @bad_boundary_value(%plain: memref<?xf64>) {
    // expected-error@+1 {{expects 'md_exec.input_set_boundary' as a unit attribute on memref.cast}}
    %a = memref.cast %plain {md_exec.input_set_boundary = true} : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    return
  }
}

// -----

module {
  // expected-error@+1 {{expects 'md_exec.input_set_boundary' as a unit attribute on memref.cast}}
  func.func @misplaced_marker() attributes {md_exec.input_set_boundary} {
    return
  }
}

// -----

module {
  func.func @plain_target(%plain: memref<?xf64>) {
    // expected-error@+1 {{input set boundary must tag a plain buffer}}
    %a = memref.cast %plain {md_exec.input_set_boundary} : memref<?xf64> to memref<?xf64>
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
