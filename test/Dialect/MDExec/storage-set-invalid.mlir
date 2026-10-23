// RUN: mdir-opt %s --convert-md-exec-to-loops -split-input-file -verify-diagnostics

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_loop(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>,
    %b: memref<?xf64, #md_exec.set_layout<@other, 1>>) {
  // expected-error@+1 {{expected a buffer of set @atoms, got a buffer of set @other}}
  md_exec.particle_for ins(%a, %b : memref<?xf64, #md_exec.set_layout<@atoms, 1>>, memref<?xf64, #md_exec.set_layout<@other, 1>>) outs(%a : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  ^bb0(%x: f64, %y: f64):
    md_exec.yield %y : f64
  }
  return
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_reuse(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
  %b = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64, #md_exec.set_layout<@other, 1>>
  return
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_reuse_through_plain(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  %plain = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64>
  // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
  %b = memref.cast %plain : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@other, 1>>
  return
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_reuse_through_view(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  %plain = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64>
  %view = memref.subview %plain[0] [10] [1] : memref<?xf64> to memref<10xf64, strided<[1]>>
  // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
  %b = memref.cast %view : memref<10xf64, strided<[1]>> to memref<10xf64, #md_exec.set_layout<@other, 1>>
  return
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms

  func.func @untagged_storage(%a: memref<?xf64>) {
    // expected-error@+1 {{expected tagged buffers in the storage form}}
    md_exec.particle_for ins(%a : memref<?xf64>) outs(%a : memref<?xf64>) {
    ^bb0(%x: f64):
      md_exec.yield %x : f64
    }
    return
  }
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @unknown_set_after_select(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>,
    %choose: i1) {
  %plain = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64>
  %selected = arith.select %choose, %plain, %plain : memref<?xf64>
  // expected-error@+1 {{cannot infer the source set of a cast to set @other}}
  %b = memref.cast %selected : memref<?xf64> to memref<?xf64, #md_exec.set_layout<@other, 1>>
  return
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.particle_set @items
  md.particle_set @other

  func.func @scatter_cross_set(
      %values: memref<?xf64, #md_exec.set_layout<@items, 1>>,
      %indices: memref<?xi32, #md_exec.set_layout<@other, 1>>,
      %dest: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
    // expected-error@+1 {{expected a buffer of set @items, got a buffer of set @other}}
    md_exec.scatter_add %values at %indices into %dest
        : memref<?xf64, #md_exec.set_layout<@items, 1>>, memref<?xi32, #md_exec.set_layout<@other, 1>>, memref<?xf64, #md_exec.set_layout<@atoms, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.tuple_set @angles on(@atoms) arity(2) orientation(ordered)

  func.func @missing_tuple_set(
      %inc: memref<?xi32, #md_exec.set_layout<@atoms, 1>>,
      %x: memref<?x3xf64, #md_exec.set_layout<@atoms, 2>>,
      %cell: !md.cell,
      %param: memref<?xf64, #md_exec.set_layout<@angles, 1>>) {
    // expected-error@+1 {{expected 'md_exec.tuple_set' for tuple parameters}}
    md_exec.tuple_for %inc, %x, %cell coordinates(displacement(0, 1))
        tuple(%param : memref<?xf64, #md_exec.set_layout<@angles, 1>>)
        arity(2) {
    ^bb0(%d: vector<3xf64>, %p: f64):
      md_exec.yield
    } : memref<?xi32, #md_exec.set_layout<@atoms, 1>>, memref<?x3xf64, #md_exec.set_layout<@atoms, 2>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms

  func.func @wrong_tuple_members(
      %members: memref<?x3xi32, #md_exec.set_layout<@other, 2>>,
      %n: index) {
    // expected-error@+1 {{expected a buffer of set @angles, got a buffer of set @other}}
    %inc = md_exec.build_incidence %members size(%n) {md_exec.tuple_set = @angles}
        : memref<?x3xi32, #md_exec.set_layout<@other, 2>> -> memref<?xi32, #md_exec.set_layout<@atoms, 1>>
    return
  }
}

// -----

module attributes {md_exec.typed_storage} {
  md.particle_set @atoms
  md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)

  func.func @missing_builder_tuple_set(
      %members: memref<?x3xi32, #md_exec.set_layout<@angles, 2>>,
      %n: index) {
    // expected-error@+1 {{expected 'md_exec.tuple_set' for tagged members}}
    %inc = md_exec.build_incidence %members size(%n)
        : memref<?x3xi32, #md_exec.set_layout<@angles, 2>> -> memref<?xi32, #md_exec.set_layout<@atoms, 1>>
    return
  }
}
