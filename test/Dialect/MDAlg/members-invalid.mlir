// The refusals of the verifier of md_alg.bin_members (D409),
// one case each: an unordered visit combines only in forms whose value does
// not depend on the order (integer combiners, slots of the item, additions
// to counters), and reads no side output.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

!d = memref<?xi32, 1>
func.func @by_unordered(%s: !d, %o: !d, %by: !d, %a: index, %b: index) {
  // expected-error @+1 {{visits by 'by' only in the order 'sorted' or 'stable'}}
  md_alg.bin_members %s[%a, %b] items %o visit unordered by %by : !d : !d, !d {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @combiners(%s: !d, %o: !d, %a: index, %b: index, %z: i32) {
  // expected-error @+1 {{takes one combiner for each initial value}}
  %r = "md_alg.bin_members"(%s, %a, %b, %o, %z) <{visit = 0 : i32, combine = ["addi", "ori"], operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 1>}> ({
  ^bb0(%item: index):
    "md_alg.yield"(%z) : (i32) -> ()
  }) : (!d, index, index, !d, i32) -> i32
  return
}

// -----

!d = memref<?xi32, 1>
func.func @results(%s: !d, %o: !d, %a: index, %b: index, %z: i32) {
  // expected-error @+1 {{has results of the types of its initial values}}
  %r = "md_alg.bin_members"(%s, %a, %b, %o, %z) <{visit = 0 : i32, combine = ["addi"], operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 1>}> ({
  ^bb0(%item: index):
    "md_alg.yield"(%z) : (i32) -> ()
  }) : (!d, index, index, !d, i32) -> i64
  return
}

// -----

!d = memref<?xi32, 1>
func.func @combiner_type(%s: !d, %o: !d, %a: index, %b: index, %z: i32) {
  // expected-error @+1 {{cannot combine 'i32' by 'addf'}}
  %r = md_alg.bin_members %s[%a, %b] items %o visit stable
      init(%z : i32) combine ["addf"] : !d, !d {
  ^bb0(%item: index):
    md_alg.yield %z : i32
  } -> i32
  return
}

// -----

!d = memref<?xi32, 1>
func.func @float_unordered(%s: !d, %o: !d, %a: index, %b: index, %f: f32) {
  // expected-error @+1 {{with the visit 'unordered' combines integers alone: 'addf' on 'f32' depends on the order of the items}}
  %r = md_alg.bin_members %s[%a, %b] items %o visit unordered
      init(%f : f32) combine ["addf"] : !d, !d {
  ^bb0(%item: index):
    md_alg.yield %f : f32
  } -> f32
  return
}

// -----

!d = memref<?xi32, 1>
func.func @argument(%s: !d, %o: !d, %a: index, %b: index) {
  // expected-error @+1 {{takes a region of one index argument, the item}}
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: i32):
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @terminator(%s: !d, %o: !d, %a: index, %b: index) {
  // expected-error @+1 {{ends its region in 'md_alg.yield'}}
  "md_alg.bin_members"(%s, %a, %b, %o) <{visit = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0>}> ({
  ^bb0(%item: index):
    "test.end"() : () -> ()
  }) : (!d, index, index, !d) -> ()
  return
}

// -----

!d = memref<?xi32, 1>
func.func @yield_types(%s: !d, %o: !d, %a: index, %b: index, %z: i32) {
  // expected-error @+1 {{yields one value of the type of each initial value}}
  %r = md_alg.bin_members %s[%a, %b] items %o visit unordered
      init(%z : i32) combine ["addi"] : !d, !d {
  ^bb0(%item: index):
    md_alg.yield
  } -> i32
  return
}

// -----

!d = memref<?xi32, 1>
func.func @own_buffer(%s: !d, %o: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: index):
    // expected-error @+1 {{uses a buffer of the 'md_alg.bin_members' of its region}}
    %v = memref.load %o[%item] : !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @own_buffer_view(%s: !d, %o: !d, %a: index, %b: index) {
  // A view of a buffer of the op, made outside the region (#874).
  %view = memref.cast %o : !d to !d
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: index):
    // expected-error @+1 {{uses a buffer of the 'md_alg.bin_members' of its region}}
    %v = memref.load %view[%item] : !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @read_side_output(%s: !d, %o: !d, %m: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%m : !d)
      : !d, !d {
  ^bb0(%item: index):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{reads a side output of the region of 'md_alg.bin_members', which another item may have written}}
    %v = memref.load %m[%c0] : !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @read_side_output_view(%s: !d, %o: !d, %m: !d, %a: index,
                                 %b: index) {
  // A view of a side output, made outside the region (#874).
  %view = memref.memory_space_cast %m : !d to memref<?xi32>
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%m : !d)
      : !d, !d {
  ^bb0(%item: index):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{reads a side output of the region of 'md_alg.bin_members', which another item may have written}}
    %v = memref.load %view[%c0] : memref<?xi32>
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @undeclared(%s: !d, %o: !d, %m: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    // expected-error @+1 {{writes a buffer that the region does not declare in 'writes'}}
    memref.store %v, %m[%item] : !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @not_a_slot(%s: !d, %o: !d, %m: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%m : !d)
      : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{writes a side output at a place that is not a slot of the item}}
    memref.store %v, %m[%c0] : !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @not_an_addition(%s: !d, %o: !d, %m: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered adds(%m : !d)
      : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{is not an addition to a counter that the region declares in 'adds'}}
    %old = memref.atomic_rmw maxs %v, %m[%c0] : (i32, !d) -> i32
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @unknown_effects(%s: !d, %o: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: index):
    // expected-error @+1 {{has effects that the region of 'md_alg.bin_members' cannot state}}
    "test.effects"() : () -> ()
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @other_write(%s: !d, %o: !d, %x: !d, %y: !d, %a: index, %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered : !d, !d {
  ^bb0(%item: index):
    // expected-error @+1 {{writes memory in the region of 'md_alg.bin_members' other than a declared side output}}
    memref.copy %x, %y : !d to !d
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @side_output_is_items(%s: !d, %o: !d, %a: index, %b: index) {
  // expected-error @+1 {{declares as a side output a buffer that it reads, or another side output, or a view of one}}
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%o : !d)
      : !d, !d {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @side_output_views_items(%s: !d, %o: !d, %a: index, %b: index) {
  %view = memref.memory_space_cast %o : !d to memref<?xi32>
  // expected-error @+1 {{declares as a side output a buffer that it reads, or another side output, or a view of one}}
  md_alg.bin_members %s[%a, %b] items %o visit unordered
      writes(%view : memref<?xi32>) : !d, !d {
  ^bb0(%item: index):
    %v = arith.index_cast %item : index to i32
    memref.store %v, %view[%item] : memref<?xi32>
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @two_side_outputs(%s: !d, %o: !d, %m: !d, %a: index, %b: index) {
  // expected-error @+1 {{declares as a side output a buffer that it reads, or another side output, or a view of one}}
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%m : !d)
      adds(%m : !d) : !d, !d {
  ^bb0(%item: index):
    md_alg.yield
  }
  return
}

// -----

!d = memref<?xi32, 1>
func.func @view_of_side_output(%s: !d, %o: !d, %m: !d, %a: index,
                               %b: index) {
  md_alg.bin_members %s[%a, %b] items %o visit unordered writes(%m : !d)
      : !d, !d {
  ^bb0(%item: index):
    // expected-error @+1 {{reads a side output of the region of 'md_alg.bin_members', which another item may have written}}
    %view = memref.cast %m : !d to !d
    %c0 = arith.constant 0 : index
    %v = memref.load %view[%c0] : !d
    md_alg.yield
  }
  return
}
