// The refusals of the verifier of md_alg.traverse_pairs (D[md-alg-traverse]),
// one case each, each a realistic bug: a write outside the declared
// combination, a read of a destination, owner registers over a list that is
// not grouped by owner, a visit weaker than the algebra needs or than the list
// gives, the forms of the list, the counts of the declarations, and the
// region. Each case is a valid op with one change.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// A store into a destination in the region: the write is not through the combination.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @store_destination(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %c0 = arith.constant 0 : index
    %z = arith.constant 0.0 : f32
    // expected-error @+1 {{uses a destination of 'md_alg.traverse_pairs' in its region}}
    memref.store %z, %f[%c0, %c0] : !f
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A store into another buffer in the region: a write that no combination declares.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @store_other(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %t: memref<?xf32>) {
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{writes memory in the region of 'md_alg.traverse_pairs'}}
    memref.store %r2, %t[%c0] : memref<?xf32>
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A read of a destination through a view made outside the region.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @read_view(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = memref.subview %f[0, 0] [1, 3] [1, 1] : !f to memref<1x3xf32, strided<[3, 1]>, 1>
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %z0 = arith.constant 0 : index
    // expected-error @+1 {{uses a destination of 'md_alg.traverse_pairs' in its region}}
    %r = memref.load %v[%z0, %z0] : memref<1x3xf32, strided<[3, 1]>, 1>
    %e = arith.extf %r : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A destination that is a view of a buffer that the op reads.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @destination_views_ins(%nl: !groups, %x: !x, %cell: !md.cell, %q: !q, %u0: f64) {
  %f = memref.cast %q : !q to memref<?xf32, 1>
  // expected-error @+1 {{has destination 0, which is or views a buffer that the op reads}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) ins(%q : !q) outs(%f : memref<?xf32, 1>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %qi, %e : f32, f64
  } -> f64
  return
}

// -----

// Owner registers over a list that is not grouped by owner: the groups.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @owner_over_groups(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes a list that does not meet its requirement}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      directed visit unordered combine owner
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// Owner registers with a visit of each pair once.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @owner_unique(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{adds each owner's row in its registers, which a visit of each pair once does not give}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine owner
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A visit of each pair once over a directed list, which would count every pair twice.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @unique_over_matrix(%nl: !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{the traversal 'unique' is required}}
  %u = md_alg.traverse_pairs structure %nl : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A visit of each pair once over a list whose excluded pairs are only marked.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @masked(%nl: !mdrt.neighbors<@atoms, unique, minimum, masked, stable(group, search)>, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes a list that does not meet its requirement}}
  %u = md_alg.traverse_pairs structure %nl : !mdrt.neighbors<@atoms, unique, minimum, masked, stable(group, search)>
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A fixed_order destination with an unordered visit: the visit is weaker than the algebra needs.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @fixed_unordered(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{visits 'unordered', which value 0, 'fixed_order', does not allow}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["fixed_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A fixed_order destination added by atomics, whose order the threads decide.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @fixed_atomic(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{adds destination 0, 'fixed_order', by the combination 'atomic'}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit stable combine atomic
      algebra ["fixed_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A fixed_order sum over the units of a visit of each pair once.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @fixed_sum(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{sums value 1, 'fixed_order', over the units}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit stable combine atomic
      algebra ["any_order", "fixed_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// An antisymmetric sum, which a visit of each pair once cannot give.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @antisymmetric_sum(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{needs every destination symmetric or antisymmetric and every sum symmetric; value 1 is antisymmetric}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "antisymmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// An algebra for the destination alone.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @algebra_count(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{expected 2 algebra entries}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// The reserved combination.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @privatized(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes the combination 'privatized', which is reserved}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine privatized
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A sorted visit, which the op does not have.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @sorted(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{visits 'unordered' or 'stable', not 'sorted'}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit sorted combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// The buffer form without its contract.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @no_contract(%lst: memref<?xi32>, %np: index, %nu: index, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{states the contract of its buffers}}
  %u = md_alg.traverse_pairs lists(%lst, %lst, %lst, %lst, %lst, %lst, %lst : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%np, %nu) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A cutoff beyond the stated reach of the buffers.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @beyond_reach(%lst: memref<?xi32>, %np: index, %nu: index, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{has a cutoff, 9.000000e-01, beyond the reach of its list, 8.000000e-01}}
  %u = md_alg.traverse_pairs lists(%lst, %lst, %lst, %lst, %lst, %lst, %lst : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%np, %nu) contract(!groups) reach(0.8)
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// The buffer form with the buffers of the outer list and inner.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @buffer_count(%lst: memref<?xi32>, %np: index, %nu: index, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{takes 12 buffers of the layout of the groups in its buffer form, not 7}}
  %u = md_alg.traverse_pairs lists(%lst, %lst, %lst, %lst, %lst, %lst, %lst : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%np, %nu) inner contract(!groups) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A structure and buffers at once.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @both_forms(%lst: memref<?xi32>, %np: index, %nu: index, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %nl: !groups) {
  // expected-error @+1 {{takes its list as a structure or as buffers, not both}}
  %u = md_alg.traverse_pairs structure %nl : !groups lists(%lst, %lst, %lst, %lst, %lst, %lst, %lst : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) sizes(%np, %nu) contract(!groups) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A region that yields no value for the sum.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!q = memref<?xf32, 1>
func.func @yields(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64) {
  // expected-error @+1 {{yields 1 values, not one per destination and sum (2)}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell)  outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    md_alg.yield %d : vector<3xf32>
  } -> f64
  return
}

// -----

// A buffer of the layout in f32, not i32.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @buffer_type(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index, %g: memref<?xf32>) {
  // expected-error @+1 {{takes rank-1 buffers of i32 for the layout of the groups}}
  %u = md_alg.traverse_pairs lists(%l, %l, %l, %l, %l, %l, %g : !i, !i, !i, !i, !i, !i, memref<?xf32>) sizes(%np, %nu) contract(!groups) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// The buffer form with the places alone, without the units.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @sizes(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{takes two sizes, the places and the units, in its buffer form}}
  %u = md_alg.traverse_pairs lists(%l, %l, %l, %l, %l, %l, %l : !i, !i, !i, !i, !i, !i, !i) sizes(%np) contract(!groups) reach(1.0)
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// The buffer form without its reach.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @no_reach(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{states the reach of its buffers in its buffer form}}
  %u = md_alg.traverse_pairs lists(%l, %l, %l, %l, %l, %l, %l : !i, !i, !i, !i, !i, !i, !i) sizes(%np, %nu) contract(!groups)
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// An exchange contract for the destination alone.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @exchange_count(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{expected 2 exchange contracts, one per destination and sum, got 1}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// Two weights for one sum.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @weights_count(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{expected 1 weights, one per sum, got 2}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9) weights [0.5, 0.5] {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// Two overwrite flags for one destination.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @overwrite_count(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{expected 1 overwrite flags, one per destination, got 2}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9) overwrite [true, false] {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A cutoff of zero.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @cutoff(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{expected a positive cutoff}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.0)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// An algebra that D301 does not have.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @unknown_algebra(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{has an unknown algebra 'any'}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// An exchange contract that the md dialect does not have.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @unknown_exchange(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{has an unknown exchange contract 'odd'}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["odd", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// Two destinations that are one buffer: one of the two sums would be lost.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @one_buffer(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{has destinations 0 and 1, which are one buffer}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f, %f : !f, !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order", "any_order"] exchange ["antisymmetric", "antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %d, %e : vector<3xf32>, vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A region with an argument for a field that the op does not take.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @arguments(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{takes a region of 2 arguments: r^2, the displacement, and two values for each buffer of}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>, %q: f32):
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}

// -----

// A region that ends in another terminator than md_alg.yield.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func @terminator(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  // expected-error @+1 {{ends its region in 'md_alg.yield'}}
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    llvm.unreachable
  } -> f64
  return
}

// -----

// A call in the region, whose effects the verifier cannot see: it could write anywhere.
!groups = !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!x = memref<?x3xf64, 1>
!f = memref<?x3xf32, 1>
!i = memref<?xi32>
func.func private @g(f32)
func.func @unseen_effects(%nl: !groups, %x: !x, %cell: !md.cell, %f: !f, %u0: f64, %l: !i, %np: index, %nu: index) {
  %u = md_alg.traverse_pairs structure %nl : !groups
      positions(%x : !x) cell(%cell : !md.cell) outs(%f : !f) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"] cutoff(0.9)  {
  ^bb0(%r2: f32, %d: vector<3xf32>):
    // expected-error @+1 {{has effects that the verifier of 'md_alg.traverse_pairs' cannot see; its region may not write}}
    func.call @g(%r2) : (f32) -> ()
    %e = arith.extf %r2 : f32 to f64
    md_alg.yield %d, %e : vector<3xf32>, f64
  } -> f64
  return
}
