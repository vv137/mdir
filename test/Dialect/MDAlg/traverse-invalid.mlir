// The refusals of the verifier of md_alg.traverse_pairs (D[md-alg-traverse]),
// one case each, each a realistic bug: a write outside the declared
// combination, a read of a destination, owner registers over a list that is
// not grouped by owner, a visit weaker than the algebra needs or than the list
// gives, and the forms of the list.
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
