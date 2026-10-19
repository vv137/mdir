// Batched bodies of neural stages (D[ml4-batched-bodies], docs/ml4-ir.md,
// Sections 2.1 and 8.3): a body of linalg on tensors with a static number
// of rows, which may compute in f32 where the declaration says f64; the
// verifier's checks.
//
// RUN: mdir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: md.stage @scale forward(@scale.f) vjp(@scale.b)
md.particle_set @atoms
md.stage @scale forward(@scale.f) vjp(@scale.b)
    : (vector<2xf64>) -> vector<2xf64>
func.func @scale.f(%h: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %e = tensor.empty() : tensor<4x2xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%h : tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) {
  ^bb0(%a: f32, %o: f32):
    %two = arith.constant 2.0 : f32
    %b = arith.mulf %a, %two : f32
    linalg.yield %b : f32
  } -> tensor<4x2xf32>
  return %y : tensor<4x2xf32>
}
func.func @scale.b(%h: tensor<4x2xf32>, %c: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %e = tensor.empty() : tensor<4x2xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%c : tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) {
  ^bb0(%a: f32, %o: f32):
    %two = arith.constant 2.0 : f32
    %b = arith.mulf %a, %two : f32
    linalg.yield %b : f32
  } -> tensor<4x2xf32>
  return %y : tensor<4x2xf32>
}
md.potential @model(%x: !md.field<@atoms, 3 x f64>, %cell: !md.cell,
                    %h: !md.field<@atoms, 2 x f64>) -> f64 {
  %g = md.map_particles gather(%h : !md.field<@atoms, 2 x f64>) {
  ^bb0(%hi: vector<2xf64>):
    %y = md.apply_stage @scale(%hi) : (vector<2xf64>) -> vector<2xf64>
    md.yield %y : vector<2xf64>
  } : !md.field<@atoms, 2 x f64>
  %zero = arith.constant 0.0 : f64
  md.return %zero : f64
}

// -----

// A tile calls both bodies on the same rows.
// expected-error@+1 {{expected its forward body and its vector-Jacobian product of the same form and the same number of rows}}
md.stage @s forward(@s.f) vjp(@s.b) : (f64) -> f64
func.func @s.f(%x: tensor<4xf32>) -> tensor<4xf32> {
  return %x : tensor<4xf32>
}
func.func @s.b(%x: tensor<8xf32>, %c: tensor<8xf32>) -> tensor<8xf32> {
  return %c : tensor<8xf32>
}

// -----

// The rows are static.
// expected-error@+1 {{or its batched form, a tensor of a static number of rows for each value}}
md.stage @s forward(@s.f) : (f64) -> f64
func.func @s.f(%x: tensor<?xf32>) -> tensor<?xf32> {
  return %x : tensor<?xf32>
}

// -----

// A vector of the declaration is a row of its components.
// expected-error@+1 {{or its batched form, a tensor of a static number of rows for each value}}
md.stage @s forward(@s.f) : (vector<3xf64>) -> f64
func.func @s.f(%x: tensor<4x2xf32>) -> tensor<4xf32> {
  %c = arith.constant dense<0.0> : tensor<4xf32>
  return %c : tensor<4xf32>
}

// -----

// The rows of every value agree.
// expected-error@+1 {{or its batched form, a tensor of a static number of rows for each value}}
md.stage @s forward(@s.f) : (f64) -> f64
func.func @s.f(%x: tensor<4xf32>) -> tensor<8xf32> {
  %c = arith.constant dense<0.0> : tensor<8xf32>
  return %c : tensor<8xf32>
}
