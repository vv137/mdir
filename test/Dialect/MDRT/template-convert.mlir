// `mdrt.template_convert`, the conversion that a template of the lowering
// leaves to its instance (D[template-ir-pme-convert], #739): it parses and
// prints for floating point and vectors of it of the same shape, and is
// refused otherwise.
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @convert(
// CHECK: mdrt.template_convert %{{.*}} : f32 to f64
// CHECK: mdrt.template_convert %{{.*}} : f64 to f64
// CHECK: mdrt.template_convert %{{.*}} : vector<9xf32> to vector<9xf64>
func.func @convert(%a: f32, %b: f64, %v: vector<9xf32>) -> (f64, f64, vector<9xf64>) {
  %x = mdrt.template_convert %a : f32 to f64
  %y = mdrt.template_convert %b : f64 to f64
  %w = mdrt.template_convert %v : vector<9xf32> to vector<9xf64>
  return %x, %y, %w : f64, f64, vector<9xf64>
}

// -----

func.func @shape(%v: vector<9xf32>) -> vector<3xf64> {
  // expected-error @+1 {{converts 'vector<9xf32>' to 'vector<3xf64>', which are not of the same shape}}
  %w = mdrt.template_convert %v : vector<9xf32> to vector<3xf64>
  return %w : vector<3xf64>
}

// -----

func.func @scalar_to_vector(%a: f32) -> vector<9xf64> {
  // expected-error @+1 {{converts 'f32' to 'vector<9xf64>', which are not of the same shape}}
  %w = mdrt.template_convert %a : f32 to vector<9xf64>
  return %w : vector<9xf64>
}

// -----

func.func @integer(%a: i32) -> f64 {
  // expected-error @+1 {{operand #0 must be}}
  %x = mdrt.template_convert %a : i32 to f64
  return %x : f64
}
