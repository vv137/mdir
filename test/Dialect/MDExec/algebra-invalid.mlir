// The verifier of `md_exec.algebra` (D301): one entry a
// reduced value, `any_order_exact` only on an integer, and only on an op that
// reduces values.
//
// RUN: mdir-opt %s -split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @count(%v: !vec) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error@+1 {{states 2 algebras for the 1 values it reduces}}
  %k = md_exec.particle_for ins(%v : !vec) reduce(%zero : f64) {
  ^bb0(%v_i: vector<3xf32>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf32>
    %v2 = vector.reduction <add>, %sq : vector<3xf32> into f32
    %wide = arith.extf %v2 : f32 to f64
    md_exec.yield %wide : f64
  } {md_exec.algebra = #md_exec.algebra<any_order, any_order_exact>} -> f64
  return %k : f64
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

// A floating-point sum rounds by its order: it is not exact.
func.func @exact(%v: !vec) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error@+1 {{states the algebra 'any_order_exact' for reduced value 0 of 'f64', whose sum rounds by the order of its terms}}
  %k = md_exec.particle_for ins(%v : !vec) reduce(%zero : f64) {
  ^bb0(%v_i: vector<3xf32>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf32>
    %v2 = vector.reduction <add>, %sq : vector<3xf32> into f32
    %wide = arith.extf %v2 : f32 to f64
    md_exec.yield %wide : f64
  } {md_exec.algebra = #md_exec.algebra<any_order_exact>} -> f64
  return %k : f64
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @nothing() -> !vec {
  // expected-error@+1 {{states an algebra but reduces no values}}
  %y = md_exec.zeros {md_exec.algebra = #md_exec.algebra<any_order_exact>} : !vec
  return %y : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f32>
md.particle_set @atoms

func.func @keyword() -> !vec {
  // expected-error@+1 {{expected 'any_order_exact', 'any_order', or 'fixed_order', got 'associative'}}
  %y = md_exec.zeros {md_exec.algebra = #md_exec.algebra<associative>} : !vec
  return %y : !vec
}

// -----

!vec = !md.field<@atoms, 3 x f32>

module attributes {md_exec.deterministic} {
  md.particle_set @atoms

  func.func @deterministic(%v: !vec) -> f64 {
    %zero = arith.constant 0.0 : f64
    // expected-error@+1 {{states the algebra 'any_order' for reduced value 0 of 'f64' in a program built for the deterministic mode, whose floating-point sums are 'fixed_order'}}
    %k = md_exec.particle_for ins(%v : !vec) reduce(%zero : f64) {
    ^bb0(%v_i: vector<3xf32>):
      %sq = arith.mulf %v_i, %v_i : vector<3xf32>
      %v2 = vector.reduction <add>, %sq : vector<3xf32> into f32
      %wide = arith.extf %v2 : f32 to f64
      md_exec.yield %wide : f64
    } {md_exec.algebra = #md_exec.algebra<any_order>} -> f64
    return %k : f64
  }
}
