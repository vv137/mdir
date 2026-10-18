// RUN: mdir-opt %s -split-input-file -verify-diagnostics

// The verifiers of the neural stages (D288) and of fields of
// features.

// expected-error@+1 {{expected each operand of f64, a vector of f64, i32, or i64, got 'f32'}}
md.stage @s forward(@f) : (f32) -> f64
func.func @f(%x: f32) -> f64 {
  %c = arith.constant 1.0 : f64
  return %c : f64
}

// -----

// expected-error@+1 {{expected each result of f64 or a vector of f64, got 'i32'}}
md.stage @s forward(@f) : (f64) -> i32
func.func @f(%x: f64) -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

// -----

// expected-error@+1 {{expected at least one result}}
md.stage @s forward(@f) : (f64) -> ()
func.func @f(%x: f64) {
  return
}

// -----

// expected-error@+1 {{'g', its forward body, does not name a function}}
md.stage @s forward(@g) : (f64) -> f64

// -----

// expected-error@+1 {{expected its forward body 'f' of the type '(f64) -> f64', got '(f64) -> vector<2xf64>'}}
md.stage @s forward(@f) : (f64) -> f64
func.func @f(%x: f64) -> vector<2xf64> {
  %v = vector.broadcast %x : f64 to vector<2xf64>
  return %v : vector<2xf64>
}

// -----

// The product takes the operands and a cotangent of each result, and
// returns a cotangent of each differentiable operand: none for the species.
// expected-error@+1 {{expected its vjp body 'b' of the type '(f64, i32, vector<2xf64>) -> f64', got '(f64, i32, vector<2xf64>) -> (f64, i32)'}}
md.stage @s forward(@f) vjp(@b) : (f64, i32) -> vector<2xf64>
func.func @f(%x: f64, %z: i32) -> vector<2xf64> {
  %v = vector.broadcast %x : f64 to vector<2xf64>
  return %v : vector<2xf64>
}
func.func @b(%x: f64, %z: i32, %y: vector<2xf64>) -> (f64, i32) {
  %s = vector.reduction <add>, %y : vector<2xf64> into f64
  return %s, %z : f64, i32
}

// -----

// expected-error@+1 {{cannot bind the declaration 'f' as its forward body: only a body in the module is lowered (an external body, such as an AOTInductor export, is step 4 of #377)}}
md.stage @s forward(@f) : (f64) -> f64
func.func private @f(f64) -> f64

// -----

md.stage @s forward(@f) : (f64) -> f64
memref.global "private" @g : memref<1xf64> = dense<0.0>
func.func @f(%x: f64) -> f64 {
  %m = memref.get_global @g : memref<1xf64>
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{in the forward body of the stage 's', which must be pure: the op has memory effects}}
  memref.store %x, %m[%c0] : memref<1xf64>
  return %x : f64
}

// -----

md.stage @s forward(@f) : (f64) -> f64
func.func @f(%x: f64) -> f64 {
  return %x : f64
}
func.func @outside(%x: f64) -> f64 {
  // expected-error@+1 {{expected to be in a kernel of the md dialect}}
  %y = md.apply_stage @s(%x) : (f64) -> f64
  return %y : f64
}

// -----

!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.function @k(%q: !real) -> !real {
  %h = md.map_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    // expected-error@+1 {{'t' does not name a stage}}
    %y = md.apply_stage @t(%qi) : (f64) -> f64
    md.yield %y : f64
  } : !real
  md.return %h : !real
}

// -----

!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.stage @s forward(@f) : (f64) -> f64
func.func @f(%x: f64) -> f64 {
  return %x : f64
}
md.function @k(%q: !real) -> !real {
  %h = md.map_particles gather(%q : !real) {
  ^bb0(%qi: f64):
    // expected-error@+1 {{the stage 's' has no vector-Jacobian product}}
    %y = md.apply_stage @s vjp(%qi, %qi) : (f64, f64) -> f64
    md.yield %y : f64
  } : !real
  md.return %h : !real
}

// -----

!real = !md.field<@atoms, f64>
md.particle_set @atoms
md.stage @s forward(@f) : (f64) -> f64
func.func @f(%x: f64) -> f64 {
  return %x : f64
}
md.function @k(%q: !md.field<@atoms, 2 x f64>) -> !real {
  %h = md.map_particles gather(%q : !md.field<@atoms, 2 x f64>) {
  ^bb0(%qi: vector<2xf64>):
    // expected-error@+1 {{expected the type '(f64) -> f64' of the stage 's', got '(vector<2xf64>) -> f64'}}
    %y = md.apply_stage @s(%qi) : (vector<2xf64>) -> f64
    md.yield %y : f64
  } : !real
  md.return %h : !real
}

// -----

// expected-error@+1 {{expected at least 1 component}}
func.func @f(%x: !md.field<@atoms, 0 x f64>)

// -----

// expected-error@+1 {{a field with several components must have element type f64 or f32, got 'i32'}}
func.func @f(%x: !md.field<@atoms, 4 x i32>)

// -----

// expected-error@+1 {{expected a block of the form '<M>x<l><e|o>', such as '4x1o'}}
func.func @f(%x: !md.field<@atoms, irreps<4x1y>, f64>)

// -----

// expected-error@+1 {{expected the layout 'mul_ir' or 'ir_mul', got 'mul'}}
func.func @f(%x: !md.field<@atoms, irreps<4x0e, mul>, f64>)

// -----

// expected-error@+1 {{expected a degree l from 0 to 8, got 9}}
func.func @f(%x: !md.field<@atoms, irreps<1x9e>, f64>)


// -----

// expected-error@+1 {{a field with several components must have element type f64 or f32, got 'i32'}}
func.func @f(%x: !md.field<@atoms, irreps<2x1o>, i32>)

// -----

!vec = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  // expected-error@+1 {{a gather of several fields has no exchange contract: expected 'exchange(none)'}}
  %a, %b = md.gather_relation %n, %x, %cell exchange(antisymmetric) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    md.yield %d, %d : vector<3xf64>, vector<3xf64>
  } : !pairs, !vec -> !vec, !vec
  md.return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.function @f(%x: !vec, %cell: !md.cell) {
  %n = md.neighborhood %x, %cell cutoff(0.5) : !vec -> !pairs
  %a, %b = md.gather_relation %n, %x, %cell exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>):
    // expected-error@+1 {{expected exactly 2 values, got 1}}
    md.yield %d : vector<3xf64>
  } : !pairs, !vec -> !vec, !vec
  md.return
}
