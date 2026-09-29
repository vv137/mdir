// RUN: mdir-opt %s -split-input-file -verify-diagnostics \
// RUN:     --md-exec-assign-precision="mode=mixed"

md.particle_set @atoms

// Two buffers that disagree hold one field.
func.func @two_buffers(%in: memref<?x3xf32>) -> memref<?x3xf64> {
  // expected-note@+1 {{see here}}
  %x = mdrt.from_buffer %in
      : memref<?x3xf32> to !md.field<@atoms, 3 x f64>
  // expected-error@+1 {{states that a field is stored as 'f64', but the field is stored as 'f32' elsewhere}}
  %out = mdrt.to_buffer %x
      : !md.field<@atoms, 3 x f64> to memref<?x3xf64>
  return %out : memref<?x3xf64>
}

// -----

md.particle_set @atoms

func.func private @other(!md.field<@atoms, 3 x f64>)

func.func @unknown_op(%x: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{cannot be assigned a precision: the op is not known to the pass and uses fields}}
  call @other(%x) : (!md.field<@atoms, 3 x f64>) -> ()
  return
}

// -----

md.particle_set @atoms

// Nothing tells whether the field holds positions or forces, and the mode
// stores the two in different types.
// expected-error@+1 {{cannot tell the type that a field is stored in: no buffer holds the field, no loop writes it, and it is not used as positions}}
func.func @unused(%a: !md.field<@atoms, 3 x f64>)
    -> !md.field<@atoms, 3 x f64> {
  return %a : !md.field<@atoms, 3 x f64>
}

// -----

md.particle_set @atoms

func.func @storage(%v: memref<?x3xf64>) {
  // expected-error@+1 {{is in the storage form; precision is assigned before storage}}
  md_exec.particle_for ins(%v : memref<?x3xf64>)
      outs(%v : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  }
  return
}
