// The declared transpose of md_exec.scatter_add (D291, D295):
// a' = a + Σ_{k : idx_k = i} v_k is linear in the values and in the
// destination. The transpose in the destination is the identity; that in
// the values, a gather by the indices, waits for an op of md_exec (the
// ruling of 2026-10-19 on #459), and the op says so.
//
// RUN: mdir-opt %s --split-input-file --test-md-transpose -verify-diagnostics -o /dev/null

md.particle_set @atoms
md.particle_set @ghosts
func.func @destination(%v: !md.field<@ghosts, 3 x f64>,
                       %i: !md.field<@ghosts, i32>,
                       %a: !md.field<@atoms, 3 x f64>) {
  // expected-remark@+1 {{transpose in operand 2: the adjoint of the result}}
  %a1 = md_exec.scatter_add %v at %i into %a {test.transpose = 2}
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @values(%v: !md.field<@ghosts, 3 x f64>,
                  %i: !md.field<@ghosts, i32>,
                  %a: !md.field<@atoms, 3 x f64>) {
  // expected-error@+1 {{has no transpose in its values yet: it is a gather by the indices, which md_exec has no op for}}
  %a1 = md_exec.scatter_add %v at %i into %a {test.transpose = 0}
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}

// -----

md.particle_set @atoms
md.particle_set @ghosts
func.func @indices(%v: !md.field<@ghosts, 3 x f64>,
                   %i: !md.field<@ghosts, i32>,
                   %a: !md.field<@atoms, 3 x f64>) {
  // expected-remark@+1 {{not linear in operand 1}}
  %a1 = md_exec.scatter_add %v at %i into %a {test.transpose = 1}
      : !md.field<@ghosts, 3 x f64>, !md.field<@ghosts, i32>
        -> !md.field<@atoms, 3 x f64>
  return
}
