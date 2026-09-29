// RUN: mdir-opt %s --md-exec-assign-storage="memory=device" \
// RUN:     -split-input-file -verify-diagnostics

// RUN: not mdir-opt %s --md-exec-assign-storage="memory=disk" 2>&1 \
// RUN: | FileCheck %s --check-prefix=MEMORY
// MEMORY: expected the memory 'host' or 'device', got 'disk'

md.particle_set @atoms

// A global sum of vectors has no buffer that could hold its contributions.
func.func @f(%v: !md.field<@atoms, 3 x f64>) -> vector<3xf64> {
  %zero = arith.constant dense<0.0> : vector<3xf64>
  // expected-error@+1 {{has a global sum of the type 'vector<3xf64>'; on a device only sums of single numbers are supported}}
  %p = md_exec.particle_for ins(%v : !md.field<@atoms, 3 x f64>)
      reduce(%zero : vector<3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  } -> vector<3xf64>
  return %p : vector<3xf64>
}
