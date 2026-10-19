// The sums across domains state their algebra too (D301,
// ruled on #470): `fixed_order` in every mode, the domains added in their
// order, which gives the same bits on every run for the same decomposition
// and build, not for another decomposition (#409). md-exec-assign-algebra
// states it in either mode, as it states every algebra; the check refuses
// a sum without one.
//
// RUN: mdir-opt %s --md-exec-check-algebra -verify-diagnostics -o /dev/null
// RUN: mdir-opt %s --md-exec-assign-algebra | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra=deterministic=true | FileCheck %s
// RUN: mdir-opt %s --md-exec-assign-algebra --md-exec-check-algebra -o /dev/null
//
// CHECK-LABEL: func.func @sum(
// CHECK: md_dist.all_reduce %{{.*}} {md_exec.algebra = #md_exec.algebra<fixed_order>} : !md_dist.partial<f64> -> f64
// CHECK: md_dist.all_reduce %{{.*}} {md_exec.algebra = #md_exec.algebra<fixed_order>} : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
func.func @sum(%p: !md_dist.partial<f64>, %w: !md_dist.partial<vector<9xf64>>)
    -> (f64, vector<9xf64>) {
  // expected-error@+1 {{'md_dist.all_reduce' op states no algebra of the values it reduces; run 'md-exec-assign-algebra'}}
  %u = md_dist.all_reduce %p : !md_dist.partial<f64> -> f64
  %v = md_dist.all_reduce %w {md_exec.algebra = #md_exec.algebra<fixed_order>}
      : !md_dist.partial<vector<9xf64>> -> vector<9xf64>
  return %u, %v : f64, vector<9xf64>
}
