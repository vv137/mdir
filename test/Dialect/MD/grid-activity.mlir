// RUN: mdir-opt %s --md-analyze-activity=argument=2 --verify-diagnostics -o /dev/null
// RUN: mdir-opt %s --md-analyze-activity=argument=3 --verify-diagnostics -o /dev/null
// RUN: mdir-opt %s --md-check-derivative-coverage -o /dev/null
// Both the tensor data and the explicit uniform scale remain active.
md.particle_set @atoms
md.potential @grid(%positions: !md.field<@atoms, 3 x f64>, %cell: !md.cell, %input: tensor<2x3x4xf64>, %scale: f64) -> f64 {
  // expected-remark @+1 {{activity: active}}
  %mapped = md.map_grid (%input, %scale) grid([2, 3, 4]) pairing("euclidean") : (tensor<2x3x4xf64>, f64) -> tensor<2x3x4xf64> {
  ^bb0(%index: vector<3xindex>, %value: f64, %factor: f64):
    %y = arith.mulf %value, %factor : f64
    md.yield %y : f64
  }
  // expected-remark @+1 {{activity: active}}
  %sum = md.sum_grid (%mapped) grid([2, 3, 4]) pairing("euclidean") : (tensor<2x3x4xf64>) -> f64 {
  ^bb0(%index: vector<3xindex>, %value: f64):
    md.yield %value : f64
  }
  // expected-remark @+1 {{activity: proven inactive}}
  %zero = arith.constant 0.0 : f64
  md.return %sum : f64
}
