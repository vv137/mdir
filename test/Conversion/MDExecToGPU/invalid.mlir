// RUN: mdir-opt %s --convert-md-exec-to-gpu -split-input-file \
// RUN:     -verify-diagnostics

md.particle_set @atoms

func.func @f(%v: memref<?x3xf64>) {
  // expected-error@+1 {{has its buffers on the host; use 'convert-md-exec-to-loops', or run 'md-exec-assign-storage' with 'memory=device'}}
  md_exec.particle_for ins(%v : memref<?x3xf64>)
      outs(%v : memref<?x3xf64>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  }
  return
}

// -----

md.particle_set @atoms

func.func @f(%v: memref<?x3xf64, 1>) -> f64 {
  %zero = arith.constant 0.0 : f64
  // expected-error@+1 {{needs 2 buffers in 'scratch' for each global sum; run 'md-exec-assign-storage' with 'memory=device'}}
  %s = md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
      reduce(%zero : f64) {
  ^bb0(%v_i: vector<3xf64>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    md_exec.yield %v2 : f64
  } -> f64
  return %s : f64
}

// -----

md.particle_set @atoms

func.func @f(%v: !md.field<@atoms, 3 x f64>) {
  // expected-error@-1 {{in its signature, which is not in the storage form; run 'md-exec-assign-storage' first}}
  return
}
