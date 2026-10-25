// RUN: mdir-opt %s --test-md-exec-loop-effects -o /dev/null | FileCheck %s

!real = !md.field<@atoms, f64>
md.particle_set @atoms
func.func private @opaque(f64) -> f64

func.func @unknown_body(%input: !real) -> !real {
  %empty = md_exec.empty : !real
  %out = md_exec.particle_for ins(%input : !real)
      outs(%empty : !real) {
  ^bb0(%x: f64):
    %value = func.call @opaque(%x) : (f64) -> f64
    md_exec.yield %value : f64
  } -> !real
  return %out : !real
}

// CHECK: md_exec.particle_for complete=0 accesses=2 reductions=0
// CHECK: operand=0 effect=read rows=own set=@atoms
// CHECK: operand=1 effect=write rows=own set=@atoms
