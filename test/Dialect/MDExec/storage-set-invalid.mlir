// RUN: mdir-opt %s --convert-md-exec-to-loops -split-input-file -verify-diagnostics

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_loop(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>,
    %b: memref<?xf64, #md_exec.set_layout<@other, 1>>) {
  // expected-error@+1 {{expected a buffer of set @atoms, got a buffer of set @other}}
  md_exec.particle_for ins(%a, %b : memref<?xf64, #md_exec.set_layout<@atoms, 1>>, memref<?xf64, #md_exec.set_layout<@other, 1>>) outs(%a : memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  ^bb0(%x: f64, %y: f64):
    md_exec.yield %y : f64
  }
  return
}

// -----

md.particle_set @atoms
md.particle_set @other

func.func @cross_set_reuse(
    %a: memref<?xf64, #md_exec.set_layout<@atoms, 1>>) {
  // expected-error@+1 {{cannot reinterpret storage of set @atoms as set @other}}
  %b = memref.cast %a : memref<?xf64, #md_exec.set_layout<@atoms, 1>> to memref<?xf64, #md_exec.set_layout<@other, 1>>
  return
}
