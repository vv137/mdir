"""Emit the serial GROUPS energy for a retained image and after rewrapping.

A spring is taken from its interacting endpoint to beyond the cutoff and
back along the same path. Each leg's work is its potential drop. Turning
off cell rounding in this fixture reproduces the difference of periods.
"""
import sys

kind, policy = sys.argv[1:]
tilted = kind == "triclinic"
width, bias = (5, 16) if tilted else (4, 4)
h = f"vector<{6 if tilted else 3}xf64>"
values = "[8.00000042, 8.0, 8.0, 0.5, 0.25, -0.5]" if tilted else "[8.00000042, 8.0, 8.0]"
identity = 1 | (bias << 16) | (bias << (16+width)) | (bias << (16+2*width))
print("!i = memref<?xi32>")
for name, size, value in (("order", 16, [0, 1] + [-1]*14),
                          ("shifts", 16, 537395712), ("entry", 64, 1),
                          ("mask", 64, 0), ("one", 1, 1), ("zero", 1, 0)):
    print(f'memref.global "private" @{name} : memref<{size}xi32> = dense<{value}>')
print('func.func private @printF64(f64)\nfunc.func private @printNewline()')
print(f'func.func @energy(%x: f64, %maskval: i32, %cell: {h}) -> f64 {{')
for name, size in (("order",16),("shifts",16),("entry",64),("mask",64),("one",1),("zero",1)):
    print(f'  %{name}_s = memref.get_global @{name} : memref<{size}xi32>')
    print(f'  %{name} = memref.cast %{name}_s : memref<{size}xi32> to !i')
print(f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %places = arith.constant 16 : index
  %zero64 = arith.constant 0.0 : f64
  %first = arith.constant 0.125 : f64
  %p = memref.alloca(%c2) : memref<?x3xf64>
  memref.store %maskval, %mask[%c0] : !i
  memref.store %first, %p[%c0,%c0] : memref<?x3xf64>
  memref.store %x, %p[%c1,%c0] : memref<?x3xf64>
  memref.store %zero64, %p[%c0,%c1] : memref<?x3xf64>
  memref.store %zero64, %p[%c0,%c2] : memref<?x3xf64>
  memref.store %zero64, %p[%c1,%c1] : memref<?x3xf64>
  memref.store %zero64, %p[%c1,%c2] : memref<?x3xf64>
  %u = md_alg.traverse_pairs
    lists(%entry,%mask,%one,%zero,%zero,%order,%shifts : !i,!i,!i,!i,!i,!i,!i)
    sizes(%places,%c1)
    contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
    reach(1.5) positions(%p : memref<?x3xf64>) cell(%cell : {h}) reduce(%zero64 : f64)
    unique visit unordered combine atomic algebra ["any_order"] exchange ["symmetric"]
    cutoff(1.0) weights [0.5] {{
    ^bb0(%r2: f32, %d: vector<3xf32>):
      %w = arith.extf %r2 : f32 to f64
      md_alg.yield %w : f64
    }} -> f64
  return %u : f64
}}
func.func @main() {{
  %input = arith.constant dense<{values}> : {h}
""")
small = h.replace('f64', 'f32')
if policy == "rounded":
    print(f'  %small = arith.truncf %input : {h} to {small}')
    print(f'  %cell = arith.extf %small : {small} to {h}')
cell = '%cell' if policy == 'rounded' else '%input'
print(f"""
  %period = vector.extract {cell}[0] : f64 from {h}
  %p = arith.constant 7.875 : f64
  %wrapped = arith.subf %p, %period : f64
  %far = arith.constant 2.0 : f64
  %retain = arith.constant {identity - (1<<16)} : i32
  %fresh = arith.constant {identity} : i32
  %a = func.call @energy(%p,%retain,{cell}) : (f64,i32,{h}) -> f64
  %b = func.call @energy(%far,%fresh,{cell}) : (f64,i32,{h}) -> f64
  %c = func.call @energy(%wrapped,%fresh,{cell}) : (f64,i32,{h}) -> f64
  %out = arith.subf %a, %b : f64
  %back = arith.subf %b, %c : f64
  %work = arith.addf %out, %back : f64
  func.call @printF64(%a) : (f64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF64(%work) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}}
""")
