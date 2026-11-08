"""Exact in-place updates and callback frames for renumbered OHHM units.

All inputs are dyadic. Python evaluates the declared scalar update for
each original particle independently; no scheduler or packed-row formula
is reused to obtain the updated coordinates or velocities.
"""
import json
import pathlib
import sys

waters, atoms = 19, 17
n = 4 * waters + atoms
permutation = [(37 * i + 11) % n for i in range(n)]
water = [permutation[3 * i:3 * i + 3] for i in range(waters)]
extra = permutation[3 * waters:4 * waters]
solute = permutation[4 * waters:]
members = [p for group in water for p in group] + solute
slots = [-1] * n
for row, p in enumerate(members):
    slots[p] = row
x = [[(p - 45) / 16, ((p * 7) % 23 - 11) / 8, (p % 13 - 6) / 4] for p in range(n)]
v = [[((p * 3 + k) % 17 - 8) / 32 for k in range(3)] for p in range(n)]
delta = [[((p * 5 + k * 7) % 19 - 9) / 64 for k in range(3)] for p in range(n)]
updated = [[a + d for a, d in zip(row, change)] for row, change in zip(x, delta)]
velocity = [[a + .5 * b for a, b in zip(row, position)] for row, position in zip(v, updated)]
reduced = [[position[0] + speed[2], float(position[0] > 0)]
           for position, speed in zip(updated, velocity)]
audit = [updated[p] for p in members]
expected = [a for table in (updated, velocity, audit, reduced) for row in table for a in row] + [1] * n

if sys.argv[1] == 'check':
    got = [float(word) for word in pathlib.Path(sys.argv[2]).read_text().split()]
    assert len(got) == len(expected), (len(got), len(expected))
    errors = [(i, a, b) for i, (a, b) in enumerate(zip(got, expected)) if a != b]
    assert not errors, {'count': len(errors), 'first': errors[:8]}
    print(f'complete-frame writer: {len(expected)} exact values; every particle visited once; extra points updated but not packed')
    raise SystemExit(0)

lines = ['md.particle_set @atoms']


def array(name, shape, data):
    lines.append(f'memref.global "private" constant @{name} : memref<{shape}> = dense<{json.dumps(data)}>')


for name, shape, data in [('x', f'{n}x3xf64', x), ('v', f'{n}x3xf64', v),
                          ('delta', f'{n}x3xf32', delta), ('slots', f'{n}xi32', slots),
                          ('members', f'{len(members)}xi32', members), ('zero', f'{n}xi32', [0] * n)]:
    array(name, shape, data)
lines.append(r'''
func.func @writer_source(%x: memref<?x3xf64, 1>, %v: memref<?x3xf64, 1>,
    %delta: memref<?x3xf32, 1>, %slots: memref<?xi32, 1>,
    %members: memref<?xi32, 1>, %waters: index, %audit: memref<?x3xf64, 1>,
    %reductions: memref<?x2xf64, 1>, %visits: memref<?xi32, 1>) {
  %zero = arith.constant 0.0 : f64
  %false = arith.constant false
  %sums:2 = md_exec.particle_for
      ins(%x, %v, %delta : memref<?x3xf64, 1>, memref<?x3xf64, 1>, memref<?x3xf32, 1>)
      outs(%x, %v : memref<?x3xf64, 1>, memref<?x3xf64, 1>)
      reduce(%zero, %false : f64, i1) {
  ^bb0(%old: vector<3xf64>, %speed: vector<3xf64>, %d: vector<3xf32>):
    %wide = arith.extf %d : vector<3xf32> to vector<3xf64>
    %next = arith.addf %old, %wide : vector<3xf64>
    %half = arith.constant dense<0.5> : vector<3xf64>
    %change = arith.mulf %half, %next : vector<3xf64>
    %new_speed = arith.addf %speed, %change : vector<3xf64>
    %a = vector.extract %next[0] : f64 from vector<3xf64>
    %b = vector.extract %new_speed[2] : f64 from vector<3xf64>
    %sum = arith.addf %a, %b : f64
    %positive = arith.cmpf ogt, %a, %zero : f64
    md_exec.yield %next, %new_speed, %sum, %positive : vector<3xf64>, vector<3xf64>, f64, i1
  } -> f64, i1
  return
}
func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func @dump(%device: memref<?x?xf64, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %device, %c0 : memref<?x?xf64, 1>
  %m = memref.dim %device, %c1 : memref<?x?xf64, 1>
  %host = memref.alloc(%n, %m) : memref<?x?xf64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?x?xf64>, memref<?x?xf64, 1>
  gpu.wait [%done]
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %m step %c1 {
      %value = memref.load %host[%i, %j] : memref<?x?xf64>
      func.call @printF64(%value) : (f64) -> ()
      func.call @printNewline() : () -> ()
    }
  }
  memref.dealloc %host : memref<?x?xf64>
  return
}
func.func @main() {
''')
serial = 0


def emit(text):
    global serial
    name = f'%a{serial}'
    serial += 1
    lines.append(f'  {name} = {text}')
    return name


def allocate(shape, name=None):
    device = emit(f'gpu.alloc() : memref<{shape}, 1>')
    if name:
        host = emit(f'memref.get_global @{name} : memref<{shape}>')
        ready = emit('gpu.wait async')
        done = emit(f'gpu.memcpy async [{ready}] {device}, {host} : memref<{shape}, 1>, memref<{shape}>')
        lines.append(f'  gpu.wait [{done}]')
    dynamic = '?x' + 'x'.join(shape.split('x')[1:])
    view = emit(f'memref.cast {device} : memref<{shape}, 1> to memref<{dynamic}, 1>')
    return device, view


buffers = [allocate(shape, name) for shape, name in
           [(f'{n}x3xf64', 'x'), (f'{n}x3xf64', 'v'), (f'{n}x3xf32', 'delta'),
            (f'{n}xi32', 'slots'), (f'{len(members)}xi32', 'members'),
            (f'{len(members)}x3xf64', None), (f'{n}x2xf64', None), (f'{n}xi32', 'zero')]]
w = emit(f'arith.constant {waters} : index')
args = [p[1] for p in buffers]
args.insert(5, w)
types = ['memref<?x3xf64, 1>'] * 2 + ['memref<?x3xf32, 1>'] + ['memref<?xi32, 1>'] * 2
types += ['index', 'memref<?x3xf64, 1>', 'memref<?x2xf64, 1>', 'memref<?xi32, 1>']
lines.append(f'  func.call @writer_source({", ".join(args)}) : ({", ".join(types)}) -> ()')
for i, cols in [(0, 3), (1, 3), (5, 3), (6, 2)]:
    view = emit(f'memref.cast {buffers[i][1]} : memref<?x{cols}xf64, 1> to memref<?x?xf64, 1>')
    lines.append(f'  func.call @dump({view}) : (memref<?x?xf64, 1>) -> ()')
host = emit(f'memref.alloc() : memref<{n}xi32>')
ready = emit('gpu.wait async')
done = emit(f'gpu.memcpy async [{ready}] {host}, {buffers[7][0]} : memref<{n}xi32>, memref<{n}xi32, 1>')
lines.append(f'  gpu.wait [{done}]')
for p in range(n):
    row = emit(f'arith.constant {p} : index')
    value = emit(f'memref.load {host}[{row}] : memref<{n}xi32>')
    wide = emit(f'arith.extsi {value} : i32 to i64')
    lines.extend([f'  func.call @printI64({wide}) : (i64) -> ()', '  func.call @printNewline() : () -> ()'])
lines.append(f'  memref.dealloc {host} : memref<{n}xi32>')
for (buffer, _), shape in zip(buffers, [f'{n}x3xf64', f'{n}x3xf64', f'{n}x3xf32', f'{n}xi32',
                                      f'{len(members)}xi32', f'{len(members)}x3xf64', f'{n}x2xf64', f'{n}xi32']):
    generic = emit(f'memref.memory_space_cast {buffer} : memref<{shape}, 1> to memref<{shape}>')
    lines.append(f'  gpu.dealloc {generic} : memref<{shape}>')
lines.extend(['  return', '}'])
pathlib.Path(sys.argv[2]).write_text('\n'.join(lines) + '\n')
