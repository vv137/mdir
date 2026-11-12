"""A zero-tolerance dyadic oracle for the actual one-warp scheduler.

Thirteen real first waters occupy a padded cluster of sixteen. Thirty-five
partner entries exercise a full batch and a partial batch. The mask excludes
(i+j)%3 == 0; every fifth entry carries a nonzero image outside the cutoff.
U=|d|^2/2, F=-grad(U), W=-d outer d, with alpha=.75 sites formed from world
positions. All inputs/products/force sums are exactly representable in f32;
energy/virial sums are exactly representable in f64. The first-run bound is
therefore exact numerical equality, including untouched padded destinations.
This tests scheduling and the derived body, not the physical PME route.
"""
import argparse
import json
import math
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('source', type=pathlib.Path)
parser.add_argument('destination', type=pathlib.Path)
parser.add_argument('--summary', action='store_true')
parser.add_argument('--rotated', action='store_true')
parser.add_argument('--screened', action='store_true',
                    help='Check erfc(3.5*r)/r against an independent f64 oracle')
args = parser.parse_args()

forces = [[0.0] * 3 for _ in range(153)]
totals = [0.0] * 10
weights = (-0.5, 0.75, 0.75)


def site(anchor, unit):
    a = 1 / 16
    offsets = (((a, 0, 0), (0, a, 0)), ((0, a, 0), (-a, 0, 0)),
               ((-a, 0, 0), (0, -a, 0)), ((0, -a, 0), (a, 0, 0)))
    first, second = offsets[unit % 4 if args.rotated else 0]
    h1 = tuple(anchor[k] + first[k] for k in range(3))
    h2 = tuple(anchor[k] + second[k] for k in range(3))
    return tuple(anchor[k] + 0.75 * ((h1[k] - anchor[k]) + (h2[k] - anchor[k]))
                 for k in range(3))


pairs = 0
for i in range(13):
    a = site((i / 64, 0.0, 0.0), i)
    for j in range(35):
        if (i + j) % 3 == 0:
            continue
        b = site((0.5 + j / 128, 0.0, 0.0), j)
        image = (2.0 if j % 5 == 0 else 0.0, 0.0, 0.0)
        d = tuple(a[k] - (b[k] + image[k]) for k in range(3))
        r2 = sum(x * x for x in d)
        if r2 >= 1.0:
            continue
        pairs += 1
        coefficient = -1.0
        if args.screened:
            radius = math.sqrt(r2)
            beta = 3.5
            totals[0] += math.erfc(beta * radius) / radius
            coefficient = (math.erfc(beta * radius) / radius +
                           2 * beta / math.sqrt(math.pi) * math.exp(-beta * beta * r2)) / r2
        else:
            totals[0] += 0.5 * r2
        for member, weight in enumerate(weights):
            for k in range(3):
                forces[3 * i + member][k] += coefficient * weight * d[k]
                forces[48 + 3 * j + member][k] -= coefficient * weight * d[k]
        for a_axis in range(3):
            for b_axis in range(3):
                totals[1 + 3 * a_axis + b_axis] += coefficient * d[a_axis] * d[b_axis]

if args.summary:
    print(json.dumps({'rotated': args.rotated, 'active_pairs': pairs, 'energy': totals[0],
                      'virial': totals[1:],
                      'max_force_component': max(abs(x) for row in forces for x in row),
                      'checks': 153 * 3 + 10,
                      'tolerance': 1e-5 if args.screened else 0.0}))

source = args.source.read_text()
if args.screened:
    old = ('%half = arith.constant 0.5 : f64\n'
           '    %e = arith.mulf %half, %r2 : f64')
    new = ('%radius = math.sqrt %r2 : f64\n'
           '    %beta = arith.constant 3.5 : f64\n'
           '    %br = arith.mulf %beta, %radius : f64\n'
           '    %screen = math.erfc %br : f64\n'
           '    %e = arith.divf %screen, %radius : f64')
    assert source.count(old) == 1
    source = source.replace(old, new)
lines = [source,
         'func.func private @template_scheduler_probe(memref<153x3xf32>, memref<32x10xf64>)',
         'func.func private @printI64(i64)',
         'func.func private @printNewline()', 'func.func @main() {']
counter = 0


def emit(expression):
    global counter
    name = f'%v{counter}'
    counter += 1
    lines.append(f'  {name} = {expression}')
    return name


indices = {}


def index(value):
    if value not in indices:
        indices[value] = emit(f'arith.constant {value} : index')
    return indices[value]


zero, one = index(0), index(1)
fzero = emit('arith.constant 0.0 : f32')
widezero = emit('arith.constant 0.0 : f64')
force = emit('memref.alloc() : memref<153x3xf32>')
sums = emit('memref.alloc() : memref<32x10xf64>')
rows, axes = index(153), index(3)
lines.extend([f'  scf.for %row = {zero} to {rows} step {one} {{',
              f'    scf.for %axis = {zero} to {axes} step {one} {{',
              f'      memref.store {fzero}, {force}[%row, %axis] : memref<153x3xf32>',
              '    }', '  }',
              f'  func.call @template_scheduler_probe({force}, {sums}) : '
              '(memref<153x3xf32>, memref<32x10xf64>) -> ()'])
good = emit('arith.constant true')


def check(actual, expected, real, number):
    global good
    ref = emit(f'arith.constant {expected:.17e} : {real}')
    if args.screened:
        # Use the existing physical gate's 1e-5 scale floor, without its
        # additional mixed-baseline/cancellation allowances. Padding is exact.
        scale = (max(abs(x) for row in forces for x in row) if number < 459
                 else abs(totals[0]) if number == 459
                 else max(abs(x) for x in totals[1:]))
        bound = 0.0 if 39 * 3 <= number < 48 * 3 else 1e-5 * scale
        delta = emit(f'arith.subf {actual}, {ref} : {real}')
        absolute = emit(f'math.absf {delta} : {real}')
        limit = emit(f'arith.constant {bound:.17e} : {real}')
        okay = emit(f'arith.cmpf ole, {absolute}, {limit} : {real}')
        bad = emit(f'arith.cmpf ugt, {absolute}, {limit} : {real}')
    else:
        okay = emit(f'arith.cmpf oeq, {actual}, {ref} : {real}')
        bad = emit(f'arith.cmpf une, {actual}, {ref} : {real}')
    good = emit(f'arith.andi {good}, {okay} : i1')
    lines.append(f'  scf.if {bad} {{')
    label = emit(f'arith.constant {number} : i64')
    lines.extend([f'    func.call @printI64({label}) : (i64) -> ()',
                  '    func.call @printNewline() : () -> ()'])
    for value in (actual, ref):
        shown = emit(f'arith.extf {value} : f32 to f64') if real == 'f32' else value
        bits = emit(f'arith.bitcast {shown} : f64 to i64')
        lines.extend([f'    func.call @printI64({bits}) : (i64) -> ()',
                      '    func.call @printNewline() : () -> ()'])
    lines.append('  }')


for row, values in enumerate(forces):
    for axis, expected in enumerate(values):
        value = emit(f'memref.load {force}[{index(row)}, {index(axis)}] : memref<153x3xf32>')
        check(value, expected, 'f32', 3 * row + axis)
lanes = index(32)
for column, expected in enumerate(totals):
    col = index(column)
    name = f'%sum{column}'
    lines.extend([f'  {name} = scf.for %lane = {zero} to {lanes} step {one} '
                  f'iter_args(%sum = {widezero}) -> (f64) {{',
                  f'    %part = memref.load {sums}[%lane, {col}] : memref<32x10xf64>',
                  '    %next = arith.addf %sum, %part : f64',
                  '    scf.yield %next : f64', '  }'])
    check(name, expected, 'f64', 459 + column)
flag = emit(f'arith.extui {good} : i1 to i64')
lines.extend([f'  func.call @printI64({flag}) : (i64) -> ()',
              '  func.call @printNewline() : () -> ()',
              f'  cf.assert {good}, "template scheduler numerical mismatch"',
              f'  memref.dealloc {force} : memref<153x3xf32>',
              f'  memref.dealloc {sums} : memref<32x10xf64>', '  return', '}'])
args.destination.write_text('\n'.join(lines) + '\n')
