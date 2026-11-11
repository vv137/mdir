"""Independent dyadic geometry for outer-build snapshots, serialized as bits.

Known whole frames define the expected packed geometry. A rational general
linear solve supplies reference fractions; independently imaged hydrogens
and unused stored EP coordinates appear only in the input. Partial group
activity must retain inactive views. All bounds are zero, including bits.
"""
import argparse
from fractions import Fraction
import json
import pathlib
import struct

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
generate = sub.add_parser('generate')
generate.add_argument('program', type=pathlib.Path)
generate.add_argument('reference', type=pathlib.Path)
generate.add_argument('--triclinic', action='store_true')
generate.add_argument('--three-site', action='store_true')
generate.add_argument('--point-offset', type=float, default=0.0)
check = sub.add_parser('check')
check.add_argument('records', type=pathlib.Path)
check.add_argument('reference', type=pathlib.Path)
args = parser.parse_args()

if args.command == 'check':
    expected = json.loads(args.reference.read_text())
    got = [int(v) for v in args.records.read_text().split()]
    assert len(got) == len(expected), (len(got), len(expected))
    errors = []
    for i, (actual, ref) in enumerate(zip(got, expected)):
        if actual != ref['bits']:
            width = 64 if ref['kind'] == 'f64' else 32
            bits = (actual & ((1 << width) - 1)).to_bytes(width // 8, 'big')
            value = struct.unpack('>d' if width == 64 else '>f', bits)[0]
            errors.append((i, ref['label'], value, ref['value'], bits.hex()))
    assert not errors, {'count': len(errors), 'first': errors[:8]}
    print(f'frame snapshots: {len(expected)} bitwise-identical values; inactive views retained; stored EPs unused')
    raise SystemExit(0)

waters, atoms = 3, 2
sites = 3 if args.three_site else 4
particles, units, rows = sites * waters + atoms, waters + atoms, 3 * waters + atoms
permutation = [(5 * i + 3) % particles for i in range(particles)]
water = [permutation[3 * i:3 * i + 3] for i in range(waters)]
extra = [] if sites == 3 else permutation[3 * waters:4 * waters]
if extra:
    water = [group + [extra[i]] for i, group in enumerate(water)]
solute = sorted(permutation[sites * waters:])
cell = [4.0, 4.0, 4.0] + ([1.0, .5, 1.0] if args.triclinic else [])
matrix = [[4, 1 if args.triclinic else 0, .5 if args.triclinic else 0],
          [0, 4, 1 if args.triclinic else 0], [0, 0, 4]]


def translate(winding):
    return [sum(matrix[i][j] * winding[j] for j in range(3)) for i in range(3)]


def solve(point):
    a = [[Fraction(x) for x in row] + [Fraction(y)] for row, y in zip(matrix, point)]
    for column in range(3):
        pivot = a[column][column]
        a[column] = [v / pivot for v in a[column]]
        for row in range(3):
            if row != column:
                factor = a[row][column]
                a[row] = [v - factor * p for v, p in zip(a[row], a[column])]
    return [float(row[-1]) for row in a]


reference = []
point_offset = args.point_offset


def central_cells(point):
    """The sequential central-image lattice cells of `point` (c, then b,
    then a), with ties to even, as the packing takes them."""
    raw, cells = list(point), [0.0] * 3
    for axis in (2, 1, 0):
        count = round(raw[axis] / matrix[axis][axis])
        cells[axis] = -float(count)
        for row in range(axis):
            raw[row] -= count * matrix[row][axis]
    return cells


def packing(anchor):
    """With the point offset t, the packing cell is c - t with c the
    central-image cell of x - H t; the frame is x + H(c - t) and the point
    its fraction plus t."""
    moved = [a - s for a, s in zip(anchor, translate([point_offset] * 3))]
    cells = [c - point_offset for c in central_cells(moved)]
    framed = [a + s for a, s in zip(anchor, translate(cells))]
    return cells, framed, [f + point_offset for f in solve(framed)]


def record(kind, label, value):
    data = struct.pack('>d' if kind == 'f64' else '>f', value)
    bits = int.from_bytes(data, 'big', signed=kind == 'f64')
    reference.append({'kind': kind, 'label': label, 'value': value, 'bits': bits})


phases = []
cached_cells, cached_fraction = [None] * units, [None] * units
packed = [None] * rows
for phase, active in enumerate([(True, True), (False, True), (True, False), (False, False)]):
    positions = [[0.0] * 3 for _ in range(particles)]
    for unit in range(units):
        winding = [17 + unit, -9 - unit, 5 + unit]
        x = (.25 if phase == 0 else .5 if phase == 1 else 2.0625 if phase == 2 else .75) + unit / 16
        local = [x, .375 + unit / 32, .25 + unit / 64]
        offset = translate(winding)
        anchor = [a + b for a, b in zip(offset, local)]
        group = water[unit][:3] if unit < waters else [solute[unit - waters]]
        positions[group[0]] = anchor
        if unit < waters:
            for site, image in [(1, (0, 1, -1)), (2, (1, 0, 1))]:
                shift = translate(image)
                positions[group[site]] = [anchor[k] + shift[k] + (.125 if k == site - 1 else 0)
                                          for k in range(3)]
        if active[0 if unit < waters else 1]:
            crossing = int(phase == 2)
            cached_cells[unit] = [-float(winding[0] + crossing), -float(winding[1]), -float(winding[2])]
            framed = [local[0] - 4 * crossing, local[1], local[2]]
            cached_fraction[unit] = solve(framed)
            general = packing(anchor)
            if point_offset == 0.0:
                # The general packing agrees with the windings at t = 0.
                assert general == (cached_cells[unit], framed, cached_fraction[unit]), general
            else:
                cached_cells[unit], framed, cached_fraction[unit] = general
            row = 3 * unit if unit < waters else 2 * waters + unit
            packed[row] = framed
            if unit < waters:
                packed[row + 1] = [framed[0] + .125, framed[1], framed[2]]
                packed[row + 2] = [framed[0], framed[1] + .125, framed[2]]
    for i, particle in enumerate(extra):
        positions[particle] = [float(100000 + i), float(100000 - 2 * i), float(100000 + 3 * i)]
    phases.append({'active': active, 'positions': positions})
    for label, values in [('outer', cached_fraction), ('cells', cached_cells), ('current', cached_fraction)]:
        for row, point in enumerate(values):
            for axis, value in enumerate(point):
                record('f64', f'{phase}:{label}:{row}:{axis}', value)
    for row, point in enumerate(packed):
        for axis, value in enumerate(point):
            record('f32', f'{phase}:packed:{row}:{axis}', value)
    extent = (9 / 512 if sites == 4 else 1 / 64) if active[0] else 0.0
    record('f64', f'{phase}:extent2', extent)

cell_type = f'vector<{len(cell)}xf64>'
lines = ['md.particle_set @atoms', '!list = !mdrt.segmented_pairs<@atoms>']
lines.append(f'memref.global "private" constant @water : memref<{waters}x{sites}xi32> = dense<{water}>')
for phase, ref in enumerate(phases):
    lines.append(f'memref.global "private" constant @positions_{phase} : memref<{particles}x3xf64> = dense<{ref["positions"]}>')
lines.extend([
    f'func.func private @snapshot_seed(!list, memref<?x3xf64, 1>, {cell_type}, i1, i1)',
    'func.func private @mdrtTemplatePairsFrameState(!list, i64) -> memref<?x3xf64, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsFrames(!list) -> memref<?x3xf32, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsCounters(!list) -> memref<4xi64, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
    'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
    'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
    'func.func private @printI64(i64)', 'func.func private @printNewline()'])
for real, width in [('f64', 64), ('f32', 32)]:
    lines.append(f'''
func.func @dump_{real}(%device: memref<?x3x{real}, 1>) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %device, %c0 : memref<?x3x{real}, 1>
  %host = memref.alloc(%n) : memref<?x3x{real}>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?x3x{real}>, memref<?x3x{real}, 1>
  gpu.wait [%done]
  scf.for %row = %c0 to %n step %c1 {{
    scf.for %axis = %c0 to %c3 step %c1 {{
      %value = memref.load %host[%row, %axis] : memref<?x3x{real}>
      %raw = arith.bitcast %value : {real} to i{width}
''')
    printed = '%raw'
    if width == 32:
        lines.append('      %wide = arith.extui %raw : i32 to i64')
        printed = '%wide'
    lines.extend([f'      func.call @printI64({printed}) : (i64) -> ()',
                  '      func.call @printNewline() : () -> ()', '    }', '  }',
                  f'  memref.dealloc %host : memref<?x3x{real}>', '  return', '}'])
lines.append(r'''
func.func @dump_extent(%list: !list) {
  %c3 = arith.constant 3 : index
  %device = func.call @mdrtTemplatePairsCounters(%list) : (!list) -> memref<4xi64, 1>
  %host = memref.alloca() : memref<4xi64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<4xi64>, memref<4xi64, 1>
  gpu.wait [%done]
  %bits = memref.load %host[%c3] : memref<4xi64>
  func.call @printI64(%bits) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
''')
serial = 0


def emit(text):
    global serial
    name = f'%v{serial}'
    serial += 1
    lines.append(f'  {name} = {text}')
    return name


n = emit(f'arith.constant {particles} : index')
types = emit(f'gpu.alloc({n}) : memref<?xi32, 1>')
positions = emit(f'gpu.alloc({n}) : memref<?x3xf64, 1>')
w = emit(f'memref.get_global @water : memref<{waters}x{sites}xi32>')
w = emit(f'memref.cast {w} : memref<{waters}x{sites}xi32> to memref<?x{sites}xi32>')
state = emit(f'md_exec.empty_segmented_pairs {types}, {w} cutoff(1.0) skin(0.25) extent(0.2) alpha(0.75) : memref<?xi32, 1>, memref<?x{sites}xi32> -> !list')
cell_value = emit(f'arith.constant dense<{cell}> : {cell_type}')
for phase, ref in enumerate(phases):
    host = emit(f'memref.get_global @positions_{phase} : memref<{particles}x3xf64>')
    host = emit(f'memref.cast {host} : memref<{particles}x3xf64> to memref<?x3xf64>')
    ready = emit('gpu.wait async')
    done = emit(f'gpu.memcpy async [{ready}] {positions}, {host} : memref<?x3xf64, 1>, memref<?x3xf64>')
    lines.append(f'  gpu.wait [{done}]')
    wa, aa = [emit(f'arith.constant {str(flag).lower()}') for flag in ref['active']]
    lines.append(f'  func.call @snapshot_seed({state}, {positions}, {cell_value}, {wa}, {aa}) : (!list, memref<?x3xf64, 1>, {cell_type}, i1, i1) -> ()')
    for slot in [0, 2, 3]:
        which = emit(f'arith.constant {slot} : i64')
        view = emit(f'func.call @mdrtTemplatePairsFrameState({state}, {which}) : (!list, i64) -> memref<?x3xf64, 1>')
        lines.append(f'  func.call @dump_f64({view}) : (memref<?x3xf64, 1>) -> ()')
    view = emit(f'func.call @mdrtTemplatePairsFrames({state}) : (!list) -> memref<?x3xf32, 1>')
    lines.append(f'  func.call @dump_f32({view}) : (memref<?x3xf32, 1>) -> ()')
    lines.append(f'  func.call @dump_extent({state}) : (!list) -> ()')
lines.extend(['  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()', '  return', '}'])
args.program.write_text('\n'.join(lines) + '\n')
args.reference.write_text(json.dumps(reference) + '\n')
print('snapshot reference:', sites, 'sites;', 'triclinic' if args.triclinic else 'orthorhombic',
      '; point offset', point_offset, ';', len(reference), 'exact values; bound 0')
