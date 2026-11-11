"""Final-coordinate writer callback against dyadic whole-frame geometry.

Expected fractional coordinates use a rational general linear solve. All
serialized geometry is compared as exact bits; booleans use deliberately
separated displacements, never a tolerance near a rounding guard.
"""
import argparse
from fractions import Fraction
import json
import pathlib
import struct

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['generate', 'check'])
p.add_argument('program', type=pathlib.Path)
p.add_argument('reference', type=pathlib.Path)
p.add_argument('--triclinic', action='store_true')
p.add_argument('--point-offset', type=float, default=0.0)
a = p.parse_args()
if a.mode == 'check':
    got = [int(v) for v in a.program.read_text().split()]
    ref = json.loads(a.reference.read_text())
    assert len(got) == len(ref), (len(got), len(ref))
    errors = [(i, x, y) for i, (x, y) in enumerate(zip(got, ref)) if x != y]
    assert not errors, {'count': len(errors), 'first': errors[:8]}
    print(f'fused writer: {len(ref)} exact values; epochs, retained images, extent history and inactive state')
    raise SystemExit(0)

cell = [4., 4., 4.] + ([1., .5, 1.] if a.triclinic else [])
h = [[4., 1. if a.triclinic else 0., .5 if a.triclinic else 0.],
     [0., 4., 1. if a.triclinic else 0.], [0., 0., 4.]]


def shift(c):
    return [sum(v * w for v, w in zip(row, c)) for row in h]


def solve(x):
    m = [[Fraction(v) for v in row] + [Fraction(v)] for row, v in zip(h, x)]
    for i in range(3):
        d = m[i][i]
        m[i] = [v / d for v in m[i]]
        for j in range(3):
            if j != i:
                d = m[j][i]
                m[j] = [v - d * w for v, w in zip(m[j], m[i])]
    return [float(row[-1]) for row in m]


expected, lines = [], []
serial = 0


def bits(v, kind):
    if kind == 'i32':
        return int(v)
    return int.from_bytes(struct.pack('>d' if kind == 'f64' else '>f', v),
                          'big', signed=kind == 'f64')


def emit(text):
    global serial
    name = f'%v{serial}'
    serial += 1
    lines.append(f'  {name} = {text}')
    return name


def alloc(name, data, kind):
    shape = f'{len(data)}x{len(data[0])}x{kind}'
    globals_.append(f'memref.global "private" constant @{name} : memref<{shape}> = dense<{json.dumps(data)}>')
    source = emit(f'memref.get_global @{name} : memref<{shape}>')
    device = emit(f'gpu.alloc() : memref<{shape}, 1>')
    ready = emit('gpu.wait async')
    token = emit(f'gpu.memcpy async [{ready}] {device}, {source} : memref<{shape}, 1>, memref<{shape}>')
    lines.append(f'  gpu.wait [{token}]')
    view = emit(f'memref.cast {device} : memref<{shape}, 1> to memref<?x{len(data[0])}x{kind}, 1>')
    return device, view, shape


globals_ = []
lines.append('func.func @main() {')
# The final case is a fresh list after constraints: no epoch view is
# touched, but an overextended alpha=.75 site must still report failure.
for case, (built, constrained) in enumerate([(True, False), (True, True),
                                           (False, False), (False, True)]):
    anchors = [[.5 + u / 8, .25, .125] for u in range(4)]
    cells = [[-17., 9., -5.] for _ in range(4)]
    positions, whole = [], []
    for u, anchor in enumerate(anchors):
        offset = shift([-c for c in cells[u]])
        world = [x + y for x, y in zip(anchor, offset)]
        local = [anchor]
        positions.append(world)
        if u < 2:
            # Parallel bonds make alpha=.75's site the limiting extent.
            d = .125 if u == 0 else .25
            for image in ([0, 1, -1], [1, 0, 1]):
                local.append([anchor[0] + d, anchor[1], anchor[2]])
                image_shift = shift(image)
                positions.append([world[k] + (d if k == 0 else 0.) + image_shift[k]
                                  for k in range(3)])
        whole.extend(local)
    # With the point offset t, every point and reference is the fraction
    # plus t, and the displacement tests compare the same differences.
    t = a.point_offset
    def point(x):
        return [f + t for f in solve(x)]
    fractions = [point(x) for x in anchors]
    outer = [point([x[0] - (0. if u % 2 == 0 else .5), x[1], x[2]])
             for u, x in enumerate(anchors)]
    inner = [point([x[0] - (0. if u % 2 else .5), x[1], x[2]])
             for u, x in enumerate(anchors)]
    packed = whole if built else [[-7.] * 3 for _ in range(8)]
    current = fractions if built else [[-9.] * 3 for _ in range(4)]
    flags = [[int(built and u < 2 and u % 2 == 1),
              int(built and u >= 2 and u % 2 == 1),
              int(built and u < 2 and u % 2 == 0),
              int(built and u >= 2 and u % 2 == 0),
              int(constrained and u == 1)] for u in range(4)]
    data = [positions, [[-7.] * 3 for _ in range(8)], cells,
            [[-9.] * 3 for _ in range(4)], outer, inner, [[-1] * 5 for _ in range(4)]]
    kinds = ['f64', 'f32', 'f64', 'f64', 'f64', 'f64', 'i32']
    buffers = [alloc(f'case{case}_{i}', values, kind)
               for i, (values, kind) in enumerate(zip(data, kinds))]
    w = emit('arith.constant 2 : index')
    c = emit(f'arith.constant dense<{json.dumps(cell)}> : vector<{len(cell)}xf64>')
    b = emit(f'arith.constant {str(built).lower()}')
    constrained_value = emit(f'arith.constant {str(constrained).lower()}')
    margins = emit('arith.constant dense<[0.25, 0.25, 0.125, 0.125]> : vector<4xf64>')
    types = [f'memref<?x{len(v[0])}x{k}, 1>' for v, k in zip(data, kinds)]
    types += ['index', f'vector<{len(cell)}xf64>', 'i1', 'i1', 'vector<4xf64>']
    arguments = [v[1] for v in buffers] + [w, c, b, constrained_value, margins]
    lines.append(f'  func.call @writer_frame({", ".join(arguments)}) : ({", ".join(types)}) -> ()')
    for i, reference in [(1, packed), (3, current), (2, cells), (4, outer), (5, inner), (6, flags)]:
        device, _, shape = buffers[i]
        kind = kinds[i]
        host = emit(f'memref.alloc() : memref<{shape}>')
        ready = emit('gpu.wait async')
        token = emit(f'gpu.memcpy async [{ready}] {host}, {device} : memref<{shape}>, memref<{shape}, 1>')
        lines.append(f'  gpu.wait [{token}]')
        for row, values in enumerate(reference):
            for col, value in enumerate(values):
                r = emit(f'arith.constant {row} : index')
                s = emit(f'arith.constant {col} : index')
                loaded = emit(f'memref.load {host}[{r}, {s}] : memref<{shape}>')
                if kind != 'i32':
                    loaded = emit(f'arith.bitcast {loaded} : {kind} to i{64 if kind == "f64" else 32}')
                if kind != 'f64':
                    loaded = emit(f'arith.extui {loaded} : i32 to i64')
                lines.extend([f'  func.call @printI64({loaded}) : (i64) -> ()',
                              '  func.call @printNewline() : () -> ()'])
                expected.append(bits(value, kind))
        lines.append(f'  memref.dealloc {host} : memref<{shape}>')
    for device, _, shape in buffers:
        generic = emit(f'memref.memory_space_cast {device} : memref<{shape}, 1> to memref<{shape}>')
        lines.append(f'  gpu.dealloc {generic} : memref<{shape}>')
lines.extend(['  return', '}'])
declarations = ['func.func private @printI64(i64)', 'func.func private @printNewline()',
                f'func.func private @writer_frame({", ".join(types)})']
a.program.write_text('\n'.join(globals_ + declarations + lines) + '\n')
a.reference.write_text(json.dumps(expected) + '\n')
