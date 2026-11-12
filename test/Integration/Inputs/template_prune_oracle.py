"""All-images rational oracle for fixed-block template pruning.

The reference enumerates each preexisting I mask directly. No ballots,
lane ranks, device loop order or floating-point displacement is reused.
Each activity combination starts with deliberately shorter inner blocks,
so a requested prune must recover entries from the immutable outer list.
"""
import argparse
from fractions import Fraction as F
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['generate', 'check'])
p.add_argument('program', type=pathlib.Path)
p.add_argument('reference', type=pathlib.Path)
p.add_argument('--triclinic', action='store_true')
a = p.parse_args()
if a.mode == 'check':
    got = [int(v) for v in a.program.read_text().split()]
    expected = json.loads(a.reference.read_text())
    assert len(got) == len(expected), (len(got), len(expected))
    differences = [(i, x, y) for i, (x, y) in enumerate(zip(got, expected)) if x != y]
    assert not differences, {'count': len(differences), 'first': differences[:8]}
    print(f'fused pruning: {len(expected)} exact integers; all classes and activity combinations; immediate same-kernel consumption')
    raise SystemExit(0)

waters, atoms = 19, 17
cell = [4., 4., 4.] + ([1., .5, 1.] if a.triclinic else [])
h = [[F(4), F(1) if a.triclinic else F(0), F(1, 2) if a.triclinic else F(0)],
     [F(0), F(4), F(1) if a.triclinic else F(0)], [F(0), F(0), F(4)]]
order = list(range(waters)) + [-1] * 13 + list(range(waters, waters + atoms)) + [-1] * 15
references = [[F((u * 5) % 17, 32), F((u * 7) % 19, 32), F((u * 3) % 13, 32)]
              for u in range(waters + atoms)]
current = [[x + F((u % 3) - 1, 8), y - F(1, 16), z + F(u % 2, 8)]
           for u, (x, y, z) in enumerate(references)]
reaches = [F(3, 2), F(5, 4), F(1)]
# Rows of the lists: a row's outer records are its chunks of 64 entries,
# [class | chunk << 2, first, chunk begin, row end] (U10). The prune
# compacts a row's surviving entries from the row's first entry across its
# chunks, and record k keeps those of rank 64 k to 64 k + 63: of a row of
# four chunks 133 entries survive (two full records, one of 5, one empty),
# of one of three 117 (a full record, one of 53, one empty).
rows = [(0, 0, 64), (0, 0, 3), (0, 16, 0), (1, 32, 33), (1, 48, 17), (1, 32, 1),
        (2, 32, 32), (2, 48, 7), (0, 0, 230), (2, 32, 180)]
units, entries, slices = [], [], []
for row, (tag, first, count) in enumerate(rows):
    begin = len(entries)
    for j in range(count):
        partner = (j * 7 + row * 3) % (waters if tag != 2 else atoms)
        if tag == 2:
            partner += waters
        image = [(j % 3) - 1 if j % 5 == 0 else 0,
                 1 if j % 11 == 0 else 0, -1 if j % 13 == 0 else 0]
        code = sum((c + 512) << (10 * axis) for axis, c in enumerate(image))
        mask = ((0xffff ^ (j * 257)) & 0xffff) or 1
        entries.append([partner, code, mask, 100 + j])
    for chunk in range(max(1, -(-count // 64))):
        units.append([tag | chunk << 2, first, begin + 64 * chunk, len(entries)])
        slices.append((tag, first, begin, chunk, min(begin + 64 * (chunk + 1), len(entries))))


def flatten(rows):
    return [v for row in rows for v in row]


def retained(record, first, water_active, atom_active):
    j, code, mask, marker = record
    active = lambda u: water_active if u < waters else atom_active
    points = [current[u] if active(u) else references[u] for u in range(waters + atoms)]
    image = [F(((code >> (10 * k)) & 1023) - 512) for k in range(3)]
    new_mask = 0
    for bit, i in enumerate(order[first:first + 16]):
        if i < 0 or not (mask & (1 << bit)):
            continue
        delta = [x - y - z for x, y, z in zip(points[i], points[j], image)]
        d = [sum(x * y for x, y in zip(row, delta)) for row in h]
        if sum(x * x for x in d) <= reaches[tag] ** 2:
            new_mask |= 1 << bit
    return [j, code, new_mask, marker]


globals_, lines, expected = [], ['func.func @main() {'], []
serial = 0


def emit(text):
    global serial
    name = f'%v{serial}'
    serial += 1
    lines.append(f'  {name} = {text}')
    return name


def alloc(name, data, kind, width=None):
    shape = f'{len(data)}x{kind}' if width is None else f'{len(data)}x{width}x{kind}'
    globals_.append(f'memref.global "private" constant @{name} : memref<{shape}> = dense<{json.dumps(data)}>')
    host = emit(f'memref.get_global @{name} : memref<{shape}>')
    device = emit(f'gpu.alloc() : memref<{shape}, 1>')
    ready = emit('gpu.wait async')
    token = emit(f'gpu.memcpy async [{ready}] {device}, {host} : memref<{shape}, 1>, memref<{shape}>')
    lines.append(f'  gpu.wait [{token}]')
    dynamic = f'?x{kind}' if width is None else f'?x{width}x{kind}'
    view = emit(f'memref.cast {device} : memref<{shape}, 1> to memref<{dynamic}, 1>')
    return device, view, shape


# Comparison is integer-only, including reserved words and untouched tails.
for phase, (wa, aa) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
    seed_units = [[tag, first, begin + 64 * chunk, min(end, begin + 64 * chunk + 2)]
                  for tag, first, begin, chunk, end in slices]
    inner_units = [r[:] for r in seed_units]
    inner_entries = [r[:] for r in entries]
    for record in inner_entries:
        record[2] &= 0x5555
    for work, (tag, first, begin, chunk, _) in enumerate(slices):
        active = wa if tag == 0 else aa if tag == 2 else wa or aa
        if active:
            end = units[work][3]
            kept = [r for record in entries[begin:end]
                    if (r := retained(record, first, wa, aa))[2]]
            inner_entries[begin:begin + len(kept)] = kept
            inner_units[work][3] = begin + min(64 * (chunk + 1), max(64 * chunk, len(kept)))
    observed = [[-999] * 4 for _ in entries]
    for _, _, begin, end in inner_units:
        observed[begin:end] = inner_entries[begin:end]
    seed_entries = [[j, code, mask & 0x5555, marker] for j, code, mask, marker in entries]
    integer_data = [flatten(units), flatten(seed_units), flatten(entries), flatten(seed_entries), order]
    buffers = [alloc(f'p{phase}_{i}', values, 'i32') for i, values in enumerate(integer_data)]
    buffers += [alloc(f'p{phase}_current', [[float(v) for v in row] for row in current], 'f64', 3),
                alloc(f'p{phase}_reference', [[float(v) for v in row] for row in references], 'f64', 3)]
    observed_buffer = alloc(f'p{phase}_observed', [-999] * (4 * len(entries)), 'i32')
    w = emit(f'arith.constant {waters} : index')
    c = emit(f'arith.constant dense<{json.dumps(cell)}> : vector<{len(cell)}xf64>')
    r = emit('arith.constant dense<[1.5, 1.25, 1.0]> : vector<3xf64>')
    yes_w = emit(f'arith.constant {str(wa).lower()}')
    yes_a = emit(f'arith.constant {str(aa).lower()}')
    types = ['memref<?xi32, 1>'] * 5 + ['memref<?x3xf64, 1>'] * 2
    types += ['index', f'vector<{len(cell)}xf64>', 'vector<3xf64>', 'i1', 'i1', 'memref<?xi32, 1>', 'memref<?xi32, 1>']
    lengths = [0] * len(order)
    for start in (0, 16, 32, 48):
        lengths[start] = sum(i >= 0 for i in order[start:start + 16])
    length_buffer = alloc(f'p{phase}_lengths', lengths, 'i32')
    args = [v[1] for v in buffers] + [w, c, r, yes_w, yes_a, observed_buffer[1], length_buffer[1]]
    lines.append(f'  func.call @prune_work({", ".join(args)}) : ({", ".join(types)}) -> ()')
    checks = [(buffers[0], flatten(units)), (buffers[1], flatten(inner_units)),
              (buffers[2], flatten(entries)), (buffers[3], flatten(inner_entries)),
              (buffers[4], order), (observed_buffer, flatten(observed))]
    for (device, _, shape), values in checks:
        host = emit(f'memref.alloc() : memref<{shape}>')
        ready = emit('gpu.wait async')
        token = emit(f'gpu.memcpy async [{ready}] {host}, {device} : memref<{shape}>, memref<{shape}, 1>')
        lines.append(f'  gpu.wait [{token}]')
        # A loop keeps the emitted program independent of the record count.
        zero = emit('arith.constant 0 : index')
        stop = emit(f'arith.constant {len(values)} : index')
        one = emit('arith.constant 1 : index')
        lines.append(f'  scf.for %i{serial} = {zero} to {stop} step {one} {{')
        loaded = emit(f'memref.load {host}[%i{serial}] : memref<{shape}>')
        wide = emit(f'arith.extsi {loaded} : i32 to i64')
        lines.extend([f'  func.call @printI64({wide}) : (i64) -> ()',
                      '  func.call @printNewline() : () -> ()', '  }'])
        lines.append(f'  memref.dealloc {host} : memref<{shape}>')
        expected.extend(values)
    for device, _, shape in buffers + [observed_buffer, length_buffer]:
        generic = emit(f'memref.memory_space_cast {device} : memref<{shape}, 1> to memref<{shape}>')
        lines.append(f'  gpu.dealloc {generic} : memref<{shape}>')
lines.extend(['  return', '}'])
declarations = ['func.func private @printI64(i64)', 'func.func private @printNewline()',
                f'func.func private @prune_work({", ".join(types)})']
a.program.write_text('\n'.join(globals_ + declarations + lines) + '\n')
a.reference.write_text(json.dumps(expected) + '\n')
