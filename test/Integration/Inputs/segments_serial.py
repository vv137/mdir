"""The serial definition of md_alg.traverse_pairs over the segments of a
list of units (D417) against the f64 oracle of D405.

    segments_serial.py emit  SYSTEM.top SYSTEM.gro PROGRAM.mlir KINDS OUT.mlir [--periodic]
    segments_serial.py check SYSTEM.top SYSTEM.gro KINDS VALUES.txt

PROGRAM.mlir is a program after convert-md-exec-to-md-alg: its one
md_alg.traverse_pairs over the segments, in the structure form. `emit`
writes a program that runs that op in its buffer form over a list of units
that this script builds itself, every pair of units of every segment once
(a superset of the pairs within the cutoff, which the op tests site pair by
site pair), and prints the energy, the force of each particle, and the
virial as the bits of an f64. `check` compares them with the oracle of
test/Driver/Inputs/kind_pairs_oracle.py, which reads the topology and the
coordinates itself.

The list: the units of each kind in groups of 16 places, a place of a taken
kind holding the position of its unit's first member in `members`; for each
segment (f, g), each group of kind f takes every unit of kind g that comes
after its own (f = g) or every unit (f != g), 64 entries a unit of work,
with the bit of each place whose unit pairs with the partner. Pairs of the
rest within one molecule are excluded (the topology excludes them all).
With --periodic the units straddle the faces of the cell and the places and
entries take shifts (build_list); the bound is the same, since the oracle's
displacements are those of the file, and the members of a unit are stored
whole about its anchor, the precondition of the buffer form.

The bound of a component is the sum over the tuples of the oracle's own
bound (the rounding of one tuple's window, D405) plus the reordering of the
sum over the T tuples, 2 gamma_T times the sum of the magnitudes of their
contributions, with gamma_T = T eps / (1 - T eps) and eps = 2^-53.
"""
import math
import re
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / 'Driver' / 'Inputs'))
from kind_pairs_oracle import EPS, System  # noqa: E402

GROUP = 16
BLOCK = 64


def read_box(gro):
    return [float(x) for x in Path(gro).read_text().splitlines()[-1].split()[:3]]


def build_list(system, box, periodic):
    """The buffers of the buffer form, and the positions as stored.

    Without `periodic` the positions are those of the file and every shift
    is zero. With it, the system is moved so that its middle is at a corner
    of the cell, and each unit is stored whole, its anchor wrapped into the
    cell: the units straddle the faces and pairs cross them. The place of a
    unit u in group G then has the shift n_u + g_G, n_u the cells its anchor
    was moved by and g_G a lattice vector of the group, so that the places
    of a group are in its frame; an entry of group G whose partner is in
    group H has the shift g_G - g_H. Its displacement is that of the file."""
    K = len(system.taken)
    units = [list(system.molecules[name]) for name in system.taken]
    units.append([[p] for p, _ in system.rest])
    home = {p: m for p, m in system.rest}
    positions = [list(x) for x in system.positions]
    cells = {}
    if periodic:
        middle = [sum(x[k] for x in system.positions) / len(positions) for k in range(3)]
        moved = [[x[k] - middle[k] for k in range(3)] for x in system.positions]
        for f in range(K + 1):
            for unit in units[f]:
                anchor = moved[unit[0]]
                n = [math.floor(anchor[k] / box[k]) for k in range(3)]
                cells[tuple(unit)] = n
                for p in unit:
                    positions[p] = [moved[p][k] - n[k] * box[k] for k in range(3)]
    order, members, place_of = [], [], {}
    groups_of, frame_of = [], {}
    for f in range(K + 1):
        groups = []
        for start in range(0, len(units[f]), GROUP):
            group = len(order) // GROUP
            groups.append(group)
            frame_of[group] = [(group % 3) - 1, ((group // 3) % 3) - 1, 1] if periodic else [0, 0, 0]
            for unit in units[f][start:start + GROUP]:
                place_of[(f, tuple(unit))] = len(order)
                if f == K:
                    order.append(unit[0])
                else:
                    frame = system.frame(system.taken[f])
                    order.append(len(members))
                    members += [unit[i] for i in frame]
            while len(order) % GROUP:
                order.append(-1)
        groups_of.append(groups)
    unit_at = {}
    for f in range(K + 1):
        for unit in units[f]:
            unit_at[place_of[(f, tuple(unit))]] = tuple(unit)
    shifts = []
    for place in range(len(order)):
        n = cells.get(unit_at.get(place), [0, 0, 0])
        g = frame_of[place // GROUP]
        s = [n[k] + g[k] + 512 for k in range(3)]
        shifts.append(s[0] | (s[1] << 10) | (s[2] << 20))
    offset = 4  # an entry's shift of zero cells (4 bits an axis, D95)
    entries, masks, counts, unit_groups, ordinals, segments = [], [], [], [], [], [0]
    for f in range(K + 1):
        for g in range(f + 1):
            for group in groups_of[f]:
                rows = []
                for b, partner in enumerate(units[g]):
                    bits = 0
                    for lane in range(GROUP):
                        place = group * GROUP + lane
                        if order[place] < 0:
                            continue
                        a = units[f][(place - groups_of[f][0] * GROUP)]
                        if f == g and units[f].index(a) >= b:
                            continue
                        if f == g == K and home[a[0]] == home[partner[0]]:
                            continue
                        bits |= 1 << lane
                    if bits:
                        other = place_of[(g, tuple(partner))]
                        e = [frame_of[group][k] - frame_of[other // GROUP][k] + offset
                             for k in range(3)]
                        assert all(0 <= x < 16 for x in e)
                        shift_bits = (e[0] << 16) | (e[1] << 20) | (e[2] << 24)
                        rows.append((other, bits | shift_bits))
                for block in range(0, len(rows), BLOCK):
                    chunk = rows[block:block + BLOCK]
                    entries += [p for p, _ in chunk] + [0] * (BLOCK - len(chunk))
                    masks += [m for _, m in chunk] + [0] * (BLOCK - len(chunk))
                    counts.append(len(chunk))
                    unit_groups.append(group)
                    ordinals.append(block // BLOCK)
            segments.append(len(counts))
    lists = {'entries': entries, 'masks': masks, 'counts': counts,
             'units': unit_groups, 'ordinals': ordinals, 'places': order,
             'shifts': shifts, 'segments': segments, 'members': members or [0]}
    straddling = sum(1 for f in range(K) for unit in units[f]
                     if len({tuple(math.floor(positions[p][k] / box[k]) for k in range(3))
                             for p in unit}) > 1)
    return lists, positions, straddling


def extract(program):
    """The op's text, the names of its captured tables, and its result types."""
    text = Path(program).read_text()
    start = text.index('md_alg.traverse_pairs structure')
    line_start = text.rfind('\n', 0, start) + 1
    results = re.match(r'\s*%[0-9]+(?::([0-9]+))?\s*=\s*', text[line_start:start])
    # The op ends after the last region: the '}' followed by '->' or the end
    # of the line.
    depth, i, seen = 0, start, False
    while True:
        c = text[i]
        if c == '{':
            depth += 1
            seen = True
        elif c == '}':
            depth -= 1
            if depth == 0 and seen:
                rest = text[i + 1:]
                nxt = rest.lstrip(' ')
                if not nxt.startswith('segment('):
                    end_line = text.index('\n', i)
                    return text[start:end_line]
        i += 1


def emit(system, program, out, box, periodic):
    op = extract(program)
    header, regions = op.split(' segment(', 1)
    regions = 'segment(' + regions
    lists, stored, straddling = build_list(system, box, periodic)
    if periodic:
        assert straddling, 'no unit straddles a face of the cell'
    n = len(system.positions)
    lines = ['module {', 'func.func private @printI64(i64)',
             'func.func private @printNewline()', 'func.func @main() {']
    counter = [0]

    def value(expression):
        counter[0] += 1
        name = f'%z{counter[0]}'
        lines.append(f'  {name} = {expression}')
        return name

    def index(k):
        return value(f'arith.constant {k} : index')

    def buffer(name, data, element='i32', width=None):
        size = index(len(data))
        shape = f'?x{width}x{element}' if width else f'?x{element}'
        lines.append(f'  %{name} = memref.alloc({size}) : memref<{shape}>')
        for k, row in enumerate(data):
            at = index(k)
            for c, x in enumerate(row if width else [row]):
                v = value(f'arith.constant {x!r} : {element}' if element == 'i32'
                          else f'arith.constant {x:.17e} : {element}')
                if width:
                    lines.append(f'  memref.store {v}, %{name}[{at}, {index(c)}] : memref<{shape}>')
                else:
                    lines.append(f'  memref.store {v}, %{name}[{at}] : memref<{shape}>')
        return f'%{name}', f'memref<{shape}>'

    order = ['entries', 'masks', 'counts', 'units', 'ordinals', 'places',
             'shifts', 'segments', 'members']
    names = [buffer('l_' + key, lists[key]) for key in order]
    positions = buffer('x', stored, 'f64', 3)
    types = buffer('types', system.type_of)
    charges = buffer('charges', system.charge_of, 'f64')
    forces = buffer('f', [[0.0, 0.0, 0.0] for _ in range(n)], 'f64', 3)
    ntypes = len(system.sigma)
    sigma = [[0.5 * (system.sigma[a] + system.sigma[b]) for b in range(ntypes)]
             for a in range(ntypes)]
    epsilon = [[math.sqrt(system.epsilon[a] * system.epsilon[b])
                for b in range(ntypes)] for a in range(ntypes)]
    tables = [buffer('table_sigma', sigma, 'f64', ntypes),
              buffer('table_epsilon', epsilon, 'f64', ntypes)]
    cell = value(f'arith.constant dense<[{box[0]:.17e}, {box[1]:.17e}, {box[2]:.17e}]> : vector<3xf64>')
    places, nunits = index(len(lists['places'])), index(len(lists['counts']))

    # The op in its buffer form.
    m = re.search(r'reduce\(([^:]*):\s*([^)]*)\)', header)
    reduce_types = [t.strip() for t in m.group(2).split(',')]
    zeros = []
    for t in reduce_types:
        if t == 'f64':
            zeros.append(value('arith.constant 0.0 : f64'))
        else:
            zeros.append(value(f'arith.constant dense<0.0> : {t}'))
    contract = re.search(r'structure %[0-9]+ : (!mdrt\.segmented_pairs<.*?\]>)', header).group(1)
    new = header
    new = re.sub(r'structure %[0-9]+ : !mdrt\.segmented_pairs<.*?\]>',
                 'lists(' + ', '.join(name for name, _ in names) + ' : '
                 + ', '.join(t for _, t in names) + f') sizes({places}, {nunits})'
                 + f' contract({contract}) reach(1.0)', new, count=1)
    new = re.sub(r'positions\([^)]*\)', f'positions({positions[0]} : {positions[1]})', new, count=1)
    new = re.sub(r'cell\([^)]*\)', f'cell({cell} : vector<3xf64>)', new, count=1)
    new = re.sub(r'ins\([^)]*\)', f'ins({types[0]}, {charges[0]} : {types[1]}, {charges[1]})', new, count=1)
    new = re.sub(r'outs\([^)]*\)', f'outs({forces[0]} : {forces[1]})', new, count=1)
    new = re.sub(r'reduce\([^)]*\)', f'reduce({", ".join(zeros)} : {", ".join(reduce_types)})', new, count=1)
    assert 'ins(%arg3, %arg2' in header, 'the ins of the op are the types and the charges'
    # The tables the regions capture are the program's arguments 4 and 5;
    # their lookups read the buffers.
    body = re.sub(r'%arg4\b', tables[0][0], regions)
    body = re.sub(r'%arg5\b', tables[1][0], body)

    def lookup(match):
        result, table, a, b = match.group(1), match.group(2), match.group(3), match.group(4)
        counter[0] += 1
        i, j = f'%lookup{counter[0]}i', f'%lookup{counter[0]}j'
        return (f'{i} = arith.index_cast {a} : i32 to index\n'
                f'      {j} = arith.index_cast {b} : i32 to index\n'
                f'      {result} = memref.load {table}[{i}, {j}] : memref<?x{ntypes}xf64>')
    body = re.sub(r'(%[0-9a-z_]+) = md\.lookup (%[a-z_]+)\[(%[0-9a-z_]+), (%[0-9a-z_]+)\] : memref<\?x\?xf64>, i32, i32 -> f64',
                  lookup, body)
    tail = body.rsplit('}', 1)
    result_types = tail[1].strip()
    count = len(reduce_types)
    lines.append(f'  %r:{count} = {new} {tail[0]}}} {result_types}' if count > 1 else
                 f'  %r = {new} {tail[0]}}} {result_types}')

    def show(scalar):
        bits = value(f'arith.bitcast {scalar} : f64 to i64')
        lines.append(f'  func.call @printI64({bits}) : (i64) -> ()')
        lines.append('  func.call @printNewline() : () -> ()')

    # The energies and the virials of the windows, summed.
    energy = value('arith.constant 0.0 : f64')
    virial = [value('arith.constant 0.0 : f64') for _ in range(9)]
    for k, t in enumerate(reduce_types):
        name = f'%r#{k}' if count > 1 else '%r'
        if t == 'f64':
            energy = value(f'arith.addf {energy}, {name} : f64')
        else:
            for c in range(9):
                part = value(f'vector.extract {name}[{c}] : f64 from vector<9xf64>')
                virial[c] = value(f'arith.addf {virial[c]}, {part} : f64')
    show(energy)
    for p in range(n):
        for c in range(3):
            v = value(f'memref.load {forces[0]}[{index(p)}, {index(c)}] : {forces[1]}')
            show(v)
    for c in range(9):
        show(virial[c])
    lines += ['  return', '}', '}']
    Path(out).write_text('\n'.join(lines) + '\n')


def check(system, values):
    bits = [int(line) for line in Path(values).read_text().split()]
    got = [struct.unpack('<d', struct.pack('<q', b))[0] for b in bits]
    n = len(system.positions)
    energy, ebound, emag = 0.0, 0.0, 0.0
    forces = [[0.0] * 3 for _ in range(n)]
    fbound = [[0.0] * 3 for _ in range(n)]
    fmag = [[0.0] * 3 for _ in range(n)]
    strain, sbound, smag = [0.0] * 9, [0.0] * 9, [0.0] * 9
    tuples, segments = 0, set()
    for f, g, a, b in system.segments():
        (e, eb), (fs, fb), (s, sb), members = system.reference(f, g, a, b)
        energy += e
        ebound += eb
        emag += abs(e)
        for member, row, brow in zip(members, fs, fb):
            for c in range(3):
                forces[member][c] += row[c]
                fbound[member][c] += brow[c]
                fmag[member][c] += abs(row[c])
        for c in range(9):
            strain[c] += s[c]
            sbound[c] += sb[c]
            smag[c] += abs(s[c])
        tuples += 1
        segments.add((f, g))
    gamma = tuples * EPS / (1 - tuples * EPS)
    expected = [(energy, ebound + 2 * gamma * emag)]
    for p in range(n):
        for c in range(3):
            expected.append((forces[p][c], fbound[p][c] + 2 * gamma * fmag[p][c]))
    for c in range(9):
        expected.append((strain[c], sbound[c] + 2 * gamma * smag[c]))
    assert len(got) == len(expected), (len(got), len(expected))
    worst, failed, nonzero = 0.0, [], 0
    for k, ((reference, bound), value) in enumerate(zip(expected, got)):
        error = abs(value - reference)
        if error > bound:
            failed.append((k, value, reference, bound))
        elif bound > 0:
            worst = max(worst, error / bound)
        nonzero += reference != 0.0
    if failed:
        for case in failed[:10]:
            print('outside the bound:', case)
        raise SystemExit(f'{len(failed)} of {len(expected)} components outside the bound')
    assert nonzero > n, 'the traversal contributes nothing'
    print(f'{len(segments)} segments, {tuples} tuples, {len(expected)} components '
          f'within the bound, largest error {worst:.2g} of it')


if __name__ == '__main__':
    if sys.argv[1] == 'emit':
        periodic = sys.argv[-1] == '--periodic'
        args = sys.argv[2:-1] if periodic else sys.argv[2:]
        top, gro, program, kinds, out = args
        emit(System(top, gro, kinds.split(',')), program, out, read_box(gro), periodic)
    else:
        periodic = sys.argv[-1] == '--periodic'
        top, gro, kinds, values = (sys.argv[2:-1] if periodic else sys.argv[2:])
        system = System(top, gro, kinds.split(','))
        if periodic:
            # The definition forms displacements from coordinates in the
            # frames of the groups, at most |x| + 3 cells from the origin:
            # the rounding of the displacements scales with that.
            system.scale = max(system.scale, 4 * max(read_box(gro)))
        check(system, values)
