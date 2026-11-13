"""An independent f64 oracle for the segments of two kinds (D[kind-pairs]).

    kind_pairs_oracle.py emit  SYSTEM.top SYSTEM.gro WINDOWS.mlir KINDS OUT.mlir
    kind_pairs_oracle.py check SYSTEM.top SYSTEM.gro KINDS VALUES.txt

KINDS names the molecule types that are taken, kinds 0 and 1, in the order
of the generator (`mdir check` numbers them by count); every other
particle is the rest, kind 2. `emit` writes a program that calls the
outlined windows (`--test-md-segment-windows`) of every segment on every
tuple of the system and prints each energy, force and virial component as
the bits of an f64; `check` compares them with this oracle.

The oracle reads the topology and the coordinates itself. It places a site
by its own linear placement, x_s = x_1 + a (x_2 - x_1) + b (x_3 - x_1), with
a and b of [ virtual_sites3 ], and transposes its force with the constant
weights (1 - a - b, a, b). The pair energy is
4 eps ((sigma/r)^12 - (sigma/r)^6) + k q_i q_j / r inside the cutoff, with
the combination rule of Lorentz and Berthelot, and k = e^2 N_A / (4 pi eps_0)
of CODATA 2022
(scripts/mdir_physical_constants.py). The force on site i is -dU/dd, d = x_i - x_j, and the strain
of a pair d (x) F_i.

The bound of each component, declared before the first run, is the sum over
the tuple's site pairs inside the cutoff of
    energy:  32 eps |u| + |u'| dx
    force:   (32 eps |u'| + (|u''| + |u'| / r) dx + 4 eps |u'|) * |w|
    strain:  r * (force bound) + |u'| dx + 32 eps r |u'|
where eps is the unit roundoff of f64 (2^-53), |u|, |u'| and |u''| are the
sums of the magnitudes of the terms of the energy and of its derivatives in
r, |w| the magnitude of the weight of the frame atom, and
dx = 8 eps L, L the largest coordinate, bounds the rounding of the
displacements: the window forms (x_a - x_b) + (x_ap - x_a) - (x_bq - x_b)
from differences that this script rounds, where the oracle subtracts the
positions once. The term 4 eps |u'| covers weights that differ from the
file's by an ulp (D403 forms 1 - a - b). No site pair is within 1e-9 nm
of the cutoff, so both sides count the same pairs.
"""
import itertools
import math
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import (AVOGADRO, ELEMENTARY_CHARGE,  # noqa: E402
                                     VACUUM_PERMITTIVITY)

CUTOFF = 0.9
EPS = 2.0 ** -53
# kJ mol^-1 nm e^-2, from the defining constants.
COULOMB = (ELEMENTARY_CHARGE ** 2 * AVOGADRO
           / (4 * math.pi * VACUUM_PERMITTIVITY) * 1e9 / 1e3)


def read_top(path):
    types, molecules, order = {}, {}, []
    section, current = None, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.strip('[] ')
            continue
        fields = line.split()
        if section == 'atomtypes':
            types[fields[0]] = (len(types), float(fields[5]), float(fields[6]))
        elif section == 'moleculetype':
            current = {'atoms': [], 'site': None}
            molecules[fields[0]] = current
        elif section == 'atoms':
            current['atoms'].append((fields[1], float(fields[6]), float(fields[7])))
        elif section == 'virtual_sites3':
            current['site'] = (int(fields[0]) - 1, float(fields[5]), float(fields[6]))
        elif section == 'molecules':
            order.append((fields[0], int(fields[1])))
    return types, molecules, order


def read_gro(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[1])
    return [tuple(float(row[20 + 8 * k:28 + 8 * k]) for k in range(3))
            for row in lines[2:2 + count]]


class System:
    def __init__(self, top, gro, kinds):
        self.types, self.moltypes, order = read_top(top)
        self.positions = read_gro(gro)
        self.taken = kinds
        self.molecules = {name: [] for name in kinds}
        self.rest = []  # (particle, molecule number)
        first, number = 0, 0
        for name, count in order:
            size = len(self.moltypes[name]['atoms'])
            for _ in range(count):
                particles = list(range(first, first + size))
                if name in kinds:
                    self.molecules[name].append(particles)
                else:
                    self.rest += [(p, number) for p in particles]
                first += size
                number += 1
        self.type_of, self.charge_of = [], []
        for name, count in order:
            for _ in range(count):
                for type, charge, _ in self.moltypes[name]['atoms']:
                    self.type_of.append(self.types[type][0])
                    self.charge_of.append(charge)
        self.sigma = {v[0]: v[1] for v in self.types.values()}
        self.epsilon = {v[0]: v[2] for v in self.types.values()}
        self.scale = max(abs(c) for p in self.positions for c in p)

    def frame(self, name):
        atoms = self.moltypes[name]['atoms']
        return [i for i, (_, _, mass) in enumerate(atoms) if mass > 0.0]

    def side(self, f, unit):
        """Places of one molecule: (particle, position, [(member, weight)])."""
        if f == len(self.taken):
            p = unit[0]
            return [(p, self.positions[p], [(0, 1.0)])], [p]
        name = self.taken[f]
        frame = self.frame(name)
        site = self.moltypes[name]['site']
        places = []
        for place, p in enumerate(unit):
            if place in frame:
                places.append((p, self.positions[p], [(frame.index(place), 1.0)]))
            else:
                index, a, b = site
                assert index == place
                x = [self.positions[unit[i]] for i in frame]
                position = tuple(x[0][k] + a * (x[1][k] - x[0][k]) + b * (x[2][k] - x[0][k])
                                 for k in range(3))
                places.append((p, position, [(0, 1.0 - a - b), (1, a), (2, b)]))
        return places, [unit[i] for i in frame]

    def segments(self):
        """Every tuple of every segment (f, g), g <= f, in a fixed order."""
        K = len(self.taken)
        units = [[m for m in self.molecules[name]] for name in self.taken]
        units.append([[p] for p, _ in self.rest])
        home = {p: m for p, m in self.rest}
        for f in range(K + 1):
            for g in range(f + 1):
                for i, a in enumerate(units[f]):
                    for j, b in enumerate(units[g]):
                        if f == g and j <= i:
                            continue
                        if f == g == K and home[a[0]] == home[b[0]]:
                            continue  # an excluded pair of the rest
                        yield f, g, a, b

    def pair(self, ti, tj, qi, qj, r):
        sigma = 0.5 * (self.sigma[ti] + self.sigma[tj])
        epsilon = math.sqrt(self.epsilon[ti] * self.epsilon[tj])
        s6 = (sigma / r) ** 6
        lj12, lj6 = 4 * epsilon * s6 * s6, 4 * epsilon * s6
        coulomb = COULOMB * qi * qj / r
        u = lj12 - lj6 + coulomb
        du = (-12 * lj12 + 6 * lj6 - coulomb) / r
        mag = abs(lj12) + abs(lj6) + abs(coulomb)
        dmag = (12 * abs(lj12) + 6 * abs(lj6) + abs(coulomb)) / r
        ddmag = (156 * abs(lj12) + 42 * abs(lj6) + 2 * abs(coulomb)) / (r * r)
        return u, du, mag, dmag, ddmag

    def reference(self, f, g, a, b):
        """Energy, member forces and strain, with their bounds."""
        A, membersA = self.side(f, a)
        B, membersB = self.side(g, b)
        arity = len(membersA) + len(membersB)
        dx = 8 * EPS * self.scale
        energy = bound = 0.0
        forces = [[0.0] * 3 for _ in range(arity)]
        fbound = [[0.0] * 3 for _ in range(arity)]
        strain, sbound = [0.0] * 9, [0.0] * 9
        for (pi, xi, wi), (pj, xj, wj) in itertools.product(A, B):
            d = [xi[k] - xj[k] for k in range(3)]
            r = math.sqrt(sum(v * v for v in d))
            assert abs(r - CUTOFF) > 1e-9, 'a site pair at the cutoff'
            if r >= CUTOFF:
                continue
            u, du, mag, dmag, ddmag = self.pair(self.type_of[pi], self.type_of[pj],
                                                self.charge_of[pi], self.charge_of[pj], r)
            energy += u
            bound += 32 * EPS * mag + dmag * dx
            force = [-du * v / r for v in d]
            fb = 32 * EPS * dmag + (ddmag + dmag / r) * dx + 4 * EPS * dmag
            for members, weights, sign in ((0, wi, 1.0), (len(membersA), wj, -1.0)):
                for member, w in weights:
                    for k in range(3):
                        forces[members + member][k] += sign * w * force[k]
                        fbound[members + member][k] += abs(w) * fb
            for i in range(3):
                for j in range(3):
                    strain[3 * i + j] += d[i] * force[j]
                    sbound[3 * i + j] += r * fb + dmag * dx + 32 * EPS * r * dmag
        return (energy, bound), (forces, fbound), (strain, sbound), membersA + membersB

    def arguments(self, f, g, a, b):
        """The window's arguments: anchor displacement, frame offsets, then
        types and charges of the members where the rest takes part."""
        _, ma = self.side(f, a)
        _, mb = self.side(g, b)
        x = self.positions
        values = [[x[ma[0]][k] - x[mb[0]][k] for k in range(3)]]
        values += [[x[p][k] - x[ma[0]][k] for k in range(3)] for p in ma[1:]]
        values += [[x[p][k] - x[mb[0]][k] for k in range(3)] for p in mb[1:]]
        members = ma + mb
        gathered = f == len(self.taken)
        return values, ([self.type_of[p] for p in members] if gathered else []), \
            ([self.charge_of[p] for p in members] if gathered else [])


def emit(system, windows, out):
    text = Path(windows).read_text()
    # The tables of the pair term, sigma and epsilon of each pair of types,
    # are the program's arguments 4 and 5 (lj_sigma and lj_epsilon): the
    # outlined lookups call these functions in their place.
    types = sorted(system.sigma)
    tables = [('window_table4', lambda a, b: 0.5 * (system.sigma[a] + system.sigma[b])),
              ('window_table5', lambda a, b: math.sqrt(system.epsilon[a] * system.epsilon[b]))]
    definitions = []
    for name, table in tables:
        declaration = f'  func.func private @{name}(i32, i32) -> f64\n'
        assert declaration in text, name
        text = text.replace(declaration, '')
        size = len(types) ** 2
        data = ', '.join(f'{table(a, b):.17e}' for a in types for b in types)
        definitions += [f'func.func @{name}(%i: i32, %j: i32) -> f64 {{',
                        f'  %n = arith.constant {len(types)} : i32',
                        '  %row = arith.muli %i, %n : i32',
                        '  %at = arith.addi %row, %j : i32',
                        '  %index = arith.index_cast %at : i32 to index',
                        f'  %table = arith.constant dense<[{data}]> : vector<{size}xf64>',
                        f'  %value = vector.extract %table[%index] : f64 from vector<{size}xf64>',
                        '  return %value : f64', '}']
    # The module's closing brace: the probe's functions go inside it.
    text = text.rstrip()
    assert text.endswith('}')
    lines = [text[:-1], 'func.func private @printI64(i64)',
             'func.func private @printNewline()'] + definitions
    lines.append('func.func @main() {')
    counter = itertools.count()

    def value(expression):
        name = f'%v{next(counter)}'
        lines.append(f'  {name} = {expression}')
        return name

    def show(scalar):
        bits = value(f'arith.bitcast {scalar} : f64 to i64')
        lines.append(f'  func.call @printI64({bits}) : (i64) -> ()')
        lines.append('  func.call @printNewline() : () -> ()')

    for f, g, a, b in system.segments():
        vectors, types_, charges = system.arguments(f, g, a, b)
        operands, signature = [], []
        for v in vectors:
            data = ', '.join(f'{c:.17e}' for c in v)
            operands.append(value(f'arith.constant dense<[{data}]> : vector<3xf64>'))
            signature.append('vector<3xf64>')
        for t in types_:
            operands.append(value(f'arith.constant {t} : i32'))
            signature.append('i32')
        for q in charges:
            operands.append(value(f'arith.constant {q:.17e} : f64'))
            signature.append('f64')
        args, sig = ', '.join(operands), ', '.join(signature)
        prefix = f'@window_{f}_{g}'
        header = next(l for l in text.splitlines() if f'func.func {prefix}_energy(' in l)
        pairs = header.split('->')[1].count('f64')
        arity = len(vectors) + 1
        for what, element, per in (('energy', 'f64', 1), ('forces', 'vector<3xf64>', arity),
                                   ('virial', 'vector<9xf64>', 1)):
            results = f'%r{next(counter)}'
            count = pairs * per
            types_out = ', '.join([element] * count)
            lines.append(f'  {results}:{count} = func.call {prefix}_{what}({args}) : '
                         f'({sig}) -> ({types_out})' if count > 1 else
                         f'  {results} = func.call {prefix}_{what}({args}) : ({sig}) -> {types_out}')
            names = [f'{results}#{i}' for i in range(count)] if count > 1 else [results]
            if element == 'f64':
                total = value('arith.constant 0.0 : f64')
                for name in names:
                    total = value(f'arith.addf {total}, {name} : f64')
                show(total)
            else:
                width = 3 if element == 'vector<3xf64>' else 9
                for slot in range(per):
                    for k in range(width):
                        total = value('arith.constant 0.0 : f64')
                        for name in names[slot::per]:
                            part = value(f'vector.extract {name}[{k}] : f64 from {element}')
                            total = value(f'arith.addf {total}, {part} : f64')
                        show(total)
    lines += ['  return', '}', '}']
    Path(out).write_text('\n'.join(lines) + '\n')


def check(system, values):
    bits = [int(line) for line in Path(values).read_text().split()]
    actual = iter(struct.unpack('<d', struct.pack('<q', b))[0] for b in bits)
    worst, components, tuples, failed = 0.0, 0, 0, []
    segments = set()
    for f, g, a, b in system.segments():
        (energy, ebound), (forces, fbound), (strain, sbound), _ = system.reference(f, g, a, b)
        expected = [(energy, ebound)]
        expected += [(v, e) for row, brow in zip(forces, fbound) for v, e in zip(row, brow)]
        expected += list(zip(strain, sbound))
        for reference, bound in expected:
            got = next(actual)
            error = abs(got - reference)
            components += 1
            if error > bound:
                failed.append((f, g, a, b, got, reference, bound))
            elif bound > 0:
                worst = max(worst, error / bound)
        tuples += 1
        segments.add((f, g))
    assert next(actual, None) is None, 'more values than the oracle has'
    if failed:
        for case in failed[:10]:
            print('outside the bound:', case)
        raise SystemExit(f'{len(failed)} of {components} components outside the bound')
    print(f'{len(segments)} segments, {tuples} tuples, {components} components '
          f'within the bound, largest error {worst:.2g} of it')


if __name__ == '__main__':
    if sys.argv[1] == 'emit':
        _, _, top, gro, windows, kinds, out = sys.argv
        emit(System(top, gro, kinds.split(',')), windows, out)
    else:
        _, _, top, gro, kinds, values = sys.argv
        check(System(top, gro, kinds.split(',')), values)
