"""Build a probe with independent f64 all-images references, before execution.

The energy is 1/2 r^2 inside the 1 nm site cutoff, zero outside. Its closed
forms are F_Ha=-d, F_Hb=d and W=-d outer d. All other atom forces vanish.
Bounds, declared before the first execution: 2e-13 * max(1, abs(reference))
for every energy, force and virial component, per image and in total.
With --site-alpha the interacting points are formed from actual world
positions; the constant-transpose coefficients are (1-2*alpha, alpha,
alpha) for each molecule, with opposite force signs for its partner.
"""
import itertools
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('source', type=pathlib.Path)
parser.add_argument('destination', type=pathlib.Path)
parser.add_argument('--site-alpha', type=float)
args = parser.parse_args()
source, destination = args.source, args.destination
text = source.read_text()
vec = 'vector<3xf64>'
tensor = 'vector<9xf64>'
argument_types = ', '.join([vec] * 5)
force_types = ', '.join([vec] * 6)
lines = [text, 'func.func private @printF64(f64)',
         'func.func private @printNewline()',
         f'func.func private @window_energy({argument_types}) -> f64',
         f'func.func private @window_forces({argument_types}) -> ({force_types})',
         f'func.func private @window_virial({argument_types}) -> {tensor}',
         'func.func @main() {']
counter = itertools.count()


def emit(expression):
    name = f'%v{next(counter)}'
    lines.append(f'  {name} = {expression}')
    return name


def scalar(value):
    return emit(f'arith.constant {float(value):.17e} : f64')


def vector(values):
    data = ', '.join(f'{float(v):.17e}' for v in values)
    return emit(f'arith.constant dense<[{data}]> : vector<{len(values)}xf64>')


good = emit('arith.constant true')


def check(actual, reference):
    global good
    ref = scalar(reference)
    difference = emit(f'arith.subf {actual}, {ref} : f64')
    absolute = emit(f'math.absf {difference} : f64')
    bound = scalar(2e-13 * max(1.0, abs(reference)))
    okay = emit(f'arith.cmpf ole, {absolute}, {bound} : f64')
    good = emit(f'arith.andi {good}, {okay} : i1')
    bad = emit(f'arith.cmpf ogt, {absolute}, {bound} : f64')
    lines.append(f'  scf.if {bad} {{')
    for value in (actual, ref, absolute, bound):
        lines.append(f'    func.call @printF64({value}) : (f64) -> ()')
        lines.append('    func.call @printNewline() : () -> ()')
    lines.append('  }')


# Real positions, not anchor-relative coordinates, define the oracle.
oa, ob = (1.1, 0.0, 0.0), (0.0, 0.0, 0.0)
ha, hb = (1.2, 0.0, 0.0), (-0.1, 0.0, 0.0)
weights = (0.0, 1.0, 0.0)
if args.site_alpha is not None:
    alpha = args.site_alpha
    ha2, hb2 = (1.072, 0.096, 0.0), (0.028, 0.096, 0.0)
    ha = tuple(oa[k] + alpha * ((ha[k] - oa[k]) + (ha2[k] - oa[k]))
               for k in range(3))
    hb = tuple(ob[k] + alpha * ((hb[k] - ob[k]) + (hb2[k] - ob[k]))
               for k in range(3))
    weights = (1.0 - 2.0 * alpha, alpha, alpha)
# Both second hydrogens are 0.1 nm from O and 0.16 nm from H1.
a1 = vector((0.1, 0.0, 0.0))
a2 = vector((-0.028, 0.096, 0.0))
b1 = vector((-0.1, 0.0, 0.0))
b2 = vector((0.028, 0.096, 0.0))
total = [scalar(0.0) for _ in range(28)]
reference_total = [0.0] * 28
inside_images = []
for image in itertools.product(range(-1, 2), repeat=3):
    shift = tuple(2.2 * n for n in image)
    # Independent brute force uses actual site positions and all 27 images.
    d = tuple(ha[k] - (hb[k] + shift[k]) for k in range(3))
    inside = sum(v * v for v in d) < 1.0
    reference = [0.0] * 28
    if inside:
        inside_images.append(image)
        reference[0] = 0.5 * sum(v * v for v in d)
        for atom, weight in enumerate(weights):
            for k in range(3):
                reference[1 + 3 * atom + k] = -weight * d[k]
                reference[1 + 3 * (atom + 3) + k] = weight * d[k]
        reference[19:] = [-d[a] * d[b] for a in range(3) for b in range(3)]
    base = vector(tuple(oa[k] - (ob[k] + shift[k]) for k in range(3)))
    arguments = ', '.join((base, a1, a2, b1, b2))
    energy = emit(f'func.call @window_energy({arguments}) : ({argument_types}) -> f64')
    force_prefix = f'%f{next(counter)}'
    lines.append(f'  {force_prefix}:6 = func.call @window_forces({arguments}) : '
                 f'({argument_types}) -> ({force_types})')
    virial = emit(f'func.call @window_virial({arguments}) : ({argument_types}) -> {tensor}')
    actual = [energy]
    for atom in range(6):
        for axis in range(3):
            actual.append(emit(f'vector.extract {force_prefix}#{atom}[{axis}] : f64 from {vec}'))
    for index in range(9):
        actual.append(emit(f'vector.extract {virial}[{index}] : f64 from {tensor}'))
    for index, (value, ref) in enumerate(zip(actual, reference)):
        check(value, ref)
        total[index] = emit(f'arith.addf {total[index]}, {value} : f64')
        reference_total[index] += ref
assert inside_images == [(1, 0, 0)], inside_images
for value, reference in zip(total, reference_total):
    check(value, reference)
one, zero = scalar(1.0), scalar(0.0)
flag = emit(f'arith.select {good}, {one}, {zero} : f64')
lines.extend([f'  func.call @printF64({flag}) : (f64) -> ()',
              '  func.call @printNewline() : () -> ()',
              f'  cf.assert {good}, "template-window numerical mismatch"',
              '  return', '}'])
destination.write_text('\n'.join(lines) + '\n')
