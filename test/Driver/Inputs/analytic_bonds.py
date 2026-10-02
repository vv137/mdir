"""Run the actual builder's one-bond kernel on difficult inputs.

Extract only the first position gather body, keeping its image handling,
mass weighting, branch and Newton fallback. Check geometric invariants
independently of the quadratic formula. No NumPy dependency.
"""

import math
import random
import re
import sys
import struct


def cases():
    rng = random.Random(137)
    result = []
    # Include repartitioned and equal masses, a bond already satisfied,
    # tiny corrections, contraction, extension, and the backward-root
    # case that must use the Newton fallback.
    for masses in [(12.0, 1.0), (9.0, 4.0), (1.0, 1.0)]:
        for scale in [1.0, 1.0 + 1e-10, 0.8, 1.2, -1.001]:
            result.append(([1.09 * scale, 0., 0.], [1.09, 0., 0.], masses, 1.09, 0.))
    # Rotate the predictor, including a small positive discriminant;
    # test periodic image shifts independently of coordinate origins.
    for i in range(50):
        direction = [rng.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(x * x for x in direction))
        old = [1.09 * x / norm for x in direction]
        new = [x + rng.uniform(-0.03, 0.03) for x in old]
        result.append((new, old, (12., 1.), 1.09, 30. if i % 2 else 0.))
    result.append(([0.1, 0.999, 0.], [1., 0., 0.], (9., 4.), 1., 30.))
    result.append(([1.02, 0.03, 0.02], [1., 0., 0.], (12., 1.), 1., [-10., 28.28, 0.]))
    return result


def emit(path):
    text = open(path).read().split('= md.gather_tuples %r_shake1,', 1)[1]
    args, body = re.search(r'\^bb0\((.*?)\):\n(.*?)\n\s*} : !rel_shake1', text, re.S).groups()
    assert '%vs_analytic' in body
    body = body.replace('md.yield', 'return')
    print('module {')
    print('func.func private @printI64(i64)')
    print('func.func private @printNewline()')
    print(f'func.func @project({args}) -> (vector<3xf64>, vector<3xf64>) {{\n{body}\n}}')
    print('func.func @main() {')
    for i, (new, old, masses, length, shift) in enumerate(cases()):
        # The raw old and new second atom lie one lattice vector away;
        # the supplied displacement is already minimum-image.
        origin = [17., -4., 6.]
        shift = shift if isinstance(shift, list) else [shift, 0., 0.]
        vectors = [new, origin, [origin[j] + old[j] + shift[j] for j in range(3)],
                   origin, [origin[j] + new[j] + shift[j] for j in range(3)]]
        names = []
        types = ['vector<3xf64>'] * 5 + ['f64'] * 3
        for j, value in enumerate(vectors + list(masses) + [length]):
            name = f'%c{i}_{j}'
            literal = ('dense<[' + ', '.join(f'{x:.17e}' for x in value) + ']>'
                       if isinstance(value, list) else f'{value:.17e}')
            print(f'{name} = arith.constant {literal} : {types[j]}')
            names.append(name)
        print(f'%a{i}, %b{i} = func.call @project({", ".join(names)}) : ({", ".join(types)}) -> (vector<3xf64>, vector<3xf64>)')
        for atom in ['a', 'b']:
            for axis in range(3):
                name = f'%v{i}_{atom}{axis}'
                print(f'{name} = vector.extract %{atom}{i}[{axis}] : f64 from vector<3xf64>')
                print(f'{name}_bits = arith.bitcast {name} : f64 to i64')
                print(f'func.call @printI64({name}_bits) : (i64) -> ()')
                print('func.call @printNewline() : () -> ()')
    print('return\n}\n}')


def check(path):
    values = [struct.unpack('d', struct.pack('q', int(v)))[0]
              for v in open(path).read().split()]
    assert len(values) == 6 * len(cases()), len(values)
    for i, (new, old, masses, length, _) in enumerate(cases()):
        da, db = values[6*i:6*i+3], values[6*i+3:6*i+6]
        assert all(math.isfinite(x) for x in da + db)
        corrected = [new[j] + db[j] - da[j] for j in range(3)]
        assert abs(sum(x*x for x in corrected) - length**2) < 2e-12, (i, corrected)
        assert max(abs(masses[0]*da[j] + masses[1]*db[j]) for j in range(3)) < 2e-12
        # Each displacement stays parallel to the OLD bond.
        for j in range(3):
            k = (j + 1) % 3
            assert abs(da[j]*old[k] - da[k]*old[j]) < 2e-12
        assert sum(x*y for x, y in zip(corrected, old)) * sum(x*y for x, y in zip(new, old)) > 0
    print('analytic bond invariants pass')


if __name__ == '__main__':
    {'emit': emit, 'check': check}[sys.argv[1]](sys.argv[2])
