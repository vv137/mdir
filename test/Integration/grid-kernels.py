"""Independent real/complex structured-grid checks; bounds precede execution."""
import argparse
import sys

import numpy as np

ATOL = 2e-12
RTOL = 2e-12


def cases():
    for length in (5, 6):
        shape = (2, 3, length)
        i = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        x = np.sin(i * 0.37) * 0.3
        dx = np.cos(i * 0.21) * 0.2
        bar = np.sin(i * 0.19 + 0.7) * 0.4
        yield shape, x, dx, bar


def expected():
    rows = []
    for shape, x, dx, bar in cases():
        a = 0.7
        rows.extend((a*x*x, 2*a*x*dx, 2*a*x*bar,
                     np.asarray(np.sum(x*x*bar))))
        z, dz, seed = (np.fft.rfftn(v) for v in (x, dx, bar))
        weights = np.full(z.shape, 2.0)
        weights[:, :, 0] = 1.0
        if shape[2] % 2 == 0:
            weights[:, :, -1] = 1.0
        rows.extend((a*z*z, 2*a*z*dz, 2*a*np.conj(z)*seed,
                     np.asarray(np.sum(weights*np.real(np.conj(seed)*z*z))),
                     np.asarray(0.5*np.sum(weights*np.abs(z)**2)),
                     np.asarray(np.sum(weights*np.real(np.conj(z)*dz))),
                     0.7*z))
    return rows


def emit():
    lines = ['func.func private @mdrtPrintFFTValue(f64)', 'func.func @main() {']
    count = 0

    def name():
        nonlocal count
        count += 1
        return f'%v{count}'

    def number(value, ty='f64'):
        v = name()
        spelling = str(value) if ty == 'index' else f'{float(value):.17e}'
        lines.append(f'  {v} = arith.constant {spelling} : {ty}')
        return v

    def tensor(value):
        scalar = 'complex<f64>' if np.iscomplexobj(value) else 'f64'
        ty = 'tensor<' + 'x'.join(str(n) for n in value.shape) + 'x' + scalar + '>'
        elements = []
        for item in value.flat:
            if np.iscomplexobj(value):
                real, imag = number(item.real), number(item.imag)
                v = name()
                lines.append(f'  {v} = complex.create {real}, {imag} : {scalar}')
            else:
                v = number(item)
            elements.append(v)
        v = name()
        lines.append(f'  {v} = tensor.from_elements {", ".join(elements)} : {ty}')
        return v, ty

    def output(v, ty, shape, complex_value=False):
        if ty == 'f64':
            lines.append(f'  func.call @mdrtPrintFFTValue({v}) : (f64) -> ()')
            return
        for xyz in np.ndindex(shape):
            indices = ', '.join(number(i, 'index') for i in xyz)
            scalar = name()
            lines.append(f'  {scalar} = tensor.extract {v}[{indices}] : {ty}')
            if complex_value:
                real, imag = name(), name()
                lines.append(f'  {real} = complex.re {scalar} : complex<f64>')
                lines.append(f'  {imag} = complex.im {scalar} : complex<f64>')
                for c in (real, imag):
                    lines.append(f'  func.call @mdrtPrintFFTValue({c}) : (f64) -> ()')
            else:
                lines.append(f'  func.call @mdrtPrintFFTValue({scalar}) : (f64) -> ()')

    for shape, x, dx, bar in cases():
        a = number(0.7)
        for is_complex in (False, True):
            arrays = tuple(np.fft.rfftn(v) for v in (x, dx, bar)) if is_complex else (x, dx, bar)
            (v, ty), (direction, _), (seed, _) = (tensor(value) for value in arrays)
            domain = str(list(shape))
            pairing = 'real_hermitian_half' if is_complex else 'euclidean'
            mapped, tangent, cotangent, uniform = (name() for _ in range(4))
            k = 'vector<2xf64>' if is_complex else 'f64'
            lines.append(f'  {mapped} = md.map_grid ({v}, {a}) grid({domain}) pairing("{pairing}") : ({ty}, f64) -> {ty} {{')
            lines.append(f'  ^bb0(%idx: vector<3xindex>, %x: {k}, %a: f64):')
            if is_complex:
                lines.extend(['    %r = vector.extract %x[0] : f64 from vector<2xf64>',
                              '    %i = vector.extract %x[1] : f64 from vector<2xf64>',
                              '    %rr = arith.mulf %r, %r : f64',
                              '    %ii = arith.mulf %i, %i : f64',
                              '    %real = arith.subf %rr, %ii : f64',
                              '    %ri = arith.mulf %r, %i : f64',
                              '    %imag = arith.addf %ri, %ri : f64',
                              '    %u = vector.from_elements %real, %imag : vector<2xf64>',
                              '    %av = vector.broadcast %a : f64 to vector<2xf64>',
                              '    %out = arith.mulf %av, %u : vector<2xf64>'])
            else:
                lines.extend(['    %u = arith.mulf %x, %x : f64',
                              '    %out = arith.mulf %a, %u : f64'])
            lines.extend([f'    md.yield %out : {k}', '  }',
                          f'  {tangent} = md.grid_jvp {mapped} wrt(0) {direction} : {ty}, {ty} -> {ty}',
                          f'  {cotangent} = md.grid_vjp {mapped} wrt(0) {seed} : {ty}, {ty} -> {ty}',
                          f'  {uniform} = md.grid_vjp {mapped} wrt(1) {seed} : {ty}, {ty} -> f64'])
            for value in (mapped, tangent, cotangent):
                output(value, ty, arrays[0].shape, is_complex)
            output(uniform, 'f64', ())
            if is_complex:
                energy, jvp, vjp = (name() for _ in range(3))
                lines.extend([f'  {energy} = md.sum_grid ({v}) grid({domain}) pairing("{pairing}") : ({ty}) -> f64 {{',
                              '  ^bb0(%idx: vector<3xindex>, %z: vector<2xf64>):',
                              '    %squared = arith.mulf %z, %z : vector<2xf64>',
                              '    %sum = vector.reduction <add>, %squared : vector<2xf64> into f64',
                              '    %half = arith.constant 0.5 : f64',
                              '    %value = arith.mulf %half, %sum : f64',
                              '    md.yield %value : f64', '  }',
                              f'  {jvp} = md.grid_jvp {energy} wrt(0) {direction} : f64, {ty} -> f64',
                              f'  {vjp} = md.grid_vjp {energy} wrt(0) {a} : f64, f64 -> {ty}'])
                output(energy, 'f64', ())
                output(jvp, 'f64', ())
                output(vjp, ty, arrays[0].shape, True)
    lines.extend(['  return', '}'])
    print('\n'.join(lines))


def check():
    actual = np.asarray([float(word) for word in sys.stdin.read().split()])
    reference = []
    for value in expected():
        if np.iscomplexobj(value):
            reference.extend(np.stack((value.real, value.imag), axis=-1).ravel())
        else:
            reference.extend(value.ravel())
    reference = np.asarray(reference)
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    error = np.abs(actual-reference)
    assert np.all(error <= ATOL + RTOL*np.abs(reference)), (error.max(), np.argmax(error))
    print(f'grid values/JVPs/VJPs: {actual.size} checked; max absolute error {error.max():.9g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['emit', 'check'])
    args = parser.parse_args()
    emit() if args.command == 'emit' else check()
