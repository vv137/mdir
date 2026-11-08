"""Check compiled stencil traversal and geometry against a dense image oracle."""
import argparse
import math
import sys

import numpy as np

GRID = (4, 5, 6)
ATOL = 2e-11
RTOL = 2e-12
FD_ATOL = 5e-8


def data():
    for skew in (False, True):
        h = np.diag([2.3, 3.1, 4.2])
        if skew:
            h[1, 0], h[2, 0], h[2, 1] = 0.31, -0.17, 0.23
        s = np.asarray([[-0.002, 0.27, 1.0001], [0.13, 0.9999, -0.21],
                        [1.0002, 0.41, 0.62]])
        x = s @ h
        q = np.asarray([0.7, 0.0, -0.4])
        dx = np.asarray([[0.13, -0.17, 0.11], [-0.09, 0.07, 0.03],
                         [0.04, 0.02, -0.12]])
        dh = np.asarray([[0.11, -0.03, 0.07], [0.04, -0.09, -0.02],
                         [-0.06, 0.08, 0.05]])
        grid = np.sin(np.arange(np.prod(GRID))*0.19).reshape(GRID)
        seed = np.asarray([0.2, -0.3, 0.4])
        yield x, h, q, dx, dh, grid, seed


def cardinal(distance, derivative=False):
    # Independent truncated-power definition, not the compiled recurrence.
    if distance <= 0.0 or distance >= 4.0:
        return 0.0
    power = 2 if derivative else 3
    return sum((-1)**j*math.comb(4, j)*max(distance-j, 0.0)**power
               for j in range(5)) / math.factorial(power)


def assignment(x, h):
    inverse = np.linalg.inv(h)
    s = x @ inverse
    scaled = (s-np.floor(s))*GRID
    a = np.empty((len(x), np.prod(GRID)))
    gx = np.empty(a.shape+(3,))
    gh = np.empty(a.shape+(3, 3))
    for i in range(len(x)):
        for flat, k in enumerate(np.ndindex(GRID)):
            weights, slopes = [], []
            for axis in range(3):
                distances = [scaled[i, axis]-k[axis]+image*GRID[axis]
                             for image in range(-2, 3)]
                weights.append(sum(cardinal(d) for d in distances))
                slopes.append(sum(cardinal(d, True) for d in distances))
            a[i, flat] = math.prod(weights)
            gu = np.asarray([slopes[axis]*math.prod(weights[j] for j in range(3) if j != axis)
                             for axis in range(3)])*GRID
            gx[i, flat] = gu @ inverse.T
            gh[i, flat] = -np.outer(s[i], gx[i, flat])
    return a, gx, gh


def expected():
    rows = []
    worst_fd = 0.0
    for x, h, q, dx, dh, grid, seed in data():
        a, gx, gh = assignment(x, h)
        g = grid.ravel()
        ax = np.einsum('ikc,ic->ik', gx, dx)
        ah = np.einsum('ikbc,bc->ik', gh, dh)
        rho = q @ a
        p = a @ g
        sjx, sjh = q @ ax, q @ ah
        svx = q[:, None]*np.einsum('ikc,k->ic', gx, g)
        svh = np.einsum('i,ikbc,k->bc', q, gh, g)
        gjx, gjh = ax @ g, ah @ g
        gvx = seed[:, None]*np.einsum('ikc,k->ic', gx, g)
        gvh = np.einsum('i,ikbc,k->bc', seed, gh, g)
        # The zero-amplitude particle retains its nonzero amplitude cotangent.
        rows.extend([rho, p, sjx, sjh, svx, svh, gjx, gjh, gvx, gvh,
                     seed @ a, a @ g])
        step = 2e-6
        px, _, _ = assignment(x+step*dx, h)
        mx, _, _ = assignment(x-step*dx, h)
        ph, _, _ = assignment(x, h+step*dh)
        mh, _, _ = assignment(x, h-step*dh)
        for value, reference in ((sjx, q @ ((px-mx)/(2*step))),
                                 (sjh, q @ ((ph-mh)/(2*step))),
                                 (gjx, ((px-mx)/(2*step)) @ g),
                                 (gjh, ((ph-mh)/(2*step)) @ g)):
            worst_fd = max(worst_fd, float(np.max(np.abs(value-reference))))
        # Forward/reverse Euclidean identities for both continuous operands.
        for left, right in ((np.dot(sjx, g), np.sum(svx*dx)),
                            (np.dot(sjh, g), np.sum(svh*dh)),
                            (np.dot(gjx, seed), np.sum(gvx*dx)),
                            (np.dot(gjh, seed), np.sum(gvh*dh))):
            assert abs(left-right) <= ATOL+RTOL*abs(left), (left, right)
    return np.concatenate([np.asarray(row).ravel() for row in rows]), worst_fd


def emit():
    lines = ['md.particle_set @atoms', 'func.func private @mdrtPrintFFTValue(f64)',
             'func.func @main() {']
    serial = 0

    def name():
        nonlocal serial
        serial += 1
        return f'%v{serial}'

    def constant(value, ty='f64'):
        v = name()
        text = str(value) if ty == 'index' else f'{float(value):.17e}'
        lines.append(f'  {v} = arith.constant {text} : {ty}')
        return v

    def field(array):
        shape = array.shape
        components = shape[1] if len(shape) == 2 else 1
        dims = 'x'.join(str(n) for n in shape)
        static = f'memref<{dims}xf64>'
        dynamic = 'memref<?' + (f'x{components}' if components != 1 else '') + 'xf64>'
        field_type = '!md.field<@atoms, ' + (f'{components} x ' if components != 1 else '') + 'f64>'
        buffer = name()
        lines.append(f'  {buffer} = memref.alloc() : {static}')
        for xyz in np.ndindex(shape):
            c = constant(array[xyz])
            indices = ', '.join(constant(i, 'index') for i in xyz)
            lines.append(f'  memref.store {c}, {buffer}[{indices}] : {static}')
        cast, result = name(), name()
        lines.append(f'  {cast} = memref.cast {buffer} : {static} to {dynamic}')
        lines.append(f'  {result} = mdrt.from_buffer {cast} : {dynamic} to {field_type}')
        return result, field_type

    def output_tensor(value):
        for xyz in np.ndindex(GRID):
            indices = ', '.join(constant(i, 'index') for i in xyz)
            v = name()
            lines.append(f'  {v} = tensor.extract {value}[{indices}] : tensor<4x5x6xf64>')
            lines.append(f'  func.call @mdrtPrintFFTValue({v}) : (f64) -> ()')

    def output_field(value, components):
        field_type = '!md.field<@atoms, ' + (f'{components} x ' if components != 1 else '') + 'f64>'
        buffer_type = 'memref<?' + (f'x{components}' if components != 1 else '') + 'xf64>'
        buffer = name()
        lines.append(f'  {buffer} = mdrt.to_buffer {value} : {field_type} to {buffer_type}')
        for i in range(3):
            for c in range(components):
                indices = [constant(i, 'index')]
                if components != 1:
                    indices.append(constant(c, 'index'))
                v = name()
                lines.append(f'  {v} = memref.load {buffer}[{", ".join(indices)}] : {buffer_type}')
                lines.append(f'  func.call @mdrtPrintFFTValue({v}) : (f64) -> ()')

    def output_vector(value):
        for c in range(9):
            v = name()
            lines.append(f'  {v} = vector.extract {value}[{c}] : f64 from vector<9xf64>')
            lines.append(f'  func.call @mdrtPrintFFTValue({v}) : (f64) -> ()')

    stencil_type = '!md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>'
    grid_type = 'tensor<4x5x6xf64>'
    for x, h, q, dx, dh, grid, seed in data():
        positions, position_type = field(x)
        amplitude, amplitude_type = field(q)
        direction, _ = field(dx)
        cotangent, _ = field(seed)
        values = [constant(h[i, j]) for i, j in ((0,0), (1,1), (2,2), (1,0), (2,0), (2,1))]
        cell = name()
        lines.append(f'  {cell} = md.triclinic_cell {", ".join(values)}')
        cell_seed = name()
        entries = ', '.join(f'{v:.17e}' for v in dh.ravel())
        lines.append(f'  {cell_seed} = arith.constant dense<[{entries}]> : vector<9xf64>')
        values = [constant(v) for v in grid.ravel()]
        grid_value = name()
        lines.append(f'  {grid_value} = tensor.from_elements {", ".join(values)} : {grid_type}')
        relation = name()
        lines.append(f'  {relation} = md.grid_stencil {positions}, {cell} : {position_type} -> {stencil_type}')
        rho, gathered = name(), name()
        for result, operation, operand, ty, result_type in (
                (rho, 'scatter', amplitude, amplitude_type, grid_type),
                (gathered, 'gather', grid_value, grid_type, amplitude_type)):
            lines.extend([f'  {result} = md.{operation}_stencil {relation}, {positions}, {cell}, {operand} {{',
                          '  ^bb0(%distance: vector<3xf64>):',
                          '    %weight = md.bspline_window %distance order(4) : vector<3xf64>',
                          '    md.yield %weight : f64',
                          f'  }} : {stencil_type}, {position_type}, {ty} -> {result_type}'])
        output_tensor(rho)
        output_field(gathered, 1)
        requests = [(rho, 'jvp', 1, direction, position_type, grid_type, 'grid'),
                    (rho, 'jvp', 2, cell_seed, 'vector<9xf64>', grid_type, 'grid'),
                    (rho, 'vjp', 1, grid_value, grid_type, position_type, 'position'),
                    (rho, 'vjp', 2, grid_value, grid_type, 'vector<9xf64>', 'cell'),
                    (gathered, 'jvp', 1, direction, position_type, amplitude_type, 'particle'),
                    (gathered, 'jvp', 2, cell_seed, 'vector<9xf64>', amplitude_type, 'particle'),
                    (gathered, 'vjp', 1, cotangent, amplitude_type, position_type, 'position'),
                    (gathered, 'vjp', 2, cotangent, amplitude_type, 'vector<9xf64>', 'cell'),
                    (gathered, 'vjp', 3, cotangent, amplitude_type, grid_type, 'grid'),
                    (rho, 'vjp', 3, grid_value, grid_type, amplitude_type, 'particle')]
        for source, kind, operand, seed_value, seed_type, result_type, output_kind in requests:
            result = name()
            source_type = grid_type if source == rho else amplitude_type
            lines.append(f'  {result} = md.stencil_{kind} {source} wrt({operand}) {seed_value} : {source_type}, {seed_type} -> {result_type}')
            if output_kind == 'grid':
                output_tensor(result)
            elif output_kind == 'cell':
                output_vector(result)
            else:
                output_field(result, 3 if output_kind == 'position' else 1)
    lines.extend(['  return', '}'])
    print('\n'.join(lines))


def check():
    actual = np.asarray([float(word) for word in sys.stdin.read().split()])
    reference, fd_error = expected()
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    error = np.abs(actual-reference)
    assert np.all(error <= ATOL+RTOL*np.abs(reference)), (error.max(), np.argmax(error))
    assert fd_error <= FD_ATOL, fd_error
    print(f'stencil primal/JVP/VJP: {actual.size} checked; max absolute error {error.max():.9g}; FD error {fd_error:.9g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['emit', 'check'])
    args = parser.parse_args()
    emit() if args.command == 'emit' else check()
