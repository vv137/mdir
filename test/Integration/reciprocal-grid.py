"""Small rectangular Gaussian chain against an independent direct DFT."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

spec = importlib.util.spec_from_file_location('stencil_reference', Path(__file__).with_name('stencil-values.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

# Fixed before the first chain comparison. Energies use the grid term only.
VALUE_ATOL = 2e-9
VALUE_RTOL = 2e-11
STRAIN_ATOL = 2e-6
BETA = 1.7
COULOMB = 138.935
GRID = reference.GRID


def inputs(tilted=False):
    frames = reference.data()
    x, h, q, *_ = next(frames)
    if tilted:
        x, h, q, *_ = next(frames)
    moduli = np.ones((3, max(GRID)))
    for axis, extent in enumerate(GRID):
        k = np.arange(extent)
        moduli[axis, :extent] = (2/3+np.cos(2*np.pi*k/extent)/3)**-2
    return x, h, q, moduli


def oracle(x, h, q, moduli):
    assignment, gradient, _ = reference.assignment(x, h)
    coordinates = np.asarray(list(np.ndindex(GRID)))
    modes = np.asarray(list(np.ndindex((GRID[0], GRID[1], GRID[2]//2+1))))
    # Direct complex DFT, independently formed without a library FFT.
    transform = np.exp(-2j*np.pi*(modes/GRID) @ coordinates.T)
    spectrum = transform @ (q @ assignment)
    signed = np.where(modes > np.asarray(GRID)//2, modes-GRID, modes)
    wave = signed @ np.linalg.inv(h).T
    squared = np.sum(wave*wave, axis=1)
    product = np.prod([moduli[a, modes[:, a]] for a in range(3)], axis=0)
    influence = np.zeros(len(modes))
    nonzero = squared > 0
    influence[nonzero] = (COULOMB/(np.pi*np.linalg.det(h))*product[nonzero]
                         * np.exp(-np.pi**2*squared[nonzero]/BETA**2)/squared[nonzero])
    weights = np.where((modes[:, 2] == 0) |
                       ((GRID[2] % 2 == 0) & (modes[:, 2] == GRID[2]//2)), 1., 2.)
    energy = 0.5*np.sum(weights*influence*np.abs(spectrum)**2)
    grid_potential = np.real(transform.conj().T @ (weights*influence*spectrum))
    potential = assignment @ grid_potential
    force = -q[:, None]*np.einsum('ikc,k->ic', gradient, grid_potential)
    return energy, force, potential


def expected(evaluator=False):
    x, h, q, moduli = inputs()
    energy, force, potential = oracle(x, h, q, moduli)
    strain = np.empty((3, 3))
    step = 2e-6
    for a in range(3):
        for b in range(3):
            epsilon = np.zeros((3, 3))
            epsilon[a, b] = step
            plus = np.eye(3)+epsilon
            minus = np.eye(3)-epsilon
            ep = oracle(x @ plus, h @ plus, q, moduli)[0]
            em = oracle(x @ minus, h @ minus, q, moduli)[0]
            strain[b, a] = -(ep-em)/(2*step)
    values = np.concatenate(([energy], force.ravel(), strain.ravel(), potential))
    return 2.5*values if evaluator else values


def emit(evaluator=False, mixed=False, force_only=False, listed=False, tilted=False):
    force_element = 'f32' if mixed else 'f64'
    x_type = '!md.field<@atoms, 3 x f64>'
    force_type = f'!md.field<@atoms, 3 x {force_element}>'
    q_type = '!md.field<@atoms, f64>'
    m_type = '!md.table<2, f64>'
    lines = ['md.particle_set @atoms',
             f'md.function @evaluate(%x: {x_type}, %q: {q_type}, %h: !md.cell, %m: {m_type}) -> (f64, {x_type}, vector<9xf64>, {q_type}) {{',
             f'  %u, %f, %w, %p = md.reciprocal %x, %q, %h, %m grid({list(GRID)}) order(4) beta({BETA}) coulomb({COULOMB}) : {x_type}, {q_type}, {m_type} -> f64, {x_type}, vector<9xf64>, {q_type}',
             f'  md.return %u, %f, %w, %p : f64, {x_type}, vector<9xf64>, {q_type}',
             '}', 'func.func private @mdrtPrintFFTValue(f64)',
             'func.func @main() attributes {llvm.emit_c_interface} {']
    if evaluator:
        lines[1:5] = [
            f'md.potential @energy(%x: {x_type}, %h: !md.cell, %q: {q_type}, %m: {m_type}) -> f64 {{',
            f'  %u, %f, %w = md.reciprocal %x, %q, %h, %m grid({list(GRID)}) order(4) beta({BETA}) coulomb({COULOMB}) : {x_type}, {q_type}, {m_type} -> f64, {x_type}, vector<9xf64>',
            '  %weight = arith.constant 2.5 : f64',
            '  %weighted = arith.mulf %weight, %u : f64',
            '  md.return %weighted : f64',
            '}',
            f'md.function @evaluate(%x: {x_type}, %q: {q_type}, %h: !md.cell, %m: {m_type}) -> (f64, {x_type}, vector<9xf64>, {q_type}) {{',
            f'  %u, %f, %w, %p = md.evaluate @energy(%x, %h, %q, %m) request [energy, forces, virial, derivative(2)] : ({x_type}, !md.cell, {q_type}, {m_type}) -> (f64, {x_type}, vector<9xf64>, {q_type})',
            f'  md.return %u, %f, %w, %p : f64, {x_type}, vector<9xf64>, {q_type}',
            '}',
        ]
    if mixed:
        del lines[1:5]
    serial = 0

    def name():
        nonlocal serial
        serial += 1
        return f'%v{serial}'

    def number(value, ty='f64'):
        v = name()
        text = str(value) if ty == 'index' else f'{float(value):.17e}'
        lines.append(f'  {v} = arith.constant {text} : {ty}')
        return v

    def buffer(array, logical):
        shape = array.shape
        static = 'memref<'+'x'.join(str(n) for n in shape)+'xf64>'
        if logical == 'table':
            dynamic, result_type = 'memref<?x?xf64>', '!md.table<2, f64>'
        elif len(shape) == 2:
            dynamic, result_type = 'memref<?x3xf64>', x_type
        else:
            dynamic, result_type = 'memref<?xf64>', '!md.field<@atoms, f64>'
        storage = name()
        lines.append(f'  {storage} = memref.alloc() : {static}')
        for xyz in np.ndindex(shape):
            scalar = number(array[xyz])
            indices = ', '.join(number(i, 'index') for i in xyz)
            lines.append(f'  memref.store {scalar}, {storage}[{indices}] : {static}')
        cast, value = name(), name()
        lines.extend([f'  {cast} = memref.cast {storage} : {static} to {dynamic}',
                      f'  {value} = mdrt.from_buffer {cast} : {dynamic} to {result_type}'])
        return value, result_type

    x, h, q, moduli = inputs(tilted)
    positions, position_type = buffer(x, 'field')
    charges, charge_type = buffer(q, 'field')
    table, table_type = buffer(moduli, 'table')
    lengths = [number(h[i, i]) for i in range(3)]
    cell = name()
    if tilted:
        lengths.extend(number(h[i, j]) for i, j in ((1, 0), (2, 0), (2, 1)))
    kind = 'triclinic' if tilted else 'orthorhombic'
    lines.append(f'  {cell} = md.{kind}_cell {", ".join(lengths)}')
    energy, force, virial, potential = (name() for _ in range(4))
    if mixed:
        subset = ''
        if listed:
            indices = name()
            ids = '!md.field<@atoms, i32>'
            lines.append(f'  {indices} = md_exec.nonzero {charges} : {charge_type} -> {ids}')
            subset = f' listed({indices} : {ids}) gather_all'
        lines.append(f'  {energy}, {virial}, {force} = md_exec.reciprocal {positions}, {charges}, {cell}, {table} grid({list(GRID)}) order(4) beta({BETA}) coulomb({COULOMB}){subset} : {position_type}, {charge_type}, {table_type} -> f64, vector<9xf64>, {force_type}')
    else:
        lines.append(f'  {energy}, {force}, {virial}, {potential} = md.call @evaluate({positions}, {charges}, {cell}, {table}) : ({position_type}, {charge_type}, !md.cell, {table_type}) -> (f64, {position_type}, vector<9xf64>, {charge_type})')
    if not force_only:
        lines.append(f'  func.call @mdrtPrintFFTValue({energy}) : (f64) -> ()')
    for value, field_type, components in ((force, force_type, 3), (potential, charge_type, 1)):
        if components == 1 and force_only:
            continue
        if components == 1:
            for i in range(9):
                scalar = name()
                lines.extend([f'  {scalar} = vector.extract {virial}[{i}] : f64 from vector<9xf64>',
                              f'  func.call @mdrtPrintFFTValue({scalar}) : (f64) -> ()'])
        element = force_element if components == 3 else 'f64'
        if mixed and components == 1:
            continue
        ty = 'memref<?' + ('x3' if components == 3 else '') + f'x{element}>'
        view = name()
        lines.append(f'  {view} = mdrt.to_buffer {value} : {field_type} to {ty}')
        for row in range(3):
            for component in range(components):
                indices = [number(row, 'index')]
                if components != 1:
                    indices.append(number(component, 'index'))
                scalar = name()
                lines.append(f'  {scalar} = memref.load {view}[{", ".join(indices)}] : {ty}')
                if element == 'f32':
                    wide = name()
                    lines.append(f'  {wide} = arith.extf {scalar} : f32 to f64')
                    scalar = wide
                lines.append(f'  func.call @mdrtPrintFFTValue({scalar}) : (f64) -> ()')
    lines.extend(['  return', '}'])
    print('\n'.join(lines))


def check(evaluator=False):
    actual = np.asarray([float(word) for word in sys.stdin.read().split()])
    wanted = expected(evaluator)
    assert actual.shape == wanted.shape, (actual.shape, wanted.shape)
    errors = np.abs(actual-wanted)
    bounds = VALUE_ATOL+VALUE_RTOL*np.abs(wanted)
    bounds[10:19] = STRAIN_ATOL
    assert np.all(errors <= bounds), [(i, float(actual[i]), float(wanted[i]), float(errors[i]))
                                     for i in np.flatnonzero(errors > bounds)]
    print(f'Gaussian grid: energy {actual[0]:.12g}; energy error {errors[0]:.9g}; '
          f'force error {errors[1:10].max():.9g}; strain FD error {errors[10:19].max():.9g}; '
          f'charge error {errors[19:].max():.9g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['emit', 'check'])
    parser.add_argument('--evaluator', action='store_true',
                        help='check the non-unit-weighted potential evaluator')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--listed', action='store_true')
    parser.add_argument('--force-only', action='store_true')
    parser.add_argument('--grid', nargs=3, type=int)
    args = parser.parse_args()
    if args.grid:
        GRID = reference.GRID = tuple(args.grid)
    emit(args.evaluator, args.mixed, args.force_only, args.listed) if args.command == 'emit' else check(args.evaluator)
