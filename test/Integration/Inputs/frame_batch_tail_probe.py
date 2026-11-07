"""Arithmetic preflight for native device updates of the LJ tail energy.

Two identical neutral particles in a cubic 4 nm cell, 2 nm apart, with a
1 nm cutoff and no truncation shift have zero direct pair energy. Their
energy is the dispersion correction plus the frame evaluator's D210 shift
estimate. The oracle is the existing host FrameEvaluator. The analytic
diagnostic follows the scalar order of the corresponding host constants.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

SIGMAS = [0.15, 0.20, 0.25, 0.30, 0.3150752406575124,
          0.3399669508423535, 0.35, 0.40]
EPSILON = 0.6363864
TRACE = ["power", "factor", "coefficient", "sum", "mean", "a", "b", "c", "tail", "n4", "n4pi", "sphere", "ratio", "shift_factor", "fixed_factor", "fixed", "scaled", "energy"]

def bits(value):
    return struct.unpack('q', struct.pack('d', value))[0]

def real(value):
    text = format(value, '.17g')
    return text if '.' in text or 'e' in text else text + '.0'

def emit(path):
    import mdir
    import numpy as np
    directory = Path(str(path) + '.model')
    directory.mkdir(exist_ok=True)
    (directory / 'system.top').write_text(
        '[ defaults ]\n1 2 no 1.0 1.0\n[ atomtypes ]\n'
        'A 0 40.0 0.0 A 0.300000 0.6363864\n'
        '[ moleculetype ]\nA 1\n[ atoms ]\n1 A 1 A A 1 0.0 40.0\n'
        '[ system ]\ntwo neutral particles\n[ molecules ]\nA 2\n')
    atoms = [f'{i:5d}{"A":<5s}{"A":>5s}{i:5d}{x:8.3f}{0.0:8.3f}{0.0:8.3f}'
             for i, x in [(1, 0.0), (2, 2.0)]]
    (directory / 'system.gro').write_text(
        '\n'.join(['two neutral particles', '2', *atoms, '4.0 4.0 4.0']) + '\n')
    loaded = mdir.load_gromacs(str(directory / 'system.top'), str(directory / 'system.gro'))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 1.0, 1.15, 1.0
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.EnergyPressure
    system.tunables = [mdir.Tunable('sigma', 'sigma_pair'),
                       mdir.Tunable('epsilon', 'epsilon_pair')]
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target, execution.precision = mdir.Target.CPU, mdir.Precision.Double
    execution.deterministic = True
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                           execution, mdir.Schedule())
    evaluator = mdir.FrameEvaluator(program)
    samples = []
    for sigma in SIGMAS:
        coefficient = 4.0 * EPSILON * math.pow(sigma, 6.0)
        pair_sum = 2.0 * coefficient
        mean = pair_sum / 2.0
        tail = -2.0 * math.pi / (3.0 * 64.0) * 2.0 * 2.0 * mean / 1.0
        fixed_factor = (1.0 - 64.0 / (2.0 * 4.0 * math.pi / 3.0)) - 1.0
        energy = (tail + tail) + tail * fixed_factor
        trace = [math.pow(sigma, 6.0), 4.0 * EPSILON, coefficient,
                 pair_sum, mean, -2.0 * math.pi / 192.0,
                 (-2.0 * math.pi / 192.0) * 2.0,
                 (-2.0 * math.pi / 192.0) * 2.0 * 2.0, tail,
                 8.0, 8.0 * math.pi, 8.0 * math.pi / 3.0,
                 64.0 / (8.0 * math.pi / 3.0),
                 1.0 - 64.0 / (8.0 * math.pi / 3.0), fixed_factor,
                 tail * fixed_factor, tail + tail, energy]
        evaluator.tunables.update({'sigma': np.array([sigma]),
                                   'epsilon': np.array([EPSILON])})
        observed = float(evaluator.evaluate(np.array([[[0.0, 0.0, 0.0],
                                                       [2.0, 0.0, 0.0]]])).energy[0])
        samples.append({'sigma': sigma, 'epsilon': EPSILON,
                        'reference': observed, 'bits': bits(observed),
                        'trace': dict(zip(TRACE, trace)), 'analytic': energy, 'analytic_difference': energy - observed})
    with open(path, 'w') as file:
        json.dump(samples, file, indent=2)
    print('module attributes {gpu.container_module} {')
    print('func.func private @printI64(i64)')
    print('func.func private @printNewline()')
    print('func.func @main() {')
    print('  %c0 = arith.constant 0 : index')
    print('  %c1 = arith.constant 1 : index')
    print('  %n = arith.constant 8 : index')
    print('  %input = memref.alloc(%n) : memref<?x2xf64>')
    print('  %repeats = arith.constant 2 : index')
    print('  %rows = arith.constant 16 : index')
    print('  %result = memref.alloc(%rows) : memref<?x18xf64>')
    for i, sample in enumerate(samples):
        print(f'  %i{i} = arith.constant {i} : index')
        print(f'  %s{i} = arith.constant {real(sample["sigma"])} : f64')
        print(f'  %e{i} = arith.constant {real(sample["epsilon"])} : f64')
        print(f'  memref.store %s{i}, %input[%i{i}, %c0] : memref<?x2xf64>')
        print(f'  memref.store %e{i}, %input[%i{i}, %c1] : memref<?x2xf64>')
    print('  %device = gpu.alloc (%n) : memref<?x2xf64, 1>')
    print('  %output = gpu.alloc (%rows) : memref<?x18xf64, 1>')
    print('  %start = gpu.wait async')
    print('  %copy = gpu.memcpy async [%start] %device, %input : memref<?x2xf64, 1>, memref<?x2xf64>')
    print('  gpu.wait [%copy]')
    print('  scf.for %repeat = %c0 to %repeats step %c1 {')
    print('  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {')
    print('    %sigma = memref.load %device[%tx, %c0] : memref<?x2xf64, 1>')
    print('    %epsilon = memref.load %device[%tx, %c1] : memref<?x2xf64, 1>')
    for name, value in [('six', 6.0), ('four', 4.0), ('two', 2.0),
                        ('denominator', 192.0), ('minus_two_pi', -2.0 * math.pi),
                        ('one', 1.0), ('volume', 64.0), ('three', 3.0), ('pi', math.pi)]:
        print(f'    %{name} = arith.constant {real(value)} : f64')
    print('    %sigma2 = arith.mulf %sigma, %sigma : f64')
    print('    %sigma4 = arith.mulf %sigma2, %sigma2 : f64')
    print('    %power = arith.mulf %sigma2, %sigma4 : f64')
    print('    %row_base = arith.muli %repeat, %n : index')
    print('    %row = arith.addi %row_base, %tx : index')
    print('    %factor = arith.mulf %four, %epsilon : f64')
    print('    %coefficient = arith.mulf %factor, %power : f64')
    print('    %sum = arith.mulf %two, %coefficient : f64')
    print('    %mean = arith.divf %sum, %two : f64')
    print('    %a = arith.divf %minus_two_pi, %denominator : f64')
    print('    %b = arith.mulf %a, %two : f64')
    print('    %c = arith.mulf %b, %two : f64')
    print('    %tail = arith.mulf %c, %mean : f64')
    print('    %n4 = arith.mulf %two, %four : f64')
    print('    %n4pi = arith.mulf %n4, %pi : f64')
    print('    %sphere = arith.divf %n4pi, %three : f64')
    print('    %ratio = arith.divf %volume, %sphere : f64')
    print('    %shift_factor = arith.subf %one, %ratio : f64')
    print('    %fixed_factor = arith.subf %shift_factor, %one : f64')
    print('    %fixed = arith.mulf %tail, %fixed_factor : f64')
    print('    %scaled = arith.addf %tail, %tail : f64')
    print('    %energy = arith.addf %scaled, %fixed : f64')
    for j, name in enumerate(TRACE):
        print(f'    %trace{j} = arith.constant {j} : index')
        print(f'    memref.store %{name}, %output[%row, %trace{j}] : memref<?x18xf64, 1>')
    print('    gpu.terminator')
    print('  }')
    print('  }')
    print('  %end = gpu.wait async')
    print('  %read = gpu.memcpy async [%end] %result, %output : memref<?x18xf64>, memref<?x18xf64, 1>')
    print('  gpu.wait [%read]')
    print('  scf.for %i = %c0 to %rows step %c1 {')
    print('    %trace_count = arith.constant 18 : index')
    print('    scf.for %j = %c0 to %trace_count step %c1 {')
    print('    %value = memref.load %result[%i, %j] : memref<?x18xf64>')
    print('    %bits = arith.bitcast %value : f64 to i64')
    print('    func.call @printI64(%bits) : (i64) -> ()')
    print('    func.call @printNewline() : () -> ()')
    print('  }')
    print('  }')
    print('  %plain_device = memref.memory_space_cast %device : memref<?x2xf64, 1> to memref<?x2xf64>')
    print('  %plain_output = memref.memory_space_cast %output : memref<?x18xf64, 1> to memref<?x18xf64>')
    print('  gpu.dealloc %plain_device : memref<?x2xf64>')
    print('  gpu.dealloc %plain_output : memref<?x18xf64>')
    print('  memref.dealloc %input : memref<?x2xf64>')
    print('  memref.dealloc %result : memref<?x18xf64>')
    print('  return')
    print('}')
    print('}')

def check(path):
    with open(path) as file:
        samples = json.load(file)
    actual = [int(line) for line in sys.stdin.read().split()]
    if len(actual) != 2 * len(samples) * len(TRACE):
        raise RuntimeError(f'expected {len(samples)} energy results, got {len(actual)}')
    repeated_identical = actual[:len(actual)//2] == actual[len(actual)//2:]
    mismatches = []
    measured = []
    for i, sample in enumerate(samples):
        trace_bits = actual[i * len(TRACE):(i + 1) * len(TRACE)]
        differences = []
        for name, observed in zip(TRACE, trace_bits):
            reference = sample['trace'][name]
            if observed != bits(reference):
                differences.append({'operation': name, 'reference': reference,
                                    'device': struct.unpack('d', struct.pack('q', observed))[0],
                                    'ulps': abs(observed - bits(reference))})
        sample['diverging_operations'] = differences
        sample['first_diverging_operation'] = differences[0] if differences else None
        got = trace_bits[-1]
        value = struct.unpack('d', struct.pack('q', got))[0]
        relative = abs(value - sample['reference']) / abs(sample['reference']) if sample['reference'] else abs(value)
        measured.append(dict(sigma=sample['sigma'], reference=sample['reference'], device=value, relative=relative, ulps=abs(got-sample['bits'])))
        if relative > 1e-14:
            mismatches.append(dict(sample, actual=value,
                                   difference=value - sample['reference'],
                                   ulps=abs(got - sample['bits'])))
    result = {'cases': len(samples), 'bitwise_identical': sum(x['ulps'] == 0 for x in measured),
              'max_relative': max(x['relative'] for x in measured),
              'max_ulps': max(x['ulps'] for x in measured),
              'repeated_device_bitwise_identical': repeated_identical, 'measured': measured,
              'mismatches': mismatches, 'relative_tolerance': 1e-14}
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(bool(mismatches) or not repeated_identical)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['emit', 'check'])
    parser.add_argument('reference')
    args = parser.parse_args()
    (emit if args.command == 'emit' else check)(args.reference)
