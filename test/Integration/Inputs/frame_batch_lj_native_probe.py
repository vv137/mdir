"""Exercise the production mapped-update and LJ constant IR constructors."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import frame_batch_tail_probe as oracle


def emit(path):
    # Reuse the independent actual host FrameEvaluator oracle, not its GPU
    # arithmetic emitter. The program under test comes from the C++ builder.
    with contextlib.redirect_stdout(io.StringIO()):
        oracle.emit(path)
    samples = json.loads(Path(path).read_text())
    print('module attributes {gpu.container_module} {')
    print(subprocess.check_output([sys.argv[3],
                                  '--emit-frame-batch-lj-functions'], text=True))
    print('func.func private @printI64(i64)')
    print('func.func private @printNewline()')
    print('func.func @main() {')
    for name, value in [('zero', 0), ('one', 1), ('two', 2), ('five', 5),
                        ('six', 6), ('eight', 8), ('sixteen', 16)]:
        print(f'%{name} = arith.constant {value} : index')
    print('%host = memref.alloc(%eight) : memref<?x2xf64>')
    for i, sample in enumerate(samples):
        print(f'%i{i} = arith.constant {i} : index')
        for column, key in enumerate(['sigma', 'epsilon']):
            print(f'%x{i}_{column} = arith.constant {sample[key]:.17e} : f64')
            print(f'memref.store %x{i}_{column}, %host[%i{i}, %{["zero", "one"][column]}] : memref<?x2xf64>')
    buffers = {
        'input': ('memref<?x2xf64, 1>', '%eight'),
        'theta': ('memref<?xf64, 1>', '%two'),
        'sigma': ('memref<?xf64, 1>', '%one'),
        'epsilon': ('memref<?xf64, 1>', '%one'),
        'excluded': ('memref<?xi64, 1>', '%zero'),
        'constants': ('memref<?xf64, 1>', '%five'),
        'output': ('memref<?x6xf64, 1>', '%sixteen'),
    }
    for name, (type_, size) in buffers.items():
        print(f'%{name} = gpu.alloc({size}) : {type_}')
    print('%input_start = gpu.wait async')
    print('%input_copy = gpu.memcpy async [%input_start] %input, %host : memref<?x2xf64, 1>, memref<?x2xf64>')
    print('gpu.wait [%input_copy]')
    print('%sigma_map = func.call @prepare_sigma_map() : () -> memref<?xi64, 1>')
    print('%epsilon_map = func.call @prepare_epsilon_map() : () -> memref<?xi64, 1>')
    print('%counts = func.call @prepare_counts() : () -> memref<?xf64, 1>')
    print('scf.for %repeat = %zero to %two step %one {')
    print('  scf.for %case = %zero to %eight step %one {')
    print('    gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one) threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one) {')
    for column, index in enumerate(['zero', 'one']):
        print(f'      %v{column} = memref.load %input[%case, %{index}] : memref<?x2xf64, 1>')
        print(f'      memref.store %v{column}, %theta[%{index}] : memref<?xf64, 1>')
    print('      gpu.terminator\n    }')
    for name in ['sigma', 'epsilon']:
        print(f'    func.call @mapped_f64(%theta, %{name}_map, %{name}) : '
              '(memref<?xf64, 1>, memref<?xi64, 1>, memref<?xf64, 1>) -> ()')
    print('    func.call @lj_constants(%sigma, %epsilon, %counts, %excluded, %constants) : '
          '(memref<?xf64, 1>, memref<?xf64, 1>, memref<?xf64, 1>, '
          'memref<?xi64, 1>, memref<?xf64, 1>) -> ()')
    print('    %base = arith.muli %repeat, %eight : index')
    print('    %row = arith.addi %base, %case : index')
    print('    gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one) threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one) {')
    for j in range(5):
        print(f'      %j{j} = arith.constant {j} : index')
        print(f'      %v{j} = memref.load %constants[%j{j}] : memref<?xf64, 1>')
        print(f'      memref.store %v{j}, %output[%row, %j{j}] : memref<?x6xf64, 1>')
    print('      %run = arith.addf %v0, %v1 : f64')
    print('      %shifted = arith.addf %run, %v2 : f64')
    print('      %energy = arith.addf %shifted, %v3 : f64')
    print('      memref.store %energy, %output[%row, %five] : memref<?x6xf64, 1>')
    print('      gpu.terminator\n    }\n  }\n}')
    print('%result = memref.alloc(%sixteen) : memref<?x6xf64>')
    print('%result_start = gpu.wait async')
    print('%result_copy = gpu.memcpy async [%result_start] %result, %output : memref<?x6xf64>, memref<?x6xf64, 1>')
    print('gpu.wait [%result_copy]')
    print('scf.for %i = %zero to %sixteen step %one {')
    print('  scf.for %j = %zero to %six step %one {')
    print('    %value = memref.load %result[%i, %j] : memref<?x6xf64>')
    print('    %bits = arith.bitcast %value : f64 to i64')
    print('    func.call @printI64(%bits) : (i64) -> ()')
    print('    func.call @printNewline() : () -> ()\n  }\n}')
    buffers.update(sigma_map=('memref<?xi64, 1>', None),
                   epsilon_map=('memref<?xi64, 1>', None),
                   counts=('memref<?xf64, 1>', None))
    for name, (device, _) in buffers.items():
        host = device.replace(', 1>', '>')
        print(f'%p_{name} = memref.memory_space_cast %{name} : {device} to {host}')
        print(f'gpu.dealloc %p_{name} : {host}')
    print('memref.dealloc %result : memref<?x6xf64>')
    print('memref.dealloc %host : memref<?x2xf64>')
    print('return\n}\n}')


def check(path):
    samples = json.loads(Path(path).read_text())
    values = [int(value) for value in sys.stdin.read().split()]
    if len(values) != 96:
        raise RuntimeError(f'expected 96 diagnostic values, got {len(values)}')
    repeated = values[:48] == values[48:]
    measured = []
    for i, sample in enumerate(samples):
        raw = values[i * 6:(i + 1) * 6]
        actual = [struct.unpack('d', struct.pack('q', value))[0] for value in raw]
        reference = sample['reference']
        relative = abs(actual[-1] - reference) / abs(reference) if reference else abs(actual[-1])
        tail, fixed = sample['trace']['tail'], sample['trace']['fixed']
        expected = [tail, 0.0, tail, fixed, 6.0 * tail, reference]
        names = ['dispersion', 'fixed', 'shift', 'fixed_shift', 'virial', 'energy']
        differences = []
        for name, want, got, got_bits in zip(names, expected, actual, raw):
            if oracle.bits(want) != got_bits:
                differences.append(dict(operation=name, reference=want, device=got,
                                        ulps=abs(oracle.bits(want) - got_bits)))
        measured.append(dict(sigma=sample['sigma'], reference=reference, device=actual[-1],
                             relative=relative, ulps=abs(raw[-1] - sample['bits']),
                             differences=differences))
    failures = [sample for sample in measured if not sample['relative'] <= 1e-14]
    print(json.dumps(dict(cases=8, relative_tolerance=1e-14,
                          max_relative=max(x['relative'] for x in measured),
                          max_ulps=max(x['ulps'] for x in measured),
                          repeated_device_bitwise_identical=repeated,
                          measured=measured, failures=failures), indent=2))
    raise SystemExit(bool(failures) or not repeated)


if __name__ == '__main__':
    (emit if sys.argv[1] == 'emit' else check)(sys.argv[2])
