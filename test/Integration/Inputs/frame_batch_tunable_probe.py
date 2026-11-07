"""Device helper oracle: tied maps, preserved sites and ordered contractions."""
import json
from pathlib import Path
import struct
import subprocess
import sys


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def bits(value):
    return struct.unpack('q', struct.pack('d', value))[0]


def emit(reference):
    n = 137
    first = [0.125 + i / 7 for i in range(9)]
    second = [-0.25 - i / 11 for i in range(9)]
    mapping = [-1 if i % 5 == 0 else (i * 7) % 9 for i in range(n)]
    initial = [13.25 + i / 16 for i in range(n)]
    source = [0.25 + i / 13 for i in range(51)]
    offsets, indices, weights, expected = [0], [], [], []
    for row in range(19):
        value = 0.0
        for j in range(row % 7):
            index = (row * 3 + j * 11) % len(source)
            weight = 0.5 + j / 17
            indices.append(index)
            weights.append(weight)
            value += source[index] * weight
        offsets.append(len(indices))
        expected.append(value)
    cases = []
    for phase, theta in enumerate([first, second, second]):
        for wide in [False, True]:
            for i, slot in enumerate(mapping):
                value = initial[i] if slot < 0 else theta[slot]
                cases.append(dict(kind='map', phase=phase, wide=wide, site=i,
                                  reference=value if wide else f32(value), tolerance=0))
        for row, value in enumerate(expected):
            cases.append(dict(kind='contraction', phase=phase, row=row,
                              reference=value, tolerance=1e-14))
    Path(reference).write_text(json.dumps(cases, indent=2))
    print('module attributes {gpu.container_module} {')
    print(subprocess.check_output([sys.argv[3],
                                  '--emit-frame-batch-tunable-functions'], text=True))
    print('func.func private @printI64(i64)')
    print('func.func private @printNewline()')
    print('func.func @main() {')
    print('%zero = arith.constant 0 : index')
    print('%one = arith.constant 1 : index')

    transfer_number = 0

    def transfer(destination, source, device_type, host_type):
        nonlocal transfer_number
        tag = transfer_number
        transfer_number += 1
        print(f'%start{tag} = gpu.wait async')
        print(f'%copy{tag} = gpu.memcpy async [%start{tag}] {destination}, {source} : {device_type}, {host_type}')
        print(f'gpu.wait [%copy{tag}]')

    def array(name, values, dtype):
        count = len(values)
        print(f'%n_{name} = arith.constant {count} : index')
        host = f'memref<?x{dtype}>'
        device = f'memref<?x{dtype}, 1>'
        if name == 'map':
            print('%d_map = func.call @prepare_map() : () -> memref<?xi64, 1>')
            return host, device
        print(f'%h_{name} = memref.alloc(%n_{name}) : {host}')
        for i, value in enumerate(values):
            text = str(int(value)) if dtype == 'i64' else format(value, '.17e')
            print(f'%i_{name}_{i} = arith.constant {i} : index')
            print(f'%v_{name}_{i} = arith.constant {text} : {dtype}')
            print(f'memref.store %v_{name}_{i}, %h_{name}[%i_{name}_{i}] : {host}')
        print(f'%d_{name} = gpu.alloc(%n_{name}) : {device}')
        transfer(f'%d_{name}', f'%h_{name}', device, host)
        return host, device

    buffers = {}
    for name, values, dtype in [('theta', first, 'f64'), ('map', mapping, 'i64'),
                              ('narrow', initial, 'f32'), ('wide', initial, 'f64'),
                              ('source', source, 'f64'), ('offsets', offsets, 'i64'),
                              ('indices', indices, 'i64'), ('weights', weights, 'f64'),
                              ('contracted', [0.0] * 19, 'f64')]:
        buffers[name] = array(name, values, dtype)

    for phase in range(3):
        if phase == 1:
            for i, value in enumerate(second):
                print(f'%theta2_{i} = arith.constant {value:.17e} : f64')
                print(f'memref.store %theta2_{i}, %h_theta[%i_theta_{i}] : memref<?xf64>')
            transfer('%d_theta', '%h_theta', 'memref<?xf64, 1>', 'memref<?xf64>')
        for name, dtype in [('narrow', 'f32'), ('wide', 'f64')]:
            print(f'func.call @mapped_{dtype}(%d_theta, %d_map, %d_{name}) : '
                  f'(memref<?xf64, 1>, memref<?xi64, 1>, memref<?x{dtype}, 1>) -> ()')
        print('func.call @contract(%d_source, %d_offsets, %d_indices, %d_weights, %d_contracted) : '
              '(memref<?xf64, 1>, memref<?xi64, 1>, memref<?xi64, 1>, '
              'memref<?xf64, 1>, memref<?xf64, 1>) -> ()')
        for name in ['narrow', 'wide', 'contracted']:
            host, device = buffers[name]
            transfer(f'%h_{name}', f'%d_{name}', host, device)
            print(f'scf.for %i = %zero to %n_{name} step %one {{')
            print(f'%raw = memref.load %h_{name}[%i] : {host}')
            if name == 'narrow':
                print('%value = arith.extf %raw : f32 to f64')
                print('%number = arith.bitcast %value : f64 to i64')
            else:
                print('%number = arith.bitcast %raw : f64 to i64')
            print('func.call @printI64(%number) : (i64) -> ()')
            print('func.call @printNewline() : () -> ()')
            print('}')
    for name, (host, device) in buffers.items():
        print(f'%p_{name} = memref.memory_space_cast %d_{name} : {device} to {host}')
        print(f'gpu.dealloc %p_{name} : {host}')
        if name != 'map':
            print(f'memref.dealloc %h_{name} : {host}')
    print('return\n}\n}')


def check(reference):
    cases = json.loads(Path(reference).read_text())
    observed = [int(value) for value in sys.stdin.read().split()]
    if len(observed) != len(cases):
        raise RuntimeError(f'expected {len(cases)} results, got {len(observed)}')
    failures, maximum = [], 0.0
    for case, got in zip(cases, observed):
        value = struct.unpack('d', struct.pack('q', got))[0]
        expected = case['reference']
        relative = abs(value - expected) / abs(expected) if expected else abs(value)
        if case['kind'] == 'map':
            failed = got != bits(expected)
        else:
            maximum = max(maximum, relative)
            failed = not relative <= case['tolerance']
        if failed:
            failures.append(dict(case, device=value, relative=relative))
    stride = len(cases) // 3
    deterministic = observed[stride:2 * stride] == observed[2 * stride:]
    print(json.dumps(dict(cases=len(cases), mismatches=failures,
                          max_contraction_relative=maximum,
                          repeated_device_bitwise_identical=deterministic), indent=2))
    raise SystemExit(bool(failures) or not deterministic)


if __name__ == '__main__':
    (emit if sys.argv[1] == 'emit' else check)(sys.argv[2])
