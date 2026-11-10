"""Independent integer reference for column/gap/rightmost-largest-gap cuts."""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('program', type=pathlib.Path)
parser.add_argument('metadata', type=pathlib.Path)
args = parser.parse_args()

cases = [
    ('empty', 0, [], [], [4, 4], [8, 8]),
    ('one', 1, [0], [7], [0, 0], [0, 0]),
    ('class-boundary', 3, [0] * 5, [2] * 5, [0, 0], [0, 0]),
    ('strict-gap-rightmost-tie', 4, [0] * 4, [0, 4, 8, 12], [4, 4], [8, 8]),
    ('cascade', 65, [0] * 65, [3 * i for i in range(65)], [99, 99], [20, 20]),
    ('multiple-blocks', 257, [(i * 5) % 7 for i in range(516)],
     [(i * 11) % 251 for i in range(516)], [13, 19], [40, 60]),
    ('scan-block-boundary', 1025, [0] * 1025, [7] * 1025, [1000, 1000], [1000, 1000]),
]
lines = []
serial = 0
metadata = []


def emit(expression):
    global serial
    value = f'%v{serial}'
    serial += 1
    lines.append(f'  {value} = {expression}')
    return value


def global_array(name, values):
    lines.append(f'memref.global "private" constant @{name} : memref<{len(values)}xi32> = dense<{values}>')


references = []
for case, (name, first_class, columns, zkeys, gaps, spans) in enumerate(cases):
    n = len(columns)
    m = max(n, 1)
    order = sorted(range(n), key=lambda i: (i >= first_class, columns[i], zkeys[i], i))
    z = [zkeys[i] for i in order]
    flags = {i for i in range(n) if i == 0 or i == first_class or
             columns[order[i]] != columns[order[i - 1]] or
             z[i] - z[i - 1] > gaps[i >= first_class]}
    passes = 0
    while True:
        starts = sorted(flags)
        segments = list(zip(starts, starts[1:] + [n]))
        added = set()
        for start, end in segments:
            for begin in range(start, end, 16):
                last = min(begin + 15, end - 1)
                if z[last] - z[begin] <= spans[begin >= first_class]:
                    continue
                differences = [(z[j] - z[j - 1], j) for j in range(begin + 1, last + 1)]
                added.add(max(differences)[1])
        passes += 1
        if not added:
            break
        assert not (added & flags)
        flags |= added
        assert passes <= n + 1
    ids = [0] * (m + 1)
    lengths = [end - start for start, end in segments]
    counts = [(length + 15) // 16 for length in lengths]
    groups = [sum(count for (start, _), count in zip(segments, counts)
                  if (start >= first_class) == bool(kind)) for kind in range(2)]
    for segment, (start, end) in enumerate(segments):
        ids[start:end] = [segment] * (end - start)
    ids[m] = len(segments)
    padded_counts = counts + [0] * (m - len(counts))
    offsets = [0]
    for count in padded_counts:
        offsets.append(offsets[-1] + count)
    inputs = [order or [0], columns or [0], zkeys or [0]]
    expected = [[int(i in flags) for i in range(m)], ids,
                starts + [0] * (m - len(starts)), lengths + [0] * (m - len(lengths)),
                padded_counts, offsets, [len(segments), *groups, 0]]
    for i, values in enumerate(inputs):
        global_array(f'input_{case}_{i}', values)
    for i, values in enumerate(expected):
        global_array(f'expected_{case}_{i}', values)
    references.append((n, m, first_class, gaps, spans, groups, passes, expected))
    metadata.append(dict(name=name, units=n, groups=groups, segments=len(segments), passes=passes))

buffer_type = 'memref<?xi32, 1>'
probe_types = [buffer_type] * 11 + ['index', 'index'] + ['i32'] * 4
lines += ['func.func private @group_plan_probe(' + ', '.join(probe_types) + ') -> (index, index, index)',
          'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
          'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
          'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
          'func.func private @printI64(i64)', 'func.func private @printNewline()']
lines.append(r'''
func.func @expect(%device: memref<?xi32, 1>, %reference: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %reference, %c0 : memref<?xi32>
  %host = memref.alloc(%n) : memref<?xi32>
  %ready = gpu.wait async
  %copy = gpu.memcpy async [%ready] %host, %device : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%copy]
  scf.for %i = %c0 to %n step %c1 {
    %got = memref.load %host[%i] : memref<?xi32>
    %wanted = memref.load %reference[%i] : memref<?xi32>
    %same = arith.cmpi eq, %got, %wanted : i32
    cf.assert %same, "spatial group planner differs from independent reference"
  }
  memref.dealloc %host : memref<?xi32>
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
''')
for case, (n, m, first, gaps, spans, groups, passes, expected) in enumerate(references):
    sizes = [m, m, m, m, m + 1, m, m, m, m + 1, 4, max((n + 1023) // 1024, 1)]
    buffers = []
    for size in sizes:
        value = emit(f'gpu.alloc() : memref<{size}xi32, 1>')
        buffers.append(emit(f'memref.cast {value} : memref<{size}xi32, 1> to {buffer_type}'))
    for i in range(3):
        host = emit(f'memref.get_global @input_{case}_{i} : memref<{m}xi32>')
        ready = emit('gpu.wait async')
        copied = emit(f'gpu.memcpy async [{ready}] {buffers[i]}, {host} : {buffer_type}, memref<{m}xi32>')
        lines.append(f'  gpu.wait [{copied}]')
    constants = [emit(f'arith.constant {value} : {kind}') for value, kind in
                 [(n, 'index'), (first, 'index')] + [(v, 'i32') for v in gaps + spans]]
    names = [f'%g0_{case}', f'%g1_{case}', f'%passes_{case}']
    lines.append('  ' + ', '.join(names) + ' = func.call @group_plan_probe(' +
                 ', '.join(buffers + constants) + ') : (' + ', '.join(probe_types) + ') -> (index, index, index)')
    for got, wanted in zip(names, groups + [passes]):
        value = emit(f'arith.constant {wanted} : index')
        same = emit(f'arith.cmpi eq, {got}, {value} : index')
        lines.append(f'  cf.assert {same}, "spatial group planner scalar differs"')
    for i, values in enumerate(expected):
        ref = emit(f'memref.get_global @expected_{case}_{i} : memref<{len(values)}xi32>')
        view = emit(f'memref.cast {ref} : memref<{len(values)}xi32> to memref<?xi32>')
        lines.append(f'  func.call @expect({buffers[i + 3]}, {view}) : ({buffer_type}, memref<?xi32>) -> ()')
    # Retain these small allocations for the runner lifetime, as the other
    # device fixtures do. The upstream dealloc lowering rejects addrspace-1
    # pointers before execution; no shared lowering is changed here.
one = emit('arith.constant 1 : i64')
lines += [f'  func.call @printI64({one}) : (i64) -> ()',
          '  func.call @printNewline() : () -> ()',
          '  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()', '  return', '}']
args.program.write_text('\n'.join(lines) + '\n')
args.metadata.write_text(json.dumps(metadata, indent=2) + '\n')
