"""Independent lexicographic spatial-order oracle: 996 exact integer checks.

Four activity combinations check keys, z keys, bin counts/starts, sorted
items, padded order and inverse places. Repeated points test fixed-ID ties;
coordinates just below one test f32 narrowing and the final-bin clamp.
"""
import argparse
import pathlib
import struct
import sys

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--large', action='store_true')
args = parser.parse_args()
waters, atoms, nx, ny = (257, 259, 7, 8) if args.large else (19, 21, 3, 2)
size = waters + atoms
points = [[((i * 7) % 32) / 32, ((i * 11) % 16) / 16, ((i * 5) % 8) / 8]
          for i in range(size)]
points[3] = points[2][:]
points[5] = points[2][:]
points[6][0] = 1 - 2 ** -30
points[22][1] = points[22][2] = 1 - 2 ** -30


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def coordinate(x, bins):
    return min(int(f32(f32(x) * f32(bins))), bins - 1)


keys, zkeys = [], []
for i, (x, y, z) in enumerate(points):
    cx, cy = coordinate(x, nx), coordinate(y, ny)
    column = cx * ny + (ny - 1 - cy if cx % 2 else cy)
    keys.append(column + (nx * ny if i >= waters else 0))
    zkeys.append(coordinate(z, 1 << 21))
items = sorted(range(size), key=lambda i: (keys[i], zkeys[i], i))
counts = [keys.count(k) for k in range(2 * nx * ny)]
starts = [0]
for count in counts:
    starts.append(starts[-1] + count)
water_places = ((waters + 15) // 16) * 16
places = water_places + ((atoms + 15) // 16) * 16
seed_order = [1000 + i for i in range(places)]
seed_inverse = [2000 + i for i in range(size)]

shapes = [f'{size}x3xf64', f'{size}xi32', f'{size}xi32', f'{len(counts)}xi32',
          f'{len(starts)}xi32', f'{len(counts)}xi32', f'{size}xi32', '1xi32',
          f'{places}xi32', f'{size}xi32', f'{4 * size}xi32']
device_types = [f'memref<{shape}, 1>' for shape in shapes]
lines = []


def global_array(name, data, shape=None):
    shape = shape or f'{len(data)}xi32'
    lines.append(f'memref.global "private" constant @{name} : memref<{shape}> = dense<{data}>')


global_array('points', points, shapes[0])
global_array('seed_order', seed_order)
global_array('seed_inverse', seed_inverse)
cases = [(True, True), (True, False), (False, True), (False, False)]
for case, (wa, aa) in enumerate(cases):
    order, inverse = seed_order[:], seed_inverse[:]
    if wa:
        order[:water_places] = [-1] * water_places
    if aa:
        order[water_places:] = [-1] * (places - water_places)
    for rank, unit in enumerate(items):
        water = rank < waters
        if (wa if water else aa):
            place = rank if water else water_places + rank - waters
            order[place] = unit
            inverse[unit] = place
    for name, data in [('keys', keys), ('zkeys', zkeys), ('counts', counts),
                       ('starts', starts), ('items', items), ('order', order), ('inverse', inverse)]:
        global_array(f'{name}_{case}', data)
lines.extend([
    'func.func private @spatial_probe(' + ', '.join(device_types + ['index'] * 4 + ['i1'] * 2) + ')',
    'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
    'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
    'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
    'func.func private @printI64(i64)', 'func.func private @printNewline()'])
lines.append(r'''
func.func @expect(%device: memref<?xi32, 1>, %ref: memref<?xi32>, %label: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %n = memref.dim %device, %c0 : memref<?xi32, 1>
  %nr = memref.dim %ref, %c0 : memref<?xi32>
  %size_okay = arith.cmpi eq, %n, %nr : index
  cf.assert %size_okay, "spatial oracle dimension differs"
  %host = memref.alloc(%n) : memref<?xi32>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%done]
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%good = %true) -> (i1) {
    %got = memref.load %host[%i] : memref<?xi32>
    %expected = memref.load %ref[%i] : memref<?xi32>
    %same = arith.cmpi eq, %got, %expected : i32
    %bad = arith.cmpi ne, %got, %expected : i32
    scf.if %bad {
      %index = arith.index_cast %i : index to i64
      %actual = arith.extsi %got : i32 to i64
      %wanted = arith.extsi %expected : i32 to i64
      func.call @printI64(%label) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%index) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%actual) : (i64) -> ()
      func.call @printNewline() : () -> ()
      func.call @printI64(%wanted) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }
    %next = arith.andi %good, %same : i1
    scf.yield %next : i1
  }
  memref.dealloc %host : memref<?xi32>
  cf.assert %all, "template spatial order oracle mismatch"
  return
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
''')
serial = 0


def emit(expression):
    global serial
    name = f'%v{serial}'
    serial += 1
    lines.append(f'  {name} = {expression}')
    return name


buffers = [emit(f'gpu.alloc() : {t}') for t in device_types]


def upload(index, global_name):
    shape = shapes[index]
    host = emit(f'memref.get_global @{global_name} : memref<{shape}>')
    ready = emit('gpu.wait async')
    done = emit(f'gpu.memcpy async [{ready}] {buffers[index]}, {host} : {device_types[index]}, memref<{shape}>')
    lines.append(f'  gpu.wait [{done}]')


upload(0, 'points')
dimensions = [emit(f'arith.constant {n} : index') for n in (waters, atoms, nx, ny)]
for case, active in enumerate(cases):
    upload(8, 'seed_order')
    upload(9, 'seed_inverse')
    flags = [emit(f'arith.constant {str(flag).lower()}') for flag in active]
    lines.append('  func.call @spatial_probe(' + ', '.join(buffers + dimensions + flags) + ') : (' +
                 ', '.join(device_types + ['index'] * 4 + ['i1'] * 2) + ') -> ()')
    for check, (slot, name, length) in enumerate([(1, 'keys', size), (2, 'zkeys', size),
            (3, 'counts', len(counts)), (4, 'starts', len(starts)), (6, 'items', size),
            (8, 'order', places), (9, 'inverse', size)]):
        view = emit(f'memref.cast {buffers[slot]} : {device_types[slot]} to memref<?xi32, 1>')
        ref = emit(f'memref.get_global @{name}_{case} : memref<{length}xi32>')
        ref = emit(f'memref.cast {ref} : memref<{length}xi32> to memref<?xi32>')
        label = emit(f'arith.constant {10 * case + check} : i64')
        lines.append(f'  func.call @expect({view}, {ref}, {label}) : (memref<?xi32, 1>, memref<?xi32>, i64) -> ()')
one = emit('arith.constant 1 : i64')
lines.extend([f'  func.call @printI64({one}) : (i64) -> ()',
              '  func.call @printNewline() : () -> ()',
              '  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()',
              '  return', '}'])
pathlib.Path(args.output).write_text('\n'.join(lines) + '\n')
