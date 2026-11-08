"""Independent f64 periodic coverage oracle for the combined bin search.

References enumerate every unit pair and all 27 images, independently of
bins, chunks and masks. Both cells have fractional reach below one, so
these are all covering images. Odd/128 reaches avoid rounding-boundary
ties on the dyadic grid. Candidate membership is compared exactly.
"""
import argparse
import itertools
import json
import pathlib
import struct

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
generate = sub.add_parser('generate')
generate.add_argument('program', type=pathlib.Path)
generate.add_argument('reference', type=pathlib.Path)
generate.add_argument('--triclinic', action='store_true')
check = sub.add_parser('check')
check.add_argument('records', type=pathlib.Path)
check.add_argument('reference', type=pathlib.Path)
args = parser.parse_args()


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def sorted_units(points, waters):
    def key(i):
        x, y, z = (f32(v) for v in points[i])
        cx, cy = min(int(f32(3 * x)), 2), min(int(f32(2 * y)), 1)
        column = 2 * cx + (1 - cy if cx % 2 else cy)
        return (i >= waters, column, min(int(f32((1 << 21) * z)), (1 << 21) - 1), i)
    return sorted(range(len(points)), key=key)


def expected_pairs(points, waters, atoms, tilted):
    exclusions = {(0, 1), (2, 3), (0, 16)}
    pairs = set()
    reach = (135 / 128, 117 / 128, 99 / 128)
    minimum_gap = 100.0
    for tag in range(3):
        if tag == 0:
            indices = itertools.combinations(range(waters), 2)
        elif tag == 1:
            indices = itertools.product(range(waters, waters + atoms), range(waters))
        else:
            indices = ((waters + i, waters + j) for i, j in itertools.combinations(range(atoms), 2)
                       if (i, j) not in exclusions)
        for i, j in indices:
            for image in itertools.product((-1, 0, 1), repeat=3):
                d = [points[i][k] - (points[j][k] + image[k]) for k in range(3)]
                cart = ((2 * d[0] + d[1]) + .5 * d[2], 2 * d[1] + d[2], 2 * d[2]) if tilted else tuple(2 * v for v in d)
                r2 = sum(v * v for v in cart)
                minimum_gap = min(minimum_gap, abs(r2 - reach[tag] ** 2))
                if r2 <= reach[tag] ** 2:
                    pairs.add((tag, i, j, *image))
    assert minimum_gap > 1e-8, minimum_gap
    return sorted(pairs), minimum_gap


if args.command == 'check':
    reference = json.loads(args.reference.read_text())
    words = iter(int(x) for x in args.records.read_text().split())
    waters, atoms = reference['waters'], reference['atoms']
    water_rows, atom_rows = (waters + 15) // 16, (atoms + 15) // 16
    for phase, ref in enumerate(reference['phases']):
        assert next(words) == 917031 and next(words) == phase
        n = next(words)
        assert n == len(ref['order']), (phase, 'order length', n)
        order = [next(words) for _ in range(n)]
        assert order == ref['order'], (phase, 'order', order, ref['order'])
        row_count = next(words)
        assert row_count == water_rows + 2 * atom_rows
        actual = set()
        duplicates = []
        entries = chunks = 0
        for row in range(row_count):
            tag, first, begin, count = (next(words) for _ in range(4))
            expected_tag = 0 if row < water_rows else 1 if row < water_rows + atom_rows else 2
            class_row = row if tag == 0 else row - water_rows if tag == 1 else row - water_rows - atom_rows
            expected_first = 16 * class_row + (0 if tag == 0 else 16 * water_rows)
            assert (tag, first, begin) == (expected_tag, expected_first, entries), (phase, row, tag, first, begin)
            assert 0 <= count <= 100000
            entries += count
            chunks += (count + 63) // 64
            for _ in range(count):
                j, code, mask, reserved = (next(words) for _ in range(4))
                assert 0 <= code < 1 << 30 and 0 < mask < 1 << 16 and reserved == 0
                image = tuple(((code >> (10 * k)) & 1023) - 512 for k in range(3))
                assert (0 <= j < waters) if tag != 2 else (waters <= j < waters + atoms)
                for lane in range(16):
                    if not mask & (1 << lane):
                        continue
                    i = order[first + lane]
                    assert i >= 0, (phase, row, 'padded I bit', lane)
                    assert (i < waters) if tag == 0 else (waters <= i < waters + atoms)
                    a, b, shift = i, j, image
                    if tag != 1 and a > b:
                        a, b, shift = b, a, tuple(-v for v in shift)
                    pair = (tag, a, b, *shift)
                    if pair in actual:
                        duplicates.append(pair)
                    actual.add(pair)
        actual_chunks, flags, actual_entries = (next(words) for _ in range(3))
        assert (actual_chunks, flags, actual_entries) == (chunks, 0, entries)
        assert next(words) == 917032
        wanted = {tuple(p) for p in ref['pairs']}
        missing, extra = sorted(wanted - actual), sorted(actual - wanted)
        if missing or extra or duplicates:
            raise AssertionError({'phase': phase, 'wanted': len(wanted), 'actual': len(actual),
                                  'missing': len(missing), 'extra': len(extra), 'duplicates': len(duplicates),
                                  'first_missing': missing[:5], 'first_extra': extra[:5], 'first_duplicate': duplicates[:5]})
        counts = [sum(pair[0] == tag for pair in wanted) for tag in range(3)]
        print(f'phase {phase}: WW/SW/SS {counts}; missing 0; extra 0; duplicates 0; entries {entries}; chunks {chunks}')
    assert next(words, None) is None, 'unexpected trailing records'
    raise SystemExit(0)

waters, atoms = 19, 17
size, particles = waters + atoms, 4 * waters + atoms
points = [[((i * 7) % 32) / 32, ((i * 11) % 16) / 16, ((i * 5) % 8) / 8] for i in range(size)]
points[3] = points[2][:]
points[5] = points[2][:]
points[6][0] = 1 - 2 ** -30
points[22][1] = points[22][2] = 1 - 2 ** -30
phases = []
water_places = ((waters + 15) // 16) * 16
order = [-1] * (water_places + ((atoms + 15) // 16) * 16)
for phase, active in enumerate([(True, True), (False, True), (True, False), (False, False)]):
    if phase == 1:
        for i in range(waters, size):
            points[i][0] = (points[i][0] + 1 / 8) % 1
            points[i][2] = (points[i][2] + 1 / 16) % 1
    if phase == 2:
        for i in range(waters):
            points[i][0] = (points[i][0] + 1 / 16) % 1
            points[i][1] = (points[i][1] + 1 / 4) % 1
    sorted_ids = sorted_units(points, waters)
    for kind, start, count in [(0, 0, waters), (1, water_places, atoms)]:
        if active[kind]:
            padded = water_places if kind == 0 else len(order) - water_places
            order[start:start + padded] = [-1] * padded
            offset = 0 if kind == 0 else waters
            order[start:start + count] = sorted_ids[offset:offset + count]
    pairs, gap = expected_pairs(points, waters, atoms, args.triclinic)
    phases.append({'active': active, 'points': [row[:] for row in points], 'order': order[:], 'pairs': pairs, 'minimum_r2_gap': gap})
cell = [2., 2., 2., 1., .5, 1.] if args.triclinic else [2., 2., 2.]
args.reference.write_text(json.dumps({'waters': waters, 'atoms': atoms, 'cell': cell, 'phases': phases}))

lines = ['md.particle_set @atoms', '!list = !mdrt.segmented_pairs<@atoms>']
lines.append('memref.global "private" constant @initial_controls : memref<4xi64> = dense<0>')
members = [[3 * w, 3 * w + 1, 3 * w + 2, 3 * waters + w] for w in range(waters)]
world_atom = 4 * waters
excluded = [[world_atom + a, world_atom + b] for a, b in [(0, 1), (2, 3), (0, 16)]]
lines.append(f'memref.global "private" constant @water : memref<{waters}x4xi32> = dense<{members}>')
lines.append(f'memref.global "private" constant @excluded : memref<3x2xi32> = dense<{excluded}>')
for phase, ref in enumerate(phases):
    lines.append(f'memref.global "private" constant @points_{phase} : memref<{size}x3xf64> = dense<{ref["points"]}>')
cell_type = f'vector<{len(cell)}xf64>'
lines.extend([
    f'func.func private @search_seed(!list, memref<?x3xf64, 1>, memref<?xi32, 1>, {cell_type}, i1, i1)',
    'func.func private @compact_seed(!list)',
    'func.func private @mdrtTemplatePairsBuffer(!list) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsArenaView(!list, i64) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsOrder(!list) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsCounters(!list) -> memref<4xi64, 1> attributes {llvm.emit_c_interface}',
    'func.func private @mdrtTemplatePairsLayoutValue(!list, i64) -> i64',
    'func.func private @mdrtTemplatePairsGrow(!list, i64) -> i64',
    'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
    'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
    'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
    'func.func private @printI64(i64)', 'func.func private @printNewline()'])
lines.append(r'''
func.func @show(%word: i32) {
  %wide = arith.extsi %word : i32 to i64
  func.call @printI64(%wide) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
func.func @download(%device: memref<?xi32, 1>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %n = memref.dim %device, %c0 : memref<?xi32, 1>
  %host = memref.alloc(%n) : memref<?xi32>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%done]
  return %host : memref<?xi32>
}
func.func @controls(%list: !list) -> memref<4xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %s3 = arith.constant 3 : i64
  %zero = arith.constant 0 : i64
  %device = func.call @mdrtTemplatePairsCounters(%list) : (!list) -> memref<4xi64, 1>
  %host = memref.alloc() : memref<4xi64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<4xi64>, memref<4xi64, 1>
  gpu.wait [%done]
  %flags = memref.load %host[%c1] : memref<4xi64>
  %good = arith.cmpi eq, %flags, %zero : i64
  cf.assert %good, "unexpected search build flag"
  %entries = memref.load %host[%c2] : memref<4xi64>
  %capacity = func.call @mdrtTemplatePairsLayoutValue(%list, %s3) : (!list, i64) -> i64
  %fits = arith.cmpi ule, %entries, %capacity : i64
  cf.assert %fits, "search fixture preallocation is insufficient"
  return %host : memref<4xi64>
}
func.func @dump(%list: !list, %phase: i64, %counter: memref<4xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %s3 = arith.constant 3 : i64
  %s5 = arith.constant 5 : i64
  %s9 = arith.constant 9 : i64
  %marker = arith.constant 917031 : i64
  func.call @printI64(%marker) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printI64(%phase) : (i64) -> ()
  func.call @printNewline() : () -> ()
  %od = func.call @mdrtTemplatePairsOrder(%list) : (!list) -> memref<?xi32, 1>
  %rd = func.call @mdrtTemplatePairsArenaView(%list, %s5) : (!list, i64) -> memref<?xi32, 1>
  %ed = func.call @mdrtTemplatePairsArenaView(%list, %s9) : (!list, i64) -> memref<?xi32, 1>
  %order = func.call @download(%od) : (memref<?xi32, 1>) -> memref<?xi32>
  %rows = func.call @download(%rd) : (memref<?xi32, 1>) -> memref<?xi32>
  %entries = func.call @download(%ed) : (memref<?xi32, 1>) -> memref<?xi32>
  %n = memref.dim %order, %c0 : memref<?xi32>
  %nw = arith.index_cast %n : index to i64
  func.call @printI64(%nw) : (i64) -> ()
  func.call @printNewline() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    %value = memref.load %order[%i] : memref<?xi32>
    func.call @show(%value) : (i32) -> ()
  }
  %row_words = memref.dim %rows, %c0 : memref<?xi32>
  %nr = arith.divui %row_words, %c4 : index
  %nrw = arith.index_cast %nr : index to i64
  func.call @printI64(%nrw) : (i64) -> ()
  func.call @printNewline() : () -> ()
  %capacity = func.call @mdrtTemplatePairsLayoutValue(%list, %s3) : (!list, i64) -> i64
  scf.for %row = %c0 to %nr step %c1 {
    %base = arith.muli %row, %c4 : index
    scf.for %column = %c0 to %c4 step %c1 {
      %at = arith.addi %base, %column : index
      %value = memref.load %rows[%at] : memref<?xi32>
      func.call @show(%value) : (i32) -> ()
    }
    %p2 = arith.addi %base, %c2 : index
    %p3 = arith.addi %base, %c3 : index
    %first32 = memref.load %rows[%p2] : memref<?xi32>
    %count32 = memref.load %rows[%p3] : memref<?xi32>
    %first64 = arith.extui %first32 : i32 to i64
    %count64 = arith.extui %count32 : i32 to i64
    %end64 = arith.addi %first64, %count64 : i64
    %fits = arith.cmpi ule, %end64, %capacity : i64
    cf.assert %fits, "search row exceeds arena capacity"
    %first = arith.index_castui %first32 : i32 to index
    %count = arith.index_castui %count32 : i32 to index
    %end = arith.addi %first, %count : index
    scf.for %entry = %first to %end step %c1 {
      %eb = arith.muli %entry, %c4 : index
      scf.for %column = %c0 to %c4 step %c1 {
        %at = arith.addi %eb, %column : index
        %value = memref.load %entries[%at] : memref<?xi32>
        func.call @show(%value) : (i32) -> ()
      }
    }
  }
  scf.for %i = %c0 to %c3 step %c1 {
    %value = memref.load %counter[%i] : memref<4xi64>
    func.call @printI64(%value) : (i64) -> ()
    func.call @printNewline() : () -> ()
  }
  %end_marker = arith.constant 917032 : i64
  func.call @printI64(%end_marker) : (i64) -> ()
  func.call @printNewline() : () -> ()
  memref.dealloc %order : memref<?xi32>
  memref.dealloc %rows : memref<?xi32>
  memref.dealloc %entries : memref<?xi32>
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


n = emit(f'arith.constant {particles} : index')
units = emit(f'arith.constant {size} : index')
types = emit(f'gpu.alloc({n}) : memref<?xi32, 1>')
device_points = emit(f'gpu.alloc({units}) : memref<?x3xf64, 1>')
inverse = emit(f'gpu.alloc({units}) : memref<?xi32, 1>')
w = emit(f'memref.get_global @water : memref<{waters}x4xi32>')
w = emit(f'memref.cast {w} : memref<{waters}x4xi32> to memref<?x4xi32>')
ex = emit('memref.get_global @excluded : memref<3x2xi32>')
ex = emit(f'memref.cast {ex} : memref<3x2xi32> to memref<?x2xi32>')
state = emit(f'md_exec.empty_segmented_pairs {types}, {w} exclude({ex} : memref<?x2xi32>) cutoff(0.7734375) skin(0.0) extent(0.140625) alpha(0.75) : memref<?xi32, 1>, memref<?x4xi32> -> !list')
required = emit(f'arith.constant {size * size * 27} : i64')
emit(f'func.call @mdrtTemplatePairsGrow({state}, {required}) : (!list, i64) -> i64')
controls_device = emit(f'func.call @mdrtTemplatePairsCounters({state}) : (!list) -> memref<4xi64, 1>')
controls_host = emit('memref.get_global @initial_controls : memref<4xi64>')
ready = emit('gpu.wait async')
done = emit(f'gpu.memcpy async [{ready}] {controls_device}, {controls_host} : memref<4xi64, 1>, memref<4xi64>')
lines.append(f'  gpu.wait [{done}]')
cell_value = emit(f'arith.constant dense<{cell}> : {cell_type}')
for phase, ref in enumerate(phases):
    host = emit(f'memref.get_global @points_{phase} : memref<{size}x3xf64>')
    host = emit(f'memref.cast {host} : memref<{size}x3xf64> to memref<?x3xf64>')
    ready = emit('gpu.wait async')
    done = emit(f'gpu.memcpy async [{ready}] {device_points}, {host} : memref<?x3xf64, 1>, memref<?x3xf64>')
    lines.append(f'  gpu.wait [{done}]')
    wa, aa = (emit(f'arith.constant {str(flag).lower()}') for flag in ref['active'])
    lines.append(f'  func.call @search_seed({state}, {device_points}, {inverse}, {cell_value}, {wa}, {aa}) : (!list, memref<?x3xf64, 1>, memref<?xi32, 1>, {cell_type}, i1, i1) -> ()')
    controls = emit(f'func.call @controls({state}) : (!list) -> memref<4xi64>')
    lines.append(f'  func.call @compact_seed({state}) : (!list) -> ()')
    phase_value = emit(f'arith.constant {phase} : i64')
    lines.append(f'  func.call @dump({state}, {phase_value}, {controls}) : (!list, i64, memref<4xi64>) -> ()')
    lines.append(f'  memref.dealloc {controls} : memref<4xi64>')
lines.extend(['  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()', '  return', '}'])
args.program.write_text('\n'.join(lines) + '\n')
print('reference:', 'triclinic' if args.triclinic else 'orthorhombic',
      [len(phase['pairs']) for phase in phases], 'minimum r2 gap', min(phase['minimum_r2_gap'] for phase in phases))
