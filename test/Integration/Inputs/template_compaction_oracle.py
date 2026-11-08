"""Exact class-compaction oracle with reversed chunk order and a 65-entry row.

Independent expected arrays are built by concatenating rows, not by the
device chunk-copy algorithm. Padding remains -777. No physical energies
or spatial search are claimed by this integer arena-layout fixture.
"""
import pathlib
import sys

mode = sys.argv[2] if len(sys.argv) == 3 else ''
assert mode in ('', '--append', '--append-overflow', '--append-index-overflow')
append = bool(mode)
overflow = mode == '--append-overflow'
index_overflow = mode == '--append-index-overflow'

counts = [65, 0, 33, 1, 0, 2]
chunks = [(3, 0, 1), (0, 1, 1), (5, 0, 2), (2, 0, 33), (0, 0, 64)]
offsets = [0]
works = [0]
rows = []
units = []
entries = []


def record(row, entry):
    return [1000 + 100 * row + entry, 2000 + entry, 3000 + row, 4000 + entry]


for row, count in enumerate(counts):
    tag = row // 2
    first = 16 * row if tag == 0 else 32 + 16 * (row % 2)
    rows.extend([tag, first, offsets[-1], count])
    for begin in range(0, count, 64):
        units.extend([tag, first, offsets[-1] + begin, offsets[-1] + min(begin + 64, count)])
    for entry in range(count):
        entries.extend(record(row, entry))
    offsets.append(offsets[-1] + count)
    works.append(works[-1] + (count + 63) // 64)
ranges = []
for start, end in [(0, 2), (2, 4), (4, 6)]:
    ranges.extend([start, end, offsets[start], offsets[end], works[start], works[end]])

lines = ['md.particle_set @atoms', '!list = !mdrt.segmented_pairs<@atoms>']
members = [[3 * w, 3 * w + 1, 3 * w + 2, 51 + w] for w in range(17)]
lines.append('memref.global "private" constant @water : memref<17x4xi32> = dense<' + str(members) + '>')
expected = {'rows': rows, 'units': units, 'entries': entries, 'counts': counts,
            'offsets': offsets, 'works': works,
            'work_counts': [(n + 63) // 64 for n in counts], 'ranges': ranges}
total = sum(counts) + ((1 << 31) - 1 if index_overflow else 0)
signed_low = (total & 0xffffffff) - (1 << 32) if total & (1 << 31) else total
expected['controls'] = [5, 0, 2 if overflow else 1 if index_overflow else 0, 0,
                        signed_low, total >> 32, 0, 0]
expected['sentinel'] = [-777]
for name, data in expected.items():
    lines.append(f'memref.global "private" constant @{name} : memref<{len(data)}xi32> = dense<{data}>')
lines.extend(['func.func private @compact_seed(!list)',
              'func.func private @mdrtTemplatePairsBuffer(!list) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}',
              'func.func private @mdrtTemplatePairsArenaView(!list, i64) -> memref<?xi32, 1> attributes {llvm.emit_c_interface}',
              'func.func private @mdrtTemplatePairsLayoutValue(!list, i64) -> i64',
              'func.func private @mdrtTemplatePairsGrow(!list, i64) -> i64',
              'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
              'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
              'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
              'func.func private @printI64(i64)', 'func.func private @printNewline()'])
if append:
    lines.append('func.func private @append_seed(!list)')
lines.append(r'''
func.func @expect(%view: memref<?xi32, 1>, %ref: memref<?xi32>, %start: index, %padding: i1) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %sentinel = arith.constant -777 : i32
  %n = memref.dim %view, %c0 : memref<?xi32, 1>
  %nr = memref.dim %ref, %c0 : memref<?xi32>
  %stop = arith.addi %start, %nr : index
  %fits = arith.cmpi ule, %stop, %n : index
  cf.assert %fits, "compaction oracle view is too short"
  %end = arith.select %padding, %n, %stop : index
  %host = memref.alloc(%n) : memref<?xi32>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %view : memref<?xi32>, memref<?xi32, 1>
  gpu.wait [%done]
  %all = scf.for %i = %start to %end step %c1 iter_args(%good = %true) -> (i1) {
    %offset = arith.subi %i, %start : index
    %inside = arith.cmpi ult, %offset, %nr : index
    %expected = scf.if %inside -> (i32) {
      %value = memref.load %ref[%offset] : memref<?xi32>
      scf.yield %value : i32
    } else {
      scf.yield %sentinel : i32
    }
    %got = memref.load %host[%i] : memref<?xi32>
    %same = arith.cmpi eq, %got, %expected : i32
    %bad = arith.cmpi ne, %got, %expected : i32
    scf.if %bad {
      %index = arith.index_cast %i : index to i64
      %actual = arith.extsi %got : i32 to i64
      %wanted = arith.extsi %expected : i32 to i64
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
  return %all : i1
}
func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c85 = arith.constant 85 : index
  %types = gpu.alloc(%c85) : memref<?xi32, 1>
  %w = memref.get_global @water : memref<17x4xi32>
  %water = memref.cast %w : memref<17x4xi32> to memref<?x4xi32>
  %list = md_exec.empty_segmented_pairs %types, %water cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
  %required = arith.constant 101 : i64
  %grew = func.call @mdrtTemplatePairsGrow(%list, %required) : (!list, i64) -> i64
  %arena = func.call @mdrtTemplatePairsBuffer(%list) : (!list) -> memref<?xi32, 1>
  %n = memref.dim %arena, %c0 : memref<?xi32, 1>
  %host = memref.alloc(%n) : memref<?xi32>
  %sentinel = arith.constant -777 : i32
  scf.for %i = %c0 to %n step %c1 {
    memref.store %sentinel, %host[%i] : memref<?xi32>
  }
''')
serial = 0


def emit(expression):
    global serial
    name = f'%v{serial}'
    serial += 1
    lines.append(f'  {name} = {expression}')
    return name


def index(n):
    return emit(f'arith.constant {n} : index')


def base(slot):
    s = emit(f'arith.constant {slot} : i64')
    offset = emit(f'func.call @mdrtTemplatePairsLayoutValue(%list, {s}) : (!list, i64) -> i64')
    return emit(f'arith.index_cast {offset} : i64 to index')


def store(where, offset, value):
    at = emit(f'arith.addi {where}, {index(offset)} : index')
    number = emit(f'arith.constant {value} : i32')
    lines.append(f'  memref.store {number}, %host[{at}] : memref<?xi32>')


stage_units, stage_entries, row_counts = base(11), base(12), base(13)
if append:
    for control in range(40, 48):
        store('%c0', control, (1 << 31) - 1 if index_overflow and control == 44 else 0)
else:
    store('%c0', 40, len(chunks))
    for chunk, (row, ordinal, count) in enumerate(chunks):
        for column, value in enumerate([row, count, ordinal, 0]):
            store(stage_units, 4 * chunk + column, value)
        for entry in range(count):
            for column, value in enumerate(record(row, 64 * ordinal + entry)):
                store(stage_entries, 4 * (64 * chunk + entry) + column, value)
    for row, count in enumerate(counts):
        store(row_counts, row, count)
lines.extend(['  %ready = gpu.wait async',
              '  %done = gpu.memcpy async [%ready] %arena, %host : memref<?xi32, 1>, memref<?xi32>',
              '  gpu.wait [%done]', '  memref.dealloc %host : memref<?xi32>'])
if append:
    lines.append('  func.call @append_seed(%list) : (!list) -> ()')
if not overflow and not index_overflow:
    lines.append('  func.call @compact_seed(%list) : (!list) -> ()')
good = emit('arith.constant true')
checks = [(5, 'rows', False, 0), (6, 'rows', False, 0),
          (7, 'units', True, 0), (8, 'units', True, 0),
          (9, 'entries', True, 0), (10, 'entries', True, 0),
          (13, 'counts', False, 0), (14, 'offsets', False, 0),
          (15, 'work_counts', False, 0), (16, 'works', False, 0),
          (None, 'ranges', False, 4), (None, 'ranges', False, 22)]
if append:
    checks.append((None, 'controls', False, 40))
if overflow or index_overflow:
    checks = [(13, 'counts', False, 0), (None, 'controls', False, 40)]
if overflow:
    checks += [(11, 'sentinel', True, 8), (12, 'sentinel', True, 512)]
for slot, name, padding, start in checks:
    if slot is None:
        view = '%arena'
    else:
        s = emit(f'arith.constant {slot} : i64')
        view = emit(f'func.call @mdrtTemplatePairsArenaView(%list, {s}) : (!list, i64) -> memref<?xi32, 1>')
    size = len(expected[name])
    ref = emit(f'memref.get_global @{name} : memref<{size}xi32>')
    ref = emit(f'memref.cast {ref} : memref<{size}xi32> to memref<?xi32>')
    padded = emit(f'arith.constant {str(padding).lower()}')
    checked = emit(f'func.call @expect({view}, {ref}, {index(start)}, {padded}) : (memref<?xi32, 1>, memref<?xi32>, index, i1) -> i1')
    good = emit(f'arith.andi {good}, {checked} : i1')
flag = emit(f'arith.extui {good} : i1 to i64')
lines.extend([f'  func.call @printI64({flag}) : (i64) -> ()',
              '  func.call @printNewline() : () -> ()',
              f'  cf.assert {good}, "template class compaction oracle mismatch"',
              '  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()',
              '  return', '}'])
pathlib.Path(sys.argv[1]).write_text('\n'.join(lines) + '\n')
