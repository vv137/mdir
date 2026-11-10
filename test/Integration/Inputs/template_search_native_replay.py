"""Generate an isolated forced-general replay of a native refresh snapshot.

No decimal round trip of reference, cell, or reach bits is permitted.
The checker compares logical ordered records, never staging addresses.
"""
import argparse
import json
import pathlib
import re


def signed(x):
    x = int(x) & 0xffffffff
    return x if x < 0x80000000 else x - 0x100000000


def logical(snapshot, arena, order):
    layout = snapshot['layout']
    rows = int(layout[2])
    records = []
    for row in range(rows):
        at = int(layout[5]) + 4 * row
        tag, first, begin, count = map(signed, arena[at:at + 4])
        entries = []
        for entry in range(begin, begin + count):
            off = int(layout[9]) + 4 * entry
            entries.append(list(map(signed, arena[off:off + 4])))
        records.append([tag, first, entries])
    # Range offsets are semantic class boundaries, unlike chunk addresses.
    ranges = [list(map(signed, arena[4 + 6 * c:8 + 6 * c])) for c in range(3)]
    return dict(rows=records, ranges=ranges, order=list(map(signed, order)))


def generate(s, destination):
    assert (s['waters'], s['atoms']) == (3, 0), 'replay initially supports exactly three waters and no atoms'
    assert len(s['layout']) == 32 and len(s['cell_bits']) in (3, 6)
    assert len(s['reaches_bits']) == 3 and len(s['outer_reference_bits']) == 3
    lines = ['md.particle_set @atoms', '!list = !mdrt.segmented_pairs<@atoms>']
    serial = 0

    def emit(expr):
        nonlocal serial
        name = f'%r{serial}'
        serial += 1
        lines.append(f'  {name} = {expr}')
        return name

    def const(n, typ='i64'):
        return emit(f'arith.constant {int(n)} : {typ}')

    def bits(n):
        return f'0x{int(n):016X}'

    def global_array(name, values, typ='i32', width=None):
        shape = str(len(values)) + (f'x{width}' if width else '')
        dense = ('[' + ', '.join('[' + ', '.join(bits(v) for v in row) + ']'
                                for row in values) + ']') if width else str(list(map(signed, values)))
        lines.append(f'memref.global "private" constant @{name} : memref<{shape}x{typ}> = dense<{dense}>')
        return f'memref<{shape}x{typ}>'

    globals_ = {}
    for key in ('arena', 'order', 'groupStarts', 'groupLengths', 'groupColumns',
                'atStart', 'keys', 'zKeys', 'places', 'items'):
        assert s[key], f'empty replay array {key}'
        globals_[key] = global_array(key, s[key])
    globals_['refs'] = global_array('refs', s['outer_reference_bits'], 'f64', 3)
    lines.append('memref.global "private" constant @members : memref<3x4xi32> = dense<[[0,1,2,9],[3,4,5,10],[6,7,8,11]]>')
    ct = f'vector<{len(s["cell_bits"])}xf64>'
    dev = 'memref<?xi32, 1>'
    declarations = [
        f'func.func private @search_seed(!list, memref<?x3xf64, 1>, {dev}, {ct}, i1, i1, vector<3xf64>)',
        'func.func private @mdrtTemplatePairsGrow(!list, i64) -> i64',
        'func.func private @mdrtTemplatePairsChooseGrid(!list, i64, i64, i64, i64) -> i64',
        'func.func private @mdrtTemplatePairsLayoutValue(!list, i64) -> i64',
        'func.func private @mdrtTemplatePairsCommitGroups(!list, i64, i64, i64, i64) -> i64',
        'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
        'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
        'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
        'func.func private @printI64(i64)', 'func.func private @printNewline()']
    for name, args, result in [('Buffer', '!list', dev), ('Order', '!list', dev),
                              ('ArenaView', '!list, i64', dev), ('GroupView', '!list, i64', dev),
                              ('Counters', '!list', 'memref<4xi64, 1>')]:
        declarations.append(f'func.func private @mdrtTemplatePairs{name}({args}) -> {result} attributes {{llvm.emit_c_interface}}')
    lines.extend(declarations)
    lines.append('func.func @main() {')
    activation = emit('func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr')
    lines.append(f'  func.call @mdrtDeviceActivationEnter({activation}) : (!llvm.ptr) -> ()')
    n = const(12, 'index')
    types = emit(f'gpu.alloc({n}) : {dev}')
    members = emit('memref.get_global @members : memref<3x4xi32>')
    members = emit(f'memref.cast {members} : memref<3x4xi32> to memref<?x4xi32>')
    owner = emit(f'md_exec.empty_segmented_pairs {types}, {members} cutoff(0.9) skin(0.9) extent(0.1) alpha(0.75) : {dev}, memref<?x4xi32> -> !list')
    capacity = const(s['layout'][3])
    emit(f'func.call @mdrtTemplatePairsGrow({owner}, {capacity}) : (!list, i64) -> i64')
    one, zero = const(1), const(0)
    nx, ny = const(s['columns_x']), const(s['columns_y'])
    emit(f'func.call @mdrtTemplatePairsChooseGrid({owner}, {nx}, {ny}, {one}, {one}) : (!list, i64, i64, i64, i64) -> i64')
    wg, ag = const(s['layout'][27]), const(s['layout'][28])
    emit(f'func.call @mdrtTemplatePairsCommitGroups({owner}, {wg}, {ag}, {one}, {one}) : (!list, i64, i64, i64, i64) -> i64')
    for slot, expected in enumerate(s['layout']):
        value = emit(f'func.call @mdrtTemplatePairsLayoutValue({owner}, {const(slot)}) : (!list, i64) -> i64')
        matches = emit(f'arith.cmpi eq, {value}, {const(expected)} : i64')
        lines.append(f'  cf.assert {matches}, "isolated replay layout mismatch at slot {slot}"')

    def view(name, slot=None):
        args = owner
        signature = '!list'
        if slot is not None:
            args += ', ' + const(slot)
            signature += ', i64'
        return emit(f'func.call @mdrtTemplatePairs{name}({args}) : ({signature}) -> {dev}')

    def upload(key, target):
        host = emit(f'memref.get_global @{key} : {globals_[key]}')
        rank = '?x3xf64' if key == 'refs' else '?xi32'
        ht, dt = f'memref<{rank}>', f'memref<{rank}, 1>'
        host = emit(f'memref.cast {host} : {globals_[key]} to {ht}')
        ready = emit('gpu.wait async')
        done = emit(f'gpu.memcpy async [{ready}] {target}, {host} : {dt}, {ht}')
        lines.append(f'  gpu.wait [{done}]')

    arena = view('Buffer')
    upload('arena', arena)
    order = view('Order')
    upload('order', order)
    for slot, key in enumerate(('groupStarts', 'groupLengths', 'groupColumns', 'atStart')):
        upload(key, view('GroupView', slot))
    views = {}
    for slot, key in ((18, 'keys'), (19, 'zKeys'), (20, 'items'), (21, 'places')):
        views[key] = view('ArenaView', slot)
        upload(key, views[key])
    count = const(3, 'index')
    points = emit(f'gpu.alloc({count}) : memref<?x3xf64, 1>')
    upload('refs', points)
    cell = emit(f'arith.constant dense<[{", ".join(bits(v) for v in s["cell_bits"])}]> : {ct}')
    reaches = emit(f'arith.constant dense<[{", ".join(bits(v) for v in s["reaches_bits"])}]> : vector<3xf64>')
    wa, aa = const(bool(s['water_active']), 'i1'), const(bool(s['atom_active']), 'i1')
    lines.append(f'  func.call @search_seed({owner}, {points}, {views["places"]}, {cell}, {wa}, {aa}, {reaches}) : (!list, memref<?x3xf64, 1>, {dev}, {ct}, i1, i1, vector<3xf64>) -> ()')

    def print_value(value):
        lines.append(f'  func.call @printI64({value}) : (i64) -> ()')
        lines.append('  func.call @printNewline() : () -> ()')

    print_value(const(795841))
    for target, size, element in ((arena, len(s['arena']), 'i32'),
                                  (order, len(s['order']), 'i32'),
                                  (emit(f'func.call @mdrtTemplatePairsCounters({owner}) : (!list) -> memref<4xi64, 1>'), 4, 'i64')):
        ht = 'memref<4xi64>' if element == 'i64' else 'memref<?xi32>'
        dt = 'memref<4xi64, 1>' if element == 'i64' else dev
        host = emit('memref.alloc() : memref<4xi64>') if element == 'i64' else emit(f'memref.alloc({const(size,"index")}) : {ht}')
        ready = emit('gpu.wait async')
        done = emit(f'gpu.memcpy async [{ready}] {host}, {target} : {ht}, {dt}')
        lines.append(f'  gpu.wait [{done}]')
        for i in range(size):
            at = const(i, 'index')
            value = emit(f'memref.load {host}[{at}] : {ht}')
            if element == 'i32':
                value = emit(f'arith.extsi {value} : i32 to i64')
            print_value(value)
        lines.append(f'  memref.dealloc {host} : {ht}')
    print_value(const(795842))
    lines.append(f'  func.call @mdrtDeviceActivationClose({activation}) : (!llvm.ptr) -> ()')
    lines.extend(['  return', '}'])
    destination.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('snapshot', type=pathlib.Path)
    gen.add_argument('program', type=pathlib.Path)
    check = sub.add_parser('check')
    check.add_argument('snapshot', type=pathlib.Path)
    check.add_argument('records', type=pathlib.Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    if args.command == 'generate':
        generate(snapshot, args.program)
        return
    values = [int(x) for x in args.records.read_text().splitlines() if re.fullmatch(r'-?\d+', x.strip())]
    begin = values.index(795841) + 1
    end = values.index(795842, begin)
    values = values[begin:end]
    na, no = len(snapshot['arena']), len(snapshot['order'])
    assert len(values) == na + no + 4, 'incomplete or duplicate replay dump'
    arena, order, counters = values[:na], values[na:na + no], values[-4:]
    assert counters[1] == 0, ('replay flags', counters)
    expected = logical(snapshot, snapshot['arena'], snapshot['order'])
    actual = logical(snapshot, arena, order)
    assert counters[2] == sum(len(row[2]) for row in actual['rows']), ('replay entry counter', counters)
    assert actual == expected, ('ordered native/general replay mismatch', actual, expected)
    print('native refresh/general replay: exact ordered rows, class ranges, order; flags 0')


if __name__ == '__main__':
    main()
