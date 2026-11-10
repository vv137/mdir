"""Generate compact-source, exact-integer GPU mask-stream fixture; no packages."""
import argparse
import json
from pathlib import Path
import random


def cases():
    rng = random.Random(795)
    result = []
    # Every basis independently in each half, including high-bit poison.
    for half in (0, 16):
        for j in range(16):
            for i in range(16):
                m = [rng.getrandbits(16) << 16 for _ in range(32)]
                m[half+j] |= 1 << i
                result.append((m, 32))
    for pattern in (0, 65535, 0x5555, 0xaaaa, 0x8001):
        result.append(([pattern | (rng.getrandbits(16)<<16) for _ in range(32)], 32))
    result.extend([([65535]*16+[0]*16,32), ([0]*16+[65535]*16,32)])
    for _ in range(256):
        result.append(([rng.getrandbits(32) for _ in range(32)], 32))
    # Padded safe-load input is deliberately nonzero; caller clears it.
    for length in range(33):
        for _ in range(4):
            result.append(([rng.getrandbits(32) | 0xffff0001 for _ in range(32)], length))
    return result


def reference(masks, length):
    return [sum((((masks[(lane & 16) + (((lane & 15)-t)&15)]
                   if (lane & 16) + (((lane & 15)-t)&15) < length else 0)
                  >> (lane & 15)) & 1) << t for t in range(16))
            for lane in range(32)]


def signed(x):
    return x if x < (1 << 31) else x-(1 << 32)


def generate(destination):
    data = cases()
    n = len(data)
    words = f'memref<{n}x32xi32>'
    lengths = f'memref<{n}xi32>'
    masks = [[signed(x) for x in row] for row, _ in data]
    refs = [reference(row, length) for row, length in data]
    dense = lambda x: json.dumps(x, separators=(',', ':'))
    lines = ['module {',
             f'  memref.global "private" constant @masks : {words} = dense<{dense(masks)}>',
             f'  memref.global "private" constant @lengths : {lengths} = dense<{dense([length for _,length in data])}>',
             f'  memref.global "private" constant @reference : {words} = dense<{dense(refs)}>',
             f'  func.func private @mask_stream_probe({words}, {lengths}, {words})',
             '  func.func private @printI64(i64)',
             '  func.func private @printNewline()',
             '  func.func @main() {',
             f'    %m = memref.get_global @masks : {words}',
             f'    %l = memref.get_global @lengths : {lengths}',
             f'    %r = memref.get_global @reference : {words}',
             f'    %o = memref.alloc() : {words}',
             f'    func.call @mask_stream_probe(%m, %l, %o) : ({words}, {lengths}, {words}) -> ()',
             '    %zero = arith.constant 0 : index',
             '    %one = arith.constant 1 : index',
             f'    %rows = arith.constant {n} : index',
             '    %lanes = arith.constant 32 : index',
             '    %true = arith.constant true',
             '    %good = scf.for %i = %zero to %rows step %one iter_args(%all = %true) -> (i1) {',
             '      %row = scf.for %j = %zero to %lanes step %one iter_args(%prior = %all) -> (i1) {',
             f'        %got = memref.load %o[%i,%j] : {words}',
             f'        %want = memref.load %r[%i,%j] : {words}',
             '        %equal = arith.cmpi eq, %got, %want : i32',
             '        %next = arith.andi %prior, %equal : i1',
             '        scf.yield %next : i1', '      }',
             '      scf.yield %row : i1', '    }',
             '    %flag = arith.extui %good : i1 to i64',
             '    func.call @printI64(%flag) : (i64) -> ()',
             '    func.call @printNewline() : () -> ()',
             '    cf.assert %good, "template mask stream oracle mismatch"',
             f'    memref.dealloc %o : {words}', '    return', '  }', '}']
    destination.write_text('\n'.join(lines)+'\n')
    return n


def selfcheck():
    # Independent direct rotation simulation checks the generation oracle.
    total = 0
    for m, length in cases():
        expected = reference(m, length)
        rotating = [x if j < length else 0 for j,x in enumerate(m)]
        streams = [0]*32
        for t in range(16):
            for lane in range(32):
                streams[lane] |= ((rotating[lane] >> (lane&15))&1)<<t
            rotating = [rotating[(lane&16)+(((lane&15)-1)&15)] for lane in range(32)]
        assert streams == expected
        total += 32
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        checked = selfcheck()
        print(f'CPU oracle exact equality: {checked} lane streams; failures=0')
    n = generate(args.destination)
    print(f'Generated {n} independent 32-lane cases')
