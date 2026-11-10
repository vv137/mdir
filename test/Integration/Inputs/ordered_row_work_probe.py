"""Direct bounded-window probes with an independent dense-stream expectation.

One record is kept for each of the last K jobs (lane0 only). Expected output
is that simple ordered sequence, not a simulation of the emitter's ballots,
shared queue, barriers, or append implementation. All allocated fixture
words are initialized, including unused capacity and overflow-skipped tails.
"""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
gen = sub.add_parser('generate')
gen.add_argument('program', type=pathlib.Path)
gen.add_argument('reference', type=pathlib.Path)
check = sub.add_parser('check')
check.add_argument('records', type=pathlib.Path)
check.add_argument('reference', type=pathlib.Path)
args = parser.parse_args()

if args.command == 'check':
    cases = json.loads(args.reference.read_text())
    words = iter(int(v) for v in args.records.read_text().split())
    for index, case in enumerate(cases):
        jobs, kept, frame_break, capacity = case
        assert (next(words), next(words)) == (198714, index)
        chunks = (kept + 63) // 64
        expected_controls = [chunks, 2 if chunks > capacity else 0, kept, 0]
        actual_controls = [next(words) for _ in range(4)]
        assert actual_controls == expected_controls, (case, 'controls', actual_controls, expected_controls)
        assert next(words) == kept, (case, 'row count')
        unit_words = [next(words) for _ in range(12)]
        entry_words = [next(words) for _ in range(768)]
        expected_units = [0] * 12
        expected_entries = [0] * 768
        dense = [[job, job, 3 * job + 7, -11 - job] for job in range(jobs - kept, jobs)]
        for chunk in range(min(chunks, capacity)):
            records = dense[64 * chunk:64 * (chunk + 1)]
            expected_units[4 * chunk:4 * chunk + 4] = [0, len(records), chunk, 0]
            for member, record in enumerate(records):
                start = 4 * (64 * chunk + member)
                expected_entries[start:start + 4] = record
        assert unit_words == expected_units, (case, 'chunk metadata', unit_words, expected_units)
        assert entry_words == expected_entries, (case, 'ordered payload/tail initialization')
        assert next(words) == 198715
    assert next(words, None) is None, 'unexpected trailing records'
    print('ordered row primitive:', len(cases), 'cases; exact controls, row counts, chunk ordinals and ordered payload')
    raise SystemExit(0)

cases = [[jobs, kept, -1, 3] for jobs in [0, 1, 127, 128, 129]
         for kept in [0, 1, 31, 32, 63, 64, 65] if kept <= jobs]
cases += [[129, 129, 63, 3], [129, 65, -1, 1], [129, 65, -1, 0]]
args.reference.write_text(json.dumps(cases, indent=2))
lines = [
    'func.func private @ordered_row_probe(memref<?xi32, 1>, memref<?xi32, 1>, memref<4xi64, 1>, memref<?xi32, 1>, i64, index, index, index)',
    'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
    'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
    'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
    'func.func private @printI64(i64)',
    'func.func private @printNewline()',
    '''func.func @show(%v: i64) {
  func.call @printI64(%v) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}
func.func @dump32(%p: memref<?xi32>, %n: index) {
  %z = arith.constant 0 : index
  %o = arith.constant 1 : index
  scf.for %i = %z to %n step %o {
    %v = memref.load %p[%i] : memref<?xi32>
    %w = arith.extsi %v : i32 to i64
    func.call @show(%w) : (i64) -> ()
  }
  return
}
func.func @main() {''']
sequence = 0

def emit(expr):
    global sequence
    sequence += 1
    name = f'%v{sequence}'
    lines.append(f'  {name} = {expr}')
    return name

z = emit('arith.constant 0 : index')
o = emit('arith.constant 1 : index')
nunits = emit('arith.constant 12 : index')
nentries = emit('arith.constant 768 : index')
four = emit('arith.constant 4 : index')
zero32 = emit('arith.constant 0 : i32')
zero64 = emit('arith.constant 0 : i64')
activation = emit('func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr')
lines.append(f'  func.call @mdrtDeviceActivationEnter({activation}) : (!llvm.ptr) -> ()')
units = emit(f'gpu.alloc({nunits}) : memref<?xi32, 1>')
entries = emit(f'gpu.alloc({nentries}) : memref<?xi32, 1>')
controls = emit('gpu.alloc() : memref<4xi64, 1>')
counts = emit(f'gpu.alloc({o}) : memref<?xi32, 1>')
hunits = emit(f'memref.alloc({nunits}) : memref<?xi32>')
hentries = emit(f'memref.alloc({nentries}) : memref<?xi32>')
hcontrols = emit('memref.alloc() : memref<4xi64>')
hcounts = emit(f'memref.alloc({o}) : memref<?xi32>')

def transfer(to, source, to_type, from_type):
    ready = emit('gpu.wait async')
    copied = emit(f'gpu.memcpy async [{ready}] {to}, {source} : {to_type}, {from_type}')
    lines.append(f'  gpu.wait [{copied}]')

for index, (njobs, kept, frame_break, capacity) in enumerate(cases):
    for host, n, typ, value in [(hunits, nunits, 'memref<?xi32>', zero32),
                                (hentries, nentries, 'memref<?xi32>', zero32),
                                (hcounts, o, 'memref<?xi32>', zero32),
                                (hcontrols, four, 'memref<4xi64>', zero64)]:
        lines += [f'  scf.for %i = {z} to {n} step {o} {{',
                  f'    memref.store {value}, {host}[%i] : {typ}', '  }']
    transfer(units, hunits, 'memref<?xi32, 1>', 'memref<?xi32>')
    transfer(entries, hentries, 'memref<?xi32, 1>', 'memref<?xi32>')
    transfer(counts, hcounts, 'memref<?xi32, 1>', 'memref<?xi32>')
    transfer(controls, hcontrols, 'memref<4xi64, 1>', 'memref<4xi64>')
    cap = emit(f'arith.constant {capacity} : i64')
    jobs = emit(f'arith.constant {njobs} : index')
    target = emit(f'arith.constant {kept} : index')
    # -1 can never match the nonnegative producer cursor.
    boundary = emit(f'arith.constant {frame_break} : index')
    lines.append(f'  func.call @ordered_row_probe({units}, {entries}, {controls}, {counts}, {cap}, {jobs}, {target}, {boundary}) : (memref<?xi32, 1>, memref<?xi32, 1>, memref<4xi64, 1>, memref<?xi32, 1>, i64, index, index, index) -> ()')
    transfer(hunits, units, 'memref<?xi32>', 'memref<?xi32, 1>')
    transfer(hentries, entries, 'memref<?xi32>', 'memref<?xi32, 1>')
    transfer(hcounts, counts, 'memref<?xi32>', 'memref<?xi32, 1>')
    transfer(hcontrols, controls, 'memref<4xi64>', 'memref<4xi64, 1>')
    marker = emit('arith.constant 198714 : i64')
    case = emit(f'arith.constant {index} : i64')
    lines += [f'  func.call @show({marker}) : (i64) -> ()', f'  func.call @show({case}) : (i64) -> ()',
              f'  scf.for %i = {z} to {four} step {o} {{']
    v = emit(f'memref.load {hcontrols}[%i] : memref<4xi64>')
    lines += [f'    func.call @show({v}) : (i64) -> ()', '  }',
              f'  func.call @dump32({hcounts}, {o}) : (memref<?xi32>, index) -> ()',
              f'  func.call @dump32({hunits}, {nunits}) : (memref<?xi32>, index) -> ()',
              f'  func.call @dump32({hentries}, {nentries}) : (memref<?xi32>, index) -> ()']
    end = emit('arith.constant 198715 : i64')
    lines.append(f'  func.call @show({end}) : (i64) -> ()')
# Like the existing search fixture, device allocations have the one-process
# activation lifetime. The standard GPU-to-LLVM test pipeline's explicit
# gpu.dealloc cannot free address-space1 memrefs (mgpuMemFree expects ptr0).
for v, typ in [(hunits, 'memref<?xi32>'), (hentries, 'memref<?xi32>'),
               (hcontrols, 'memref<4xi64>'), (hcounts, 'memref<?xi32>')]:
    lines.append(f'  memref.dealloc {v} : {typ}')
lines += [f'  func.call @mdrtDeviceActivationClose({activation}) : (!llvm.ptr) -> ()', '  return', '}']
args.program.write_text('\n'.join(lines) + '\n')
print('ordered row primitive fixture:', len(cases), 'cases')
