"""R1 exact publication fixture: identity constraints, final SSA, tail/across.

Generate identical marked input for fused and explicitly unfused executions. The marked tuples include real OHH membership and, in the across-warp case,
an unmatched constrained triplet and nonconsecutive original members.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('program', type=pathlib.Path)
p.add_argument('reference', type=pathlib.Path)
p.add_argument('--snapshot', action='store_true')
p.add_argument('--across', action='store_true')
p.add_argument('--bad-extent', action='store_true')
p.add_argument('--no-proof', action='store_true')
a = p.parse_args()
subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('template_native_oracle.py')),
                'generate', str(a.program), str(a.reference)], check=True)
s = a.program.read_text()
if a.across:
    permutation = [0, 31, 40, 3, 4, 5, 9, 10, 11, 13, 14, 15, 41, 42, 43, 44]
    positions = json.loads(re.search(r'@positions.*?dense<(\[.*\])>', s)[1])
    weights = json.loads(re.search(r'@weights.*?dense<(\[.*\])>', s)[1])
    new_positions = [[10.0, 0.0, 0.0] for _ in range(49)]
    new_weights = [0.0] * 49
    for old, new in enumerate(permutation):
        new_positions[new] = positions[old]
        new_weights[new] = weights[old]
    new_weights[7] = 1.0
    water = [[permutation[q] for q in row] for row in
             [[0, 1, 2, 12], [3, 4, 5, 13], [6, 7, 8, 14], [9, 10, 11, 15]]]
    s = re.sub(r'(@water.*?dense<)\[.*\](>)', lambda m: m[1] + json.dumps(water) + m[2], s)
    s = re.sub(r'(@positions.*?dense<)\[.*\](>)', lambda m: m[1] + json.dumps(new_positions) + m[2], s)
    s = re.sub(r'(@weights.*?dense<)\[.*\](>)', lambda m: m[1] + json.dumps(new_weights) + m[2], s)
    s = s.replace('16x3xf64', '49x3xf64').replace('16xf64', '49xf64')
    s = s.replace('%n = arith.constant 16 : index', '%n = arith.constant 49 : index')
    # This case intentionally has a larger initial capacity; growth itself
    # remains covered by the unchanged native epoch fixture.
    s = s.replace('cf.assert %grew, "native fixture did not exercise automatic arena growth"', '')
# A fresh list widens the initial extent. Normal constrained updates must
# stay within one; the negative case retains the original too-small .2.
if not a.bad_extent:
    s = s.replace('extent(0.2)', 'extent(1.0)')
needle = '  %zero = arith.constant 0.0 : f64'
assert s.count(needle) == 1
s = s.replace(needle, needle + '''
  %pub_c0 = arith.constant 0 : index
  %pub_c1 = arith.constant 1 : index
  %pub_c3 = arith.constant 3 : index
  %pub_n = memref.dim %x, %pub_c0 : memref<?x3xf64, 1>
  %pub_w = memref.dim %members, %pub_c0 : memref<?x4xi32>
  %pub_members = memref.alloc(%pub_w) : memref<?x3xi32>
  scf.for %pub_i = %pub_c0 to %pub_w step %pub_c1 {
    scf.for %pub_j = %pub_c0 to %pub_c3 step %pub_c1 {
      %pub_member = memref.load %members[%pub_i, %pub_j] : memref<?x4xi32>
      memref.store %pub_member, %pub_members[%pub_i, %pub_j] : memref<?x3xi32>
    }
  }
  %pub_incidence = md_exec.build_incidence %pub_members size(%pub_n)
      : memref<?x3xi32> -> memref<?xi32, 1>
  %pub_predictor = gpu.alloc(%pub_n) : memref<?x3xf64, 1>
  %pub_correction = gpu.alloc(%pub_n) : memref<?x3xf64, 1>
  // Correction storage includes unmatched particles read by the final writer.
  // Initialize every component before either fused or standalone execution.
  md_exec.particle_for ins(%x : memref<?x3xf64, 1>)
      outs(%pub_correction : memref<?x3xf64, 1>) {
  ^bb0(%pub_unused: vector<3xf64>):
    %pub_all_zero = arith.constant dense<0.0> : vector<3xf64>
    md_exec.yield %pub_all_zero : vector<3xf64>
  }
  %pub_no = arith.constant false
''')
if a.across:
    # A different constrained triplet is deliberately not in the selected
    # template membership. The native epoch marker applies to the entire
    # constraint set, so this tuple must publish three ordinary atom units.
    s = s.replace('  %pub_members = memref.alloc(%pub_w)',
                  '  %pub_wc = arith.addi %pub_w, %pub_c1 : index\n'
                  '  %pub_members = memref.alloc(%pub_wc)')
    s = s.replace('  %pub_incidence = md_exec.build_incidence', '''
  %pub_atom1 = arith.constant 1 : i32
  %pub_atom2 = arith.constant 7 : i32
  %pub_atom6 = arith.constant 8 : i32
  %pub_c2 = arith.constant 2 : index
  memref.store %pub_atom1, %pub_members[%pub_w, %pub_c0] : memref<?x3xi32>
  memref.store %pub_atom2, %pub_members[%pub_w, %pub_c1] : memref<?x3xi32>
  memref.store %pub_atom6, %pub_members[%pub_w, %pub_c2] : memref<?x3xi32>
  %pub_incidence = md_exec.build_incidence''')
pattern = re.compile(r'  md_exec.particle_for ins\(%x, %weights.*?\n  }', re.S)
writers = list(pattern.finditer(s))
assert len(writers) == 6
for serial, match in reversed(list(enumerate(writers))):
    predictor = match[0].replace('outs(%x :', 'outs(%pub_predictor :')
    marker = '' if a.no_proof else ' {md.template_constraint_epoch}'
    event = f'''
  %pub_failure{serial} = md_exec.tuple_for %pub_incidence, %pub_predictor, %cell{serial+1}
      coordinates(displacement(1, 0))
      outs(%pub_correction : memref<?x3xf64, 1>) reduce(%pub_no : i1)
      arity(3) overwrite [true] disjoint {{
  ^bb0(%pub_d: vector<3xf64>):
    %pub_zero = arith.subf %pub_d, %pub_d : vector<3xf64>
    %pub_false = arith.constant false
    md_exec.yield %pub_zero, %pub_zero, %pub_zero, %pub_false
        : vector<3xf64>, vector<3xf64>, vector<3xf64>, i1
  }}{marker} : memref<?xi32, 1>, memref<?x3xf64, 1> -> i1
  md_exec.particle_for ins(%pub_predictor, %pub_correction : memref<?x3xf64, 1>, memref<?x3xf64, 1>)
      outs(%x : memref<?x3xf64, 1>) {{
  ^bb0(%pub_pred: vector<3xf64>, %pub_delta: vector<3xf64>):
    %pub_final = arith.addf %pub_pred, %pub_delta : vector<3xf64>
    md_exec.yield %pub_final : vector<3xf64>
  }}
'''
    if a.snapshot:
        event += f'  func.call @r1_snapshot(%r{serial}, %x, %pub_failure{serial}) : (!list, memref<?x3xf64, 1>, i1) -> ()\n'
    if a.across and serial == 0:
        event += '  func.call @check_unmatched_publication(%r0) : (!list) -> ()\n'
    s = s[:match.start()] + predictor + event + s[match.end():]
if a.across:
    # Member7 is an unmatched constrained atom, fixed unit7. Its first
    # predictor moves by1/64 in a cell of4; packing cells remain−2, giving
    # the exact current fraction(10+1/64−8)/4 =129/256. Check before refresh
    # can repair stale metadata. This fails the old blanket-water branch.
    check = '''
func.func @check_unmatched_publication(%list: !list) {
  %slot = arith.constant 3 : i64
  %device = func.call @mdrtTemplatePairsFrameState(%list, %slot)
      : (!list, i64) -> memref<?x3xf64, 1>
  %zero = arith.constant 0 : index
  %seven = arith.constant 7 : index
  %n = memref.dim %device, %zero : memref<?x3xf64, 1>
  %host = memref.alloc(%n) : memref<?x3xf64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device
      : memref<?x3xf64>, memref<?x3xf64, 1>
  gpu.wait [%done]
  %actual = memref.load %host[%seven, %zero] : memref<?x3xf64>
  %expected = arith.constant 0.50390625 : f64
  %same = arith.cmpf oeq, %actual, %expected : f64
  cf.assert %same, "unmatched constrained atom lost final SSA publication"
  memref.dealloc %host : memref<?x3xf64>
  return
}
'''
    s += check
# Keep the independent seven-energy oracle. Snapshot mode separately compares
# exact initialized publication state between identical marked executions.
for slot in [4, 5, 6]:
    s = s.replace(f'  func.call @dump_epoch(%view{slot}) : (memref<?x3xf64, 1>) -> ()', '')

if a.snapshot:
    def dump_helper(name, scalar):
        bits = 'i64' if scalar == 'f64' else 'i32'
        widen = '' if bits == 'i64' else '    %wide = arith.extui %bits : i32 to i64\n'
        value = '%bits' if bits == 'i64' else '%wide'
        return f"""
func.func @{name}(%device: memref<?x3x{scalar}, 1>, %rows: index, %cols: index, %tag: i64) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Gather only initialized logical components into dense device storage.
  // A one-row strided subview is rejected by the memcpy lowering, even
  // though its selected row prefix is contiguous. This copy-only kernel
  // preserves every floating-point bit and does not inspect capacity tails.
  %size = arith.muli %rows, %cols : index
  %dense = gpu.alloc(%size) : memref<?x{scalar}, 1>
  %host = memref.alloc(%size) : memref<?x{scalar}>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
      threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {{
    scf.for %k = %c0 to %size step %c1 {{
      %i = arith.divui %k, %cols : index
      %j = arith.remui %k, %cols : index
      %x = memref.load %device[%i, %j] : memref<?x3x{scalar}, 1>
      memref.store %x, %dense[%k] : memref<?x{scalar}, 1>
    }}
    gpu.terminator
  }}
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %dense
      : memref<?x{scalar}>, memref<?x{scalar}, 1>
  gpu.wait [%done]
  %length = arith.index_cast %size : index to i64
  func.call @printI64(%tag) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printI64(%length) : (i64) -> ()
  func.call @printNewline() : () -> ()
  scf.for %k = %c0 to %size step %c1 {{
    %x = memref.load %host[%k] : memref<?x{scalar}>
    %bits = arith.bitcast %x : {scalar} to {bits}
{widen}    func.call @printI64({value}) : (i64) -> ()
    func.call @printNewline() : () -> ()
  }}
  memref.dealloc %host : memref<?x{scalar}>
  return
}}
"""
    # Four matched waters; across has 33 unmatched real atom units.
    units, packed = (37, 45) if a.across else (4, 12)
    helper = """
func.func private @mdrtTemplatePairsFrames(!list) -> memref<?x3xf32, 1> attributes {llvm.emit_c_interface}
func.func @r1_snapshot(%list: !list, %x: memref<?x3xf64, 1>, %failed: i1) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %physical = memref.dim %x, %c0 : memref<?x3xf64, 1>
  %t0 = arith.constant -795100 : i64
  func.call @r1_dump_f64(%x, %physical, %c3, %t0) : (memref<?x3xf64, 1>, index, index, i64) -> ()
"""
    for slot in range(7):
        rows = units if slot < 4 else (6 if a.across else 2) if slot < 6 else 2
        cols = 1 if slot == 6 and not a.across else 3
        helper += f"""
  %slot{slot} = arith.constant {slot} : i64
  %view{slot} = func.call @mdrtTemplatePairsFrameState(%list, %slot{slot}) : (!list, i64) -> memref<?x3xf64, 1>
  %rows{slot} = arith.constant {rows} : index
  %cols{slot} = arith.constant {cols} : index
  %tag{slot} = arith.constant {-795101-slot} : i64
  func.call @r1_dump_f64(%view{slot}, %rows{slot}, %cols{slot}, %tag{slot}) : (memref<?x3xf64, 1>, index, index, i64) -> ()
"""
    helper += f"""
  %packed = func.call @mdrtTemplatePairsFrames(%list) : (!list) -> memref<?x3xf32, 1>
  %np = arith.constant {packed} : index
  %tp = arith.constant -795108 : i64
  func.call @r1_dump_f32(%packed, %np, %c3, %tp) : (memref<?x3xf32, 1>, index, index, i64) -> ()
  %tf = arith.constant -795109 : i64
  %one = arith.constant 1 : i64
  %flag = arith.extui %failed : i1 to i64
  func.call @printI64(%tf) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printI64(%one) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printI64(%flag) : (i64) -> ()
  func.call @printNewline() : () -> ()
  return
}}
"""
    s += dump_helper('r1_dump_f64', 'f64') + dump_helper('r1_dump_f32', 'f32') + helper

a.program.write_text(s)
a.reference.write_text(json.dumps(json.loads(a.reference.read_text())[:7]) + '\n')
