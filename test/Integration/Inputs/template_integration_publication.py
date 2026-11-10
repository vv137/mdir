"""R1 exact publication fixture: identity constraints, final SSA, tail/across.

Generate only; the test declares fused/reference executions before any GPU
run. The marked tuple set is exactly the real OHH membership, including
nonconsecutive original members in the across-warp case.
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
# Check independent original energy values; epoch bits are compared directly
# between fused and unfused runs rather than changing their expected values.
for slot in [4, 5, 6]:
    s = s.replace(f'  func.call @dump_epoch(%view{slot}) : (memref<?x3xf64, 1>) -> ()', '')
a.program.write_text(s)
a.reference.write_text(json.dumps(json.loads(a.reference.read_text())[:7]) + '\n')
